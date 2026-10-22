#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <Utils.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Defined in BinaryOp.cpp; copies a texture payload into a dense buffer, and
// shares a payload that is already dense as it stands.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

// Also defined in BinaryOp.cpp: the element walk every binary op shares, with
// the broadcasting and the two storages already routed.
Tensor binary_op_tensor_dispatch(
    const char* shader_name,
    const char* buffer_shader_name,
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha,
    const char* name);
Tensor binary_op_scalar_dispatch(
    const char* shader_name,
    const char* buffer_shader_name,
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha,
    const char* name);

namespace {

// Invocations owning one row each, so a walk asks for a group of this width.
// The width stays inside what every adapter guarantees a workgroup may hold.
constexpr uint32_t kRowWalkWidth = 128u;

struct RowBlock final {
  int32_t rows;
  int32_t classes;
  // The code to skip, split into the words a wide target stores it in
  int32_t skip_low;
  int32_t skip_high;
  int32_t fill;
  int32_t fill1;
};

struct GradBlock final {
  int32_t rows;
  int32_t classes;
  int32_t skip_low;
  int32_t skip_high;
  // 1: the rows share one gradient, divided by the weight total
  int32_t mean_mode;
  // 1: each row owns an entry of the incoming gradient
  int32_t per_row_grad;
};

// A target keeps one pair of words per element, so the code a caller asks to
// skip arrives split the same way: it is matched word against word rather than
// ordered.
void split_code(int64_t code, int32_t* low, int32_t* high) {
  const uint64_t bits = static_cast<uint64_t>(code);
  *low = static_cast<int32_t>(bits & 0xFFFFFFFFu);
  *high = static_cast<int32_t>(bits >> 32);
}

// The rows a loss walks, the classes it picks between, and whether the loss
// asked for a single unbatched row.
struct LossShape final {
  int64_t rows;
  int64_t classes;
  bool unbatched;
};

LossShape loss_shape(const Tensor& self, const char* name) {
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      std::string("Vulkan ") + name +
          " supports Float32 and Float16 tensors only");
  TP_CHECK(
      self.dim() == 1 || self.dim() == 2,
      std::string("Vulkan ") + name +
          " expects a 1d row of classes or a 2d batch of rows");
  const bool unbatched = self.dim() == 1;
  return LossShape{
      unbatched ? 1 : self.size(0),
      unbatched ? self.size(0) : self.size(1),
      unbatched};
}

// What both passes read: the wide target and the class weights when the caller
// handed some.  The tensors are held here because the memory the passes read
// belongs to them - a payload made by an earlier op returns its memory to the
// pool as soon as the last handle to it drops.
struct LossOperands final {
  Tensor target;
  Tensor weight;
  api::vTensor v_target;
  // Without class weights the passes fold their factor away, and the binding
  // still needs a readable payload of its own; the target is already bound
  // read-only, so naming it twice cannot alias a write.
  api::vTensor v_weight;
  int32_t skip_low = 0;
  int32_t skip_high = 0;
};

LossOperands prepare_operands(
    api::Context* context,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t rows,
    int64_t classes,
    int64_t ignore_index,
    const char* name) {
  const Tensor device_target =
      target.device().is_vulkan() ? target : target.to(vulkan_device());
  const Tensor wide_target = device_target.dtype() == DType::Int64
      ? device_target.contiguous()
      : device_target.to(DType::Int64).contiguous();
  TP_CHECK(
      wide_target.numel() == rows,
      std::string("Vulkan ") + name + ": expected a target of " +
          std::to_string(rows) + " elements, got " +
          std::to_string(wide_target.numel()));

  Tensor device_weight;
  if (weight.has_value() && weight->defined()) {
    TP_CHECK(
        weight->numel() == classes,
        std::string("Vulkan ") + name +
            ": weight must cover every class or none");
    const Tensor on_device =
        weight->device().is_vulkan() ? *weight : weight->to(vulkan_device());
    device_weight = on_device.dtype() == DType::Float32
        ? on_device.contiguous()
        : on_device.to(DType::Float32).contiguous();
  }

  LossOperands operands;
  operands.target = wide_target;
  operands.weight = device_weight;
  operands.v_target = convert(wide_target);
  operands.v_weight = device_weight.defined()
      ? to_buffer_storage_dispatch(context, convert(device_weight))
      : operands.v_target;
  split_code(ignore_index, &operands.skip_low, &operands.skip_high);
  return operands;
}

// The rows walk totals in single precision whatever the likelihood plane was
// read as, so a narrow plane rounds its loss and its normalizer once, at the end
// of the walk.
Tensor answered_as(const Tensor& totals, DType dtype) {
  return totals.dtype() == dtype ? totals : tpx::ops::to(totals, dtype);
}

// One pass over the rows: each row reads the class its target names and answers
// with its loss and the class weight it contributes to the normalizer.  Nothing
// accumulates here, so a row neither reads nor races another row's word, and the
// totals a reduction asks for are a sum of a payload away - no row value crosses
// through the host.
std::pair<Tensor, Tensor> loss_rows(
    const Tensor& self,
    const LossOperands& operands,
    int64_t rows,
    int64_t classes,
    const std::vector<int64_t>& loss_sizes) {
  api::Context* const context = api::context();
  const api::vTensor v_self = convert(self.contiguous());
  const bool packed = v_self.storage_type() != api::StorageType::BUFFER;
  const bool has_weight = operands.weight.defined();

  api::vTensor v_loss{
      context,
      loss_sizes,
      DType::Float32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  api::vTensor v_weights{
      context,
      std::vector<int64_t>{rows},
      DType::Float32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  if (rows > 0) {
    const struct RowBlock block{
        static_cast<int32_t>(rows),
        static_cast<int32_t>(classes),
        operands.skip_low,
        operands.skip_high,
        0,
        0,
    };
    api::UniformParamsBuffer params(context, block);
    // A dense narrow plane holds two classes per word, so its own walk reads the
    // pair a word keeps; a plane with an image form needs no such variant, as
    // the texel fetch converts the narrow component on its way in.
    std::string shader =
        packed ? "nll_loss_rows" : "nll_loss_rows_buffer";
    if (!packed && self.dtype() == DType::Float16) {
      shader += "_f16";
    }
    if (has_weight) {
      shader += "_weight";
    }
    api::PipelineBarrier barrier{};
    const api::utils::uvec3 extent{
        api::utils::safe_downcast_to_u32(rows), 1u, 1u};
    const api::utils::uvec3 local{kRowWalkWidth, 1u, 1u};

    if (packed) {
      context->submit_compute_job(
          VK_KERNEL_FROM_STR(shader),
          barrier,
          extent,
          local,
          VK_NULL_HANDLE,
          v_loss.buffer(
              barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_weights.buffer(
              barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_self.image(barrier, api::PipelineStage::COMPUTE),
          operands.v_target.buffer(barrier, api::PipelineStage::COMPUTE),
          operands.v_weight.buffer(barrier, api::PipelineStage::COMPUTE),
          params.buffer());
    } else {
      context->submit_compute_job(
          VK_KERNEL_FROM_STR(shader),
          barrier,
          extent,
          local,
          VK_NULL_HANDLE,
          v_loss.buffer(
              barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_weights.buffer(
              barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_self.buffer(barrier, api::PipelineStage::COMPUTE),
          operands.v_target.buffer(barrier, api::PipelineStage::COMPUTE),
          operands.v_weight.buffer(barrier, api::PipelineStage::COMPUTE),
          params.buffer());
    }
  }

  return {convert(v_loss), convert(v_weights)};
}

// The payload a gradient reads its incoming value from: one entry per row for an
// unreduced loss, one shared entry otherwise.
Tensor grad_incoming(
    const Tensor& grad_output,
    int64_t rows,
    int64_t reduction,
    const char* name) {
  const Tensor device_grad = grad_output.device().is_vulkan()
      ? grad_output
      : grad_output.to(vulkan_device());
  TP_CHECK(
      device_grad.dtype() == DType::Float32 ||
          device_grad.dtype() == DType::Float16,
      std::string("Vulkan ") + name +
          " answers from a Float32 or Float16 gradient");
  // The walk reads the values it hands out as single-precision words, so a
  // narrow gradient is widened here - one entry per row, or one shared entry -
  // rather than in the pass over the plane.
  const Tensor incoming = device_grad.dtype() == DType::Float32
      ? device_grad.contiguous()
      : tpx::ops::to(device_grad, DType::Float32).contiguous();
  TP_CHECK(
      incoming.numel() > 0,
      std::string("Vulkan ") + name + " expects a gradient to hand out");
  if (reduction == 0) {
    TP_CHECK(
        incoming.numel() == rows,
        std::string("Vulkan ") + name +
            ": an unreduced gradient hands one value to every row");
  }
  return incoming;
}

} // namespace

std::tuple<Tensor, Tensor> nll_loss_kernel(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  constexpr const char* kName = "nll_loss";
  const LossShape shape = loss_shape(self, kName);
  TP_CHECK(
      reduction >= 0 && reduction <= 2,
      std::string("Vulkan ") + kName + ": invalid reduction mode");

  api::Context* const context = api::context();
  const LossOperands operands = prepare_operands(
      context,
      target,
      weight,
      shape.rows,
      shape.classes,
      ignore_index,
      kName);

  if (reduction == 0) {
    // Unreduced: every row keeps its loss.  A single unbatched row has no batch
    // axis to keep, and it reports the weight it carries; a batch of rows
    // reports no normalizer at all.
    const std::vector<int64_t> loss_sizes =
        shape.unbatched ? std::vector<int64_t>{}
                        : std::vector<int64_t>{shape.rows};
    const auto rows = loss_rows(
        self,
        operands,
        shape.rows,
        shape.classes,
        loss_sizes);
    if (shape.unbatched) {
      return {
          answered_as(rows.first, self.dtype()),
          answered_as(tpx::ops::sum(rows.second), self.dtype())};
    }
    return {
        answered_as(rows.first, self.dtype()),
        zeros_kernel({}, self.dtype(), self.device(), false)};
  }

  const auto rows =
      loss_rows(self, operands, shape.rows, shape.classes,
                std::vector<int64_t>{shape.rows});
  Tensor loss_total = tpx::ops::sum(rows.first);
  const Tensor total_weight = tpx::ops::sum(rows.second);
  if (reduction == 1) {
    // Rows that all named the skipped code divide nothing by nothing, the
    // answer an average over no rows takes.  The divide runs in single
    // precision, so a narrow plane is rounded only once its answer is known.
    loss_total = tpx::ops::div(loss_total, total_weight);
  }
  return {
      answered_as(loss_total, self.dtype()),
      answered_as(total_weight, self.dtype())};
}

Tensor nll_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  constexpr const char* kName = "nll_loss_backward";
  const LossShape shape = loss_shape(self, kName);
  TP_CHECK(
      reduction >= 0 && reduction <= 2,
      std::string("Vulkan ") + kName + ": invalid reduction mode");

  api::Context* const context = api::context();
  const LossOperands operands = prepare_operands(
      context,
      target,
      weight,
      shape.rows,
      shape.classes,
      ignore_index,
      kName);
  const Tensor incoming =
      grad_incoming(grad_output, shape.rows, reduction, kName);

  api::vTensor v_grad = allocate_output(
      context,
      convert(self.contiguous()).storage_type(),
      std::vector<int64_t>(self.sizes().begin(), self.sizes().end()),
      self.dtype());
  const bool packed =
      v_grad.storage_type() != api::StorageType::BUFFER;

  const api::vTensor v_incoming =
      to_buffer_storage_dispatch(context, convert(incoming));

  // The averaged form divides the shared gradient by the weight the forward pass
  // measured.  Without that measurement on hand, the rows walk makes it.
  Tensor measured;
  api::vTensor v_total = v_incoming;
  int32_t mean_mode = 0;
  if (reduction == 1) {
    mean_mode = 1;
    if (total_weight.defined()) {
      const Tensor on_device = total_weight.device().is_vulkan()
          ? total_weight
          : total_weight.to(vulkan_device());
      measured = on_device.dtype() == DType::Float32
          ? on_device
          : tpx::ops::to(on_device, DType::Float32);
    } else {
      measured = tpx::ops::sum(
          loss_rows(
              self,
              operands,
              shape.rows,
              shape.classes,
              std::vector<int64_t>{shape.rows})
              .second);
    }
    TP_CHECK(
        measured.numel() == 1,
        std::string("Vulkan ") + kName +
            ": the averaged form needs one weight total");
    v_total = to_buffer_storage_dispatch(context, convert(measured));
  }

  const struct GradBlock block{
      static_cast<int32_t>(shape.rows),
      static_cast<int32_t>(shape.classes),
      operands.skip_low,
      operands.skip_high,
      mean_mode,
      reduction == 0 ? 1 : 0,
  };
  api::UniformParamsBuffer params(context, block);

  if (shape.rows > 0 && shape.classes > 0) {
    const bool narrow = self.dtype() == DType::Float16;
    // A narrow plane is stored as narrow texels or as two classes per word, so
    // both storages answer with their own variant.
    std::string shader =
        packed ? "nll_loss_grad" : "nll_loss_grad_buffer";
    if (narrow) {
      shader += "_f16";
    }
    if (operands.weight.defined()) {
      shader += "_weight";
    }
    api::PipelineBarrier barrier{};
    // Every element of the plane gets exactly one writer: the classes an image
    // row holds ride along its width, a dense plane walks word by word, and a
    // dense narrow plane walks the pairs its words hold.
    const int64_t elements = shape.rows * shape.classes;
    const api::utils::uvec3 extent = packed
        ? v_grad.extents()
        : api::utils::uvec3{
              api::utils::safe_downcast_to_u32(
                  narrow ? (elements + 1) / 2 : elements),
              1u,
              1u};
    const api::utils::uvec3 local = packed
        ? adaptive_work_group_size(extent)
        : api::utils::uvec3{kRowWalkWidth, 1u, 1u};

    if (packed) {
      context->submit_compute_job(
          VK_KERNEL_FROM_STR(shader),
          barrier,
          extent,
          local,
          VK_NULL_HANDLE,
          v_grad.image(
              barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          operands.v_target.buffer(barrier, api::PipelineStage::COMPUTE),
          v_incoming.buffer(barrier, api::PipelineStage::COMPUTE),
          v_total.buffer(barrier, api::PipelineStage::COMPUTE),
          operands.v_weight.buffer(barrier, api::PipelineStage::COMPUTE),
          params.buffer());
    } else {
      context->submit_compute_job(
          VK_KERNEL_FROM_STR(shader),
          barrier,
          extent,
          local,
          VK_NULL_HANDLE,
          v_grad.buffer(
              barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          operands.v_target.buffer(barrier, api::PipelineStage::COMPUTE),
          v_incoming.buffer(barrier, api::PipelineStage::COMPUTE),
          v_total.buffer(barrier, api::PipelineStage::COMPUTE),
          operands.v_weight.buffer(barrier, api::PipelineStage::COMPUTE),
          params.buffer());
    }
  }

  return convert(v_grad);
}

// A plane compared with another plane: one walk reads both and lays down the
// square of how far each element sits from the value it is matched with.  The
// squares answer as one payload, so a loss that totals them is a reduction away
// rather than two element walks and a difference kept on the device.
Tensor mse_loss_kernel(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  constexpr const char* kName = "mse_loss";
  TP_CHECK(
      reduction >= 0 && reduction <= 2,
      std::string("Vulkan ") + kName + ": invalid reduction mode");
  const Tensor compared =
      target.device().is_vulkan() ? target : target.to(vulkan_device());
  const Tensor squared = binary_op_tensor_dispatch(
      "sq_diff", "buffer_sq_diff", self, compared, Scalar(1.0), kName);
  if (reduction == 0) {
    return squared;
  }
  return reduction == 1 ? tpx::ops::mean(squared) : tpx::ops::sum(squared);
}

// The gradient one element of the plane hands back is twice its distance from
// the value it was compared with, shared by the count that was averaged when
// the loss asks for an average, and carried by whatever the loss is answered
// with.
Tensor mse_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  constexpr const char* kName = "mse_loss_backward";
  TP_CHECK(
      reduction >= 0 && reduction <= 2,
      std::string("Vulkan ") + kName + ": invalid reduction mode");
  const double norm = reduction == 1
      ? 2.0 / static_cast<double>(self.numel())
      : 2.0;
  const Tensor compared =
      target.device().is_vulkan() ? target : target.to(vulkan_device());
  const Tensor incoming = grad_output.device().is_vulkan()
      ? grad_output
      : grad_output.to(vulkan_device());
  Tensor difference = binary_op_tensor_dispatch(
      "sub", "buffer_sub", self, compared, Scalar(1.0), kName);
  difference = binary_op_scalar_dispatch(
      "mul_scalar", "buffer_mul_scalar", difference, Scalar(norm),
      Scalar(1.0), kName);
  return binary_op_tensor_dispatch(
      "mul", "buffer_mul", difference, incoming, Scalar(1.0), kName);
}

// An operand of a binary likelihood walk: read at single precision, however
// narrow the payload it came from is.  A likelihood clamps a logarithm and
// divides by the product of two lanes, both of which lose their meaning in a
// narrow width.
Tensor loss_walk_operand(const Tensor& t) {
  return t.dtype() == DType::Float32 ? t : tpx::ops::to(t, DType::Float32);
}

// A plane read as probabilities has to sit between the bounds a logarithm is
// taken of.  A lane outside them has no loss to answer, and a lane that is not a
// measurement at all fails the same way: the pair of bounds is a test a missing
// measurement answers neither way.
void loss_check_probability(const Tensor& payload, const char* what) {
  const Tensor inside = tpx::ops::logical_and(
      tpx::ops::ge(payload, Scalar(0.0)),
      tpx::ops::le(payload, Scalar(1.0)));
  TP_CHECK(
      tpx::ops::all(inside).to(Device(DeviceType::CPU)).item().to<bool>(),
      std::string("all elements of ") + what + " should be between 0 and 1");
}

// A plane read as probabilities, brought onto the device and widened for the walk.
Tensor loss_probability_operand(const Tensor& t, const char* what) {
  const Tensor on_device =
      t.device().is_vulkan() ? t : t.to(vulkan_device());
  const Tensor walked = loss_walk_operand(on_device);
  loss_check_probability(walked, what);
  return walked;
}

// L = -w (t log(x) + (1-t) log(1-x)), with each logarithm clamped at -100 so a
// lane that sits exactly on a bound answers a finite loss rather than an
// infinity.  The two clamped logarithms are read once and shared by every
// reduction mode; a per-lane weight rides the same walk the two planes do.
Tensor binary_cross_entropy_kernel(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction) {
  constexpr const char* kName = "binary_cross_entropy";
  TP_CHECK(
      reduction >= 0 && reduction <= 2,
      std::string("Vulkan ") + kName + ": invalid reduction mode");
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      std::string("Vulkan ") + kName +
          " supports Float32 and Float16 tensors only");
  const bool has_weight = weight_opt.has_value() && weight_opt->defined();

  const Tensor x = loss_probability_operand(self, "input");
  const Tensor t = loss_probability_operand(target, "target");

  const Tensor log_x = tpx::ops::clamp_min(tpx::ops::log(x), Scalar(-100.0));
  const Tensor log_one_minus_x = tpx::ops::clamp_min(
      tpx::ops::log1p(tpx::ops::neg(x)), Scalar(-100.0));
  Tensor loss = tpx::ops::sub(
      tpx::ops::mul(tpx::ops::add(t, Scalar(-1.0)), log_one_minus_x),
      tpx::ops::mul(t, log_x));
  if (has_weight) {
    loss = tpx::ops::mul(
        loss,
        loss_walk_operand(
            weight_opt->device().is_vulkan()
                ? *weight_opt
                : weight_opt->to(vulkan_device())));
  }
  if (reduction == 0) {
    return tpx::ops::to(loss, self.dtype());
  }
  return tpx::ops::to(
      reduction == 1 ? tpx::ops::mean(loss) : tpx::ops::sum(loss),
      self.dtype());
}

// d(L)/d(x) = w (x - t) / max((1 - x) x, 1e-12), carried by whatever the loss is
// answered with and shared by the count that was averaged for the mean form.
// The denominator is held off zero rather than trusted: a lane exactly on a bound
// would otherwise answer an infinity the loss itself keeps finite.
Tensor binary_cross_entropy_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction) {
  constexpr const char* kName = "binary_cross_entropy_backward";
  TP_CHECK(
      reduction >= 0 && reduction <= 2,
      std::string("Vulkan ") + kName + ": invalid reduction mode");
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      std::string("Vulkan ") + kName +
          " supports Float32 and Float16 tensors only");
  const bool has_weight = weight_opt.has_value() && weight_opt->defined();

  const Tensor x = loss_walk_operand(
      self.device().is_vulkan() ? self : self.to(vulkan_device()));
  const Tensor t = loss_walk_operand(
      target.device().is_vulkan() ? target : target.to(vulkan_device()));
  const Tensor incoming = loss_walk_operand(
      grad_output.device().is_vulkan() ? grad_output
                                       : grad_output.to(vulkan_device()));

  const Tensor denominator = tpx::ops::clamp_min(
      tpx::ops::mul(tpx::ops::rsub(x, Scalar(1.0)), x), Scalar(1e-12));
  Tensor gradient = tpx::ops::div(
      tpx::ops::mul(incoming, tpx::ops::sub(x, t)), denominator);
  if (has_weight) {
    gradient = tpx::ops::mul(
        gradient,
        loss_walk_operand(
            weight_opt->device().is_vulkan()
                ? *weight_opt
                : weight_opt->to(vulkan_device())));
  }
  if (reduction == 1 && self.numel() > 0) {
    gradient = tpx::ops::mul(
        gradient, Scalar(1.0 / static_cast<double>(self.numel())));
  }
  return tpx::ops::to(gradient, self.dtype());
}

// The loss a logit answers before it is pushed through a sigmoid:
// (1-t) x - log_sigmoid(x) ((pos_weight - 1) t + 1), per-lane weighted and shared
// by the count the walk covered.  Reading log_sigmoid rather than a sigmoid keeps
// a saturated logit finite, which is the reason this form is answered on its own
// instead of as a sigmoid and a likelihood in two walks.
Tensor binary_cross_entropy_with_logits_kernel(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt) {
  constexpr const char* kName = "binary_cross_entropy_with_logits";
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      std::string("Vulkan ") + kName +
          " supports Float32 and Float16 tensors only");
  const bool has_weight = weight_opt.has_value() && weight_opt->defined();
  const bool has_pos_weight =
      pos_weight_opt.has_value() && pos_weight_opt->defined();
  if (self.numel() == 0) {
    // Nothing was walked, so the average is the zero an empty count hands back.
    return Tensor::zeros(std::vector<int64_t>{}, DType::Float32, self.device());
  }

  const Tensor x = loss_walk_operand(
      self.device().is_vulkan() ? self : self.to(vulkan_device()));
  const Tensor t = loss_walk_operand(
      target.device().is_vulkan() ? target : target.to(vulkan_device()));

  Tensor log_sigmoid = tpx::ops::log_sigmoid(x);
  if (has_pos_weight) {
    const Tensor pos_weight = loss_walk_operand(
        pos_weight_opt->device().is_vulkan()
            ? *pos_weight_opt
            : pos_weight_opt->to(vulkan_device()));
    log_sigmoid = tpx::ops::mul(
        log_sigmoid,
        tpx::ops::add(
            tpx::ops::mul(tpx::ops::add(pos_weight, Scalar(-1.0)), t),
            Scalar(1.0)));
  }
  Tensor loss = tpx::ops::sub(
      tpx::ops::mul(tpx::ops::rsub(t, Scalar(1.0)), x), log_sigmoid);
  if (has_weight) {
    loss = tpx::ops::mul(
        loss,
        loss_walk_operand(
            weight_opt->device().is_vulkan()
                ? *weight_opt
                : weight_opt->to(vulkan_device())));
  }
  // The average of a likelihood is answered at single precision.
  return tpx::ops::mean(loss);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, LossKernels) {
  m.impl("nll_loss", &tensorplay::vulkan::ops::nll_loss_kernel);
  m.impl(
      "nll_loss_backward",
      &tensorplay::vulkan::ops::nll_loss_backward_kernel);
  m.impl("mse_loss", &tensorplay::vulkan::ops::mse_loss_kernel);
  m.impl(
      "mse_loss_backward",
      &tensorplay::vulkan::ops::mse_loss_backward_kernel);
  m.impl(
      "binary_cross_entropy",
      &tensorplay::vulkan::ops::binary_cross_entropy_kernel);
  m.impl(
      "binary_cross_entropy_backward",
      &tensorplay::vulkan::ops::binary_cross_entropy_backward_kernel);
  m.impl(
      "binary_cross_entropy_with_logits",
      &tensorplay::vulkan::ops::binary_cross_entropy_with_logits_kernel);
}

#endif /* USE_VULKAN */
