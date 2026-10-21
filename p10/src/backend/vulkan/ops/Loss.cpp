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
      self.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
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
    std::string shader =
        packed ? "nll_loss_rows" : "nll_loss_rows_buffer";
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
      device_grad.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 gradients only");
  const Tensor incoming = device_grad.contiguous();
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
      return {rows.first, tpx::ops::sum(rows.second)};
    }
    return {
        rows.first,
        zeros_kernel({}, DType::Float32, self.device(), false)};
  }

  const auto rows =
      loss_rows(self, operands, shape.rows, shape.classes,
                std::vector<int64_t>{shape.rows});
  Tensor loss_total = tpx::ops::sum(rows.first);
  const Tensor total_weight = tpx::ops::sum(rows.second);
  if (reduction == 1) {
    // Rows that all named the skipped code divide nothing by nothing, the
    // answer an average over no rows takes.
    loss_total = tpx::ops::div(loss_total, total_weight);
  }
  return {loss_total, total_weight};
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
      DType::Float32);
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
    std::string shader =
        packed ? "nll_loss_grad" : "nll_loss_grad_buffer";
    if (operands.weight.defined()) {
      shader += "_weight";
    }
    api::PipelineBarrier barrier{};
    // Every element of the plane gets exactly one writer: the classes an image
    // row holds ride along its width, and a dense plane walks word by word.
    const api::utils::uvec3 extent = packed
        ? v_grad.extents()
        : api::utils::uvec3{
              api::utils::safe_downcast_to_u32(
                  shape.rows * shape.classes),
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

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, LossKernels) {
  m.impl("nll_loss", &tensorplay::vulkan::ops::nll_loss_kernel);
  m.impl(
      "nll_loss_backward",
      &tensorplay::vulkan::ops::nll_loss_backward_kernel);
}

#endif /* USE_VULKAN */
