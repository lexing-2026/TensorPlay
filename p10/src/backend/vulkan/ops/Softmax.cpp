#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <algorithm>
#include <string>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

Tensor softmax_impl(
    const Tensor& input_arg,
    int64_t dim,
    bool log_mode) {
  TP_CHECK(
      input_arg.dtype() == DType::Float32 || input_arg.dtype() == DType::Float16,
      "Vulkan softmax supports Float32 and Float16 tensors only");
  TP_CHECK(
      input_arg.dim() >= 1 && input_arg.dim() <= 4,
      "Vulkan softmax supports 1d to 4d tensors");

  const int64_t ndim = input_arg.dim();
  dim = dim < 0 ? dim + ndim : dim;
  TP_CHECK(
      dim >= 0 && dim < ndim,
      "Vulkan softmax: dim out of range");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input_arg);

  // The result is released in the operand's storage class: the row-walking
  // variant below reads and writes words, so a dense input needs a dense
  // result even when the shape would also fit an image.
  api::vTensor v_output =
      allocate_output_like(context, v_input, v_input.sizes(), v_input.dtype());

  // Buffer-stored payloads (any axis beyond the 3-D image limit) reduce
  // through a linear row-walking variant.  A half payload normalizes two
  // elements per output word, so its grid counts words instead of rows.
  if (v_output.storage_type() == api::StorageType::BUFFER) {
    const bool is_half = v_input.dtype() == DType::Float16;
    TP_CHECK(
        v_input.dtype() == DType::Float32 || is_half,
        "Vulkan softmax over linear storage supports Float32 and Float16 "
        "tensors only");
    const int64_t axis_size = input_arg.size(dim);
    int64_t inner = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
      inner *= input_arg.size(d);
    }
    const int64_t rows = input_arg.numel() / axis_size;
    const struct SoftmaxBufferBlock final {
      int32_t axis_size;
      int32_t inner;
      int32_t rows;
    } block{
        static_cast<int32_t>(axis_size), static_cast<int32_t>(inner),
        static_cast<int32_t>(rows),
    };
    const std::string shader =
        std::string(log_mode ? "log_softmax_buffer" : "softmax_buffer") +
        (is_half ? "_half" : "");
    const int64_t grid =
        is_half ? (input_arg.numel() + 1) / 2 : rows;
    // A half payload dispatches one invocation per output word along the x
    // axis; the float builds are workgroup-cooperative, so the row rides on
    // the y axis and x carries the 256 lanes that stride along it.
    api::utils::uvec3 dispatch_grid{static_cast<uint32_t>(grid), 1u, 1u};
    if (!is_half) {
      dispatch_grid = api::utils::uvec3{1u, static_cast<uint32_t>(grid), 1u};
    }
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(shader.c_str()),
        pipeline_barrier,
        dispatch_grid, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
    return convert(v_output);
  }

  const int64_t wrapped_dim = input_arg.dim() - 1 - dim;

  const struct SoftmaxBlock final {
    ivec4 sizes;
    int c_depth;
    int axis;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      c_depth_of(v_input.sizes()),
      static_cast<int32_t>(wrapped_dim),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  const api::ShaderInfo& shader =
      (wrapped_dim == 2)
      ? (log_mode ? kernel_for("log_softmax_channel", input_arg.dtype())
                  : kernel_for("softmax_channel", input_arg.dtype()))
      : (log_mode ? kernel_for("log_softmax", input_arg.dtype())
                 : kernel_for("softmax", input_arg.dtype()));

  context->submit_compute_job(
      // shader descriptor
      shader,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

} // namespace

Tensor softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
  (void)dtype;
  return softmax_impl(self, dim, /*log_mode=*/false);
}

Tensor log_softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
  (void)dtype;
  return softmax_impl(self, dim, /*log_mode=*/true);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, SoftmaxKernels) {
  m.impl("softmax", &tensorplay::vulkan::ops::softmax_kernel);
  m.impl("log_softmax", &tensorplay::vulkan::ops::log_softmax_kernel);
}

#endif /* USE_VULKAN */
