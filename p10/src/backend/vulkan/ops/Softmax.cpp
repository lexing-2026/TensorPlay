#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <algorithm>
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
      input_arg.dtype() == DType::Float32,
      "Vulkan softmax supports Float32 tensors only");
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

  api::vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };

  const int64_t wrapped_dim = input_arg.dim() - 1 - dim;

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan softmax requires texture storage");
  }

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
      ? (log_mode ? VK_KERNEL(log_softmax_channel)
                  : VK_KERNEL(softmax_channel))
      : (log_mode ? VK_KERNEL(log_softmax) : VK_KERNEL(softmax));

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
