#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
  TP_CHECK(
      self.dtype() == DType::Float32 && mat2.dtype() == DType::Float32,
      "Vulkan mm supports Float32 tensors only");
  TP_CHECK(self.dim() == 2, "Vulkan mm: self must be a matrix");
  TP_CHECK(mat2.dim() == 2, "Vulkan mm: mat2 must be a matrix");
  TP_CHECK(
      self.size(1) == mat2.size(0),
      "Vulkan mm: matrix dimensions do not match");

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);
  api::vTensor v_other = convert(mat2);

  const int64_t M = self.size(0);
  const int64_t N = mat2.size(1);
  const int64_t K = self.size(1);

  api::vTensor v_output{
      context,
      {M, N},
      self.dtype(),
  };

  if (M == 0 || N == 0 || K == 0) {
    // Zero-sized operands: with K == 0 every element is zero by
    // definition; empty results need no work at all.
    if (K == 0 && M != 0 && N != 0) {
      Tensor out = convert(v_output);
      out.fill_(Scalar(0.0));
      return out;
    }
    return convert(v_output);
  }

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan mm requires texture storage");
  }

  const struct MMBlock final {
    ivec4 out_sizes;
    ivec4 in1_sizes;
    ivec4 in2_sizes;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      make_whcn_ivec4(v_self.sizes()),
      make_whcn_ivec4(v_other.sizes()),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(mm),
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, MatmulKernels) {
  m.impl("mm", &tensorplay::vulkan::ops::mm_kernel);
}

#endif /* USE_VULKAN */
