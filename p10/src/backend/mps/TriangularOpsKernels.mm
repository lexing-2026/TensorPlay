#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace tensorplay {
namespace mps {

// The bundled Metal shader library (TriangularOps.metal).
static auto& lib = MetalShaderLibrary::getBundledLibrary();

namespace {

template <typename T>
std::vector<T> reverse_array(const std::vector<int64_t>& arr) {
  std::vector<T> rc(arr.size());
  for (size_t i = 0; i < arr.size(); ++i) {
    rc[i] = static_cast<T>(arr[arr.size() - 1 - i]);
  }
  return rc;
}

void triu_tril_impl(const Tensor& self, int64_t k, const Tensor& out,
                    const std::string& name, bool inplace) {
  if (self.numel() == 0) {
    return;
  }
  // The shader walks storage with explicit strides; bind contiguous tensors
  // so the (reversed) stride array describes the actual buffer exactly.
  const Tensor self_c = self.contiguous();
  auto sizes = reverse_array<uint32_t>(
      static_cast<std::vector<int64_t>>(self_c.shape()));
  auto inp_strides = reverse_array<int32_t>(self_c.strides());
  auto out_strides = reverse_array<int32_t>(out.strides());
  std::array<int, 2> k_ndim = {int(k), int(self_c.dim())};
  const std::string kernel_name =
      name + (inplace ? "_inplace" : "") + "_int_" +
      scalarToMetalTypeString(self_c.dtype());
  auto pso = lib.getPipelineStateForFunc(kernel_name);
  const uint32_t max_threads_per_group = [pso maxTotalThreadsPerThreadgroup];
  const uint32_t w = sizes[0];
  const uint32_t batch =
      static_cast<uint32_t>(self_c.numel() / (sizes[0] * sizes[1]));

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      stream->endKernelCoalescing();
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:pso];
      if (inplace) {
        mtl_setArgs(computeEncoder, self_c, inp_strides, sizes, k_ndim);
      } else {
        mtl_setArgs(computeEncoder, out, self_c, out_strides, inp_strides,
                    sizes, k_ndim);
      }
      [computeEncoder dispatchThreads:MTLSizeMake(sizes[0], sizes[1], batch)
                threadsPerThreadgroup:MTLSizeMake(
                                           std::min(max_threads_per_group, w),
                                           1, 1)];
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
  });
}

Tensor tril_kernel(const Tensor& self, int64_t diagonal) {
  Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                self.dtype(), self.device());
  triu_tril_impl(self, diagonal, result, "tril", false);
  return result;
}

Tensor triu_kernel(const Tensor& self, int64_t diagonal) {
  Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                self.dtype(), self.device());
  triu_tril_impl(self, diagonal, result, "triu", false);
  return result;
}

Tensor& tril_inplace_kernel(Tensor& self, int64_t diagonal) {
  triu_tril_impl(self, diagonal, self, "tril", true);
  return self;
}

Tensor& triu_inplace_kernel(Tensor& self, int64_t diagonal) {
  triu_tril_impl(self, diagonal, self, "triu", true);
  return self;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(MPS, TriangularOpsKernels) {
  m.impl("tril", tril_kernel);
  m.impl("triu", triu_kernel);
  m.impl("tril_", tril_inplace_kernel);
  m.impl("triu_", triu_inplace_kernel);
}

} // namespace mps
} // namespace tensorplay