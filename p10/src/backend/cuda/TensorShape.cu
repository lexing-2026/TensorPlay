// Tensor-shape aliases.

#include "Tensor.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>

namespace tensorplay::cuda {

namespace ops = tensorplay::tpx::ops;

Tensor concat_native_cuda(const std::vector<Tensor>& tensors, int64_t dim) {
    return ops::cat(tensors, dim);
}

Tensor concatenate_native_cuda(const std::vector<Tensor>& tensors, int64_t dim) {
    return ops::cat(tensors, dim);
}

Tensor diagflat_native_cuda(const Tensor& self, int64_t offset) {
    Tensor flat = ops::view(ops::contiguous(self, 0), {-1});
    return ops::diag(flat, offset);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeTensorShape) {
    m.impl("concat", concat_native_cuda);
    m.impl("concatenate", concatenate_native_cuda);
    m.impl("diagflat", diagflat_native_cuda);
}

} // namespace tensorplay::cuda
