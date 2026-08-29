// Tensor-shape aliases.

#include "Tensor.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>

namespace tensorplay::cpu {

namespace ops = tensorplay::tpx::ops;

Tensor concat_native_cpu(const std::vector<Tensor>& tensors, int64_t dim) {
    return ops::cat(tensors, dim);
}

Tensor concatenate_native_cpu(const std::vector<Tensor>& tensors, int64_t dim) {
    return ops::cat(tensors, dim);
}

Tensor diagflat_native_cpu(const Tensor& self, int64_t offset) {
    Tensor flat = ops::view(ops::contiguous(self, 0), {-1});
    return ops::diag(flat, offset);
}

TENSORPLAY_LIBRARY_IMPL(CPU, NativeTensorShape) {
    m.impl("concat", concat_native_cpu);
    m.impl("concatenate", concatenate_native_cpu);
    m.impl("diagflat", diagflat_native_cpu);
}

} // namespace tensorplay::cpu
