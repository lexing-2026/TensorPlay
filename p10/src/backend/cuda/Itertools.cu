// Itertools native implementation.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>

namespace tensorplay::cuda {

namespace ops = tensorplay::tpx::ops;

Tensor cartesian_prod_native_cuda(const std::vector<Tensor>& tensors) {
    for (const Tensor& tensor : tensors) {
        if (tensor.dim() != 1) {
            TP_THROW(RuntimeError, "Expect a 1D vector, but got shape ",
                     tensor.shape().toString());
        }
    }
    if (tensors.size() == 1) return tensors[0];

    std::vector<Tensor> grids = ops::meshgrid(tensors, "ij");
    for (Tensor& grid : grids) grid = ops::flatten(grid, 0, -1);
    return ops::stack(grids, 1);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeItertools) {
    m.impl("cartesian_prod", cartesian_prod_native_cuda);
}

} // namespace tensorplay::cuda
