// Sorting operators - CUDA kernels.
#include "Tensor.h"
#include "Dispatcher.h"

#include <tuple>

namespace tensorplay {
namespace cuda {

namespace {
Tensor msort_cuda(const Tensor& self) {
    // msort: values of sort along dim 0.
    Tensor values = std::get<0>(self.sort(0, false));
    return values;
}


} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SortingKernels) {
    m.impl("msort", msort_cuda);
}

} // namespace cuda
} // namespace tensorplay
