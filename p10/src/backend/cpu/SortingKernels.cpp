// Sorting operators - CPU kernels.
#include "Tensor.h"
#include "Dispatcher.h"

#include <tuple>

namespace tensorplay {
namespace cpu {

namespace {
Tensor msort_cpu(const Tensor& self) {
    // msort: values of sort along dim 0.
    Tensor values = std::get<0>(self.sort(0, false));
    return values;
}


}  // namespace

TENSORPLAY_LIBRARY_IMPL(CPU, SortingKernels) {
    m.impl("msort", msort_cpu);
}

} // namespace cpu
} // namespace tensorplay
