// Sorting operators - CPU kernels.
#include "Tensor.h"
#include "Dispatcher.h"

#include <tuple>

namespace tensorplay {
namespace cpu {

namespace {
// msort is the values of a sort along dim 0, on any device, and it is
// differentiated through that sort.
Tensor msort_composite(const Tensor& self) {
    return std::get<0>(self.sort(0, false));
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(Composite, SortingComposites) {
    m.impl("msort", msort_composite);
}

} // namespace cpu
} // namespace tensorplay
