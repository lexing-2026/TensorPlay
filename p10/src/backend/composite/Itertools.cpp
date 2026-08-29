// Composite kernel: combinations.
// grids -> monotonic-index mask -> nonzero -> gather.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor combinations_native(const Tensor& self, int64_t r,
                           bool with_replacement) {
    if (self.dim() != 1) {
        TP_THROW(RuntimeError, "Expect a 1D vector, but got shape ",
                 self.shape().toString());
    }
    if (r < 0) {
        TP_THROW(RuntimeError, "Expect a non-negative number, but got ", r);
    }
    const int64_t n = self.size(0);
    if (r == 0) return ops::empty({0}, self.dtype(), self.device());

    const Tensor range = ops::arange(Scalar(int64_t(0)), Scalar(n),
                                     Scalar(int64_t(1)), DType::Int64,
                                     self.device());
    std::vector<Tensor> ranges(static_cast<size_t>(r), range);
    std::vector<Tensor> grids = ops::meshgrid(ranges, "ij");
    Tensor mask;
    for (int64_t i = 0; i + 1 < r; ++i) {
        Tensor cond = with_replacement ? ops::le(grids[i], grids[i + 1])
                                       : ops::lt(grids[i], grids[i + 1]);
        mask = mask.defined() ? ops::logical_and(mask, cond) : cond;
    }
    const Tensor indices = ops::nonzero(mask);
    if (indices.size(0) == 0) {
        // Alignment-baseline contract: an empty combination set collapses to
        // a 1-D empty tensor regardless of r.
        return ops::empty({0}, self.dtype(), self.device());
    }
    return ops::take(self, indices);
}

TENSORPLAY_LIBRARY_IMPL(Composite, ItertoolsComposite) {
    m.impl("combinations", combinations_native);
}

} // namespace composite
} // namespace tensorplay
