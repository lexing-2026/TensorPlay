// Composite kernel: chain_matmul.
// alias of linalg.multi_dot).  The optimal-parenthesization DP only
// changes evaluation order, so the sequential matmul fold is numerically
// equivalent up to fp associativity.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor chain_matmul_native(const std::vector<Tensor>& matrices) {
    for (const auto& m : matrices) {
        if (m.dim() != 2) {
            TP_THROW(RuntimeError,
                     "chain_matmul(): all matrices must be 2-D, but got a ",
                     m.dim(), "-D tensor");
        }
    }
    if (matrices.empty()) {
        TP_THROW(RuntimeError,
                 "chain_matmul(): Expected one or more matrices");
    }
    if (matrices.size() == 1) return ops::clone(matrices[0], kContiguous);
    Tensor result = matrices[0];
    for (size_t i = 1; i < matrices.size(); ++i) {
        result = ops::matmul(result, matrices[i]);
    }
    return result;
}

TENSORPLAY_LIBRARY_IMPL(Composite, LinearAlgebraComposite) {
    m.impl("chain_matmul", chain_matmul_native);
}

} // namespace composite
} // namespace tensorplay
