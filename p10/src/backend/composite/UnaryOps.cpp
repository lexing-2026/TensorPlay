// Composite kernels: resolve_conj / resolve_neg.
// unchanged).

#include "Tensor.h"
#include "Dispatcher.h"

namespace tensorplay {
namespace composite {

Tensor resolve_conj_native(const Tensor& self) { return self; }

Tensor resolve_neg_native(const Tensor& self) { return self; }

TENSORPLAY_LIBRARY_IMPL(Composite, UnaryOpsComposite) {
    m.impl("resolve_conj", resolve_conj_native);
    m.impl("resolve_neg", resolve_neg_native);
}

} // namespace composite
} // namespace tensorplay
