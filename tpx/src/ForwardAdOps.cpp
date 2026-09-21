#include "Autograd.h"
#include "Exception.h"
#include "TensorImpl.h"

#include <string>
#include <tuple>
#include <utility>

namespace tensorplay {
namespace tpx {
namespace {

// Refuse tensors carrying an active transform level: their public metadata
// describes an unbatched view while the payload lives in the transform
// wrapper, so these storage-level kernels must not touch it.
void reject_transform(const Tensor& t, const char* op) {
    const auto impl = t.unsafeGetTensorImpl();
    if (impl && impl->is_batched()) {
        TP_THROW(NotImplementedError, std::string(op),
                 " is not supported for tensors inside an active transform layer");
    }
}

// Alias that shares the source's storage and version counter so writes
// through the alias stay tracked; autograd metadata is not carried over.
Tensor alias_sharing_version(const Tensor& self) {
    const auto impl = self.unsafeGetTensorImpl();
    Tensor out(impl->storage(), impl->sizes().vec(), impl->strides().vec(),
               impl->dtype(), impl->storage_offset());
    out.unsafeGetTensorImpl()->share_version_counter(*impl);
    return out;
}

// Native kernel for dual creation: the dual is an alias of the primal.  The
// tangent is validated against the primal's forward state and attached to
// the result here, so every entry path through the dispatcher sees the same
// semantics.
Tensor make_dual_native(const Tensor& primal, const Tensor& tangent,
                        int64_t level) {
    TP_CHECK(primal.defined(), "_make_dual expected a defined primal tensor");
    TP_CHECK(level >= 0, "_make_dual expected a non-negative level");
    reject_transform(primal, "_make_dual");
    TP_CHECK(
        !impl::is_fw_grad_defined(primal, static_cast<uint64_t>(level)),
        "Making a dual Tensor based on a Tensor that already has a forward "
        "gradient at the same level ", level, " is not supported.");
    Tensor out = alias_sharing_version(primal);
    impl::set_fw_grad(out, tangent, static_cast<uint64_t>(level),
                      /* is_inplace_op */ false);
    return out;
}

// Primal extraction: returns an alias that carries no forward tangent, so
// arithmetic on the result cannot re-enter forward propagation.
Tensor fw_primal_native(const Tensor& self, int64_t level) {
    TP_CHECK(self.defined(), "_fw_primal expected a defined tensor");
    reject_transform(self, "_fw_primal");
    if (impl::is_fw_grad_defined(self, static_cast<uint64_t>(level))) {
        TP_CHECK(level == 0, "Invalid level given to _fw_primal");
    }
    return alias_sharing_version(self);
}

// Unpack a dual tensor into (primal, tangent).  The tangent is read from the
// forward-gradient storage at the given level and may be undefined.
std::tuple<Tensor, Tensor> unpack_dual_native(const Tensor& dual,
                                              int64_t level) {
    TP_CHECK(dual.defined(), "_unpack_dual expected a defined tensor");
    reject_transform(dual, "_unpack_dual");
    return std::make_tuple(
        alias_sharing_version(dual),
        impl::fw_grad(dual, static_cast<uint64_t>(level)));
}

} // namespace

// The kernels are backend-neutral storage-level code, so they register under
// the composite key and are reachable for every backend through the
// composite fallthrough.
TENSORPLAY_LIBRARY_IMPL(Composite, ForwardAdComposite) {
    m.impl("_make_dual", make_dual_native);
    m.impl("_fw_primal", fw_primal_native);
    m.impl("_unpack_dual", unpack_dual_native);
}

// The composite fallthrough only covers backend keys, but dispatch can land
// directly on the autograd keys (e.g. a vmap batch rule re-dispatching below
// the transform on a payload whose keyset names AutogradCPU).  The same
// storage-level kernels serve there: these ops carry no backward formulas,
// so there is no autograd bookkeeping to add.
TENSORPLAY_LIBRARY_IMPL(AutogradCPU, ForwardAdAutogradCPU) {
    m.impl("_fw_primal", fw_primal_native);
    m.impl("_unpack_dual", unpack_dual_native);
}

TENSORPLAY_LIBRARY_IMPL(AutogradCUDA, ForwardAdAutogradCUDA) {
    m.impl("_fw_primal", fw_primal_native);
    m.impl("_unpack_dual", unpack_dual_native);
}

} // namespace tpx
} // namespace tensorplay
