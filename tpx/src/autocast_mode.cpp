#include "autocast_mode.h"
#include "autocast_cast.h"
#include "Autograd.h"

namespace tensorplay {
namespace autocast {

Tensor cached_cast(DType to_type, const Tensor& arg, DeviceType device_type) {
    if (is_eligible(arg, device_type) && (arg.dtype() != to_type)) {
        // callers hand GEMMs a fresh `weight.t()` view every call, so caching
        // on the view's TensorImpl* never hits and the fallback would redo a
        // strided scalar cast each iteration.  Cast the dense parent instead
        // -- its address is stable, so the weight-cache hits across calls --
        // and re-apply the transpose as a zero-copy view of the result.
        if (!arg.is_contiguous() && arg.dim() == 2 && arg.dtype() == DType::Float32 &&
            arg.stride(0) == 1 && arg.stride(1) == arg.size(0)) {
            // Stable-key cache: the view's data_ptr aliases the parent
            // parameter's storage and views share its version counter.
            Tensor hit = cache_lookup_ptr(arg.data_ptr(), arg);
            if (hit.defined()) {
                return hit.t();
            }
            Tensor dense = arg.t();
            Tensor casted = cached_cast(to_type, dense, device_type);
            if (casted.defined()) {
                if (!casted.is_contiguous()) {
                    casted = casted.contiguous();
                }
                cache_store_ptr(arg.data_ptr(), arg, casted);
                return casted.t();
            }
        }
        bool can_try_cache =
            (to_type == get_lower_precision_fp_from_device_type(device_type) &&
             arg.dtype() == DType::Float32 && is_autocast_cache_enabled());

        if (can_try_cache) {
            // Cache eligible fp32 leaves that require grad. Non-grad tensors
            // are safe to cache too: the source version counter invalidates a
            // stale copy after an in-place mutation, avoiding repeated
            // conversion of every weight during a no-grad forward.
            if (arg.requires_grad() && !tpx::impl::is_leaf(arg)) {
                can_try_cache = false;
            }
        }

        if (can_try_cache) {
            Tensor hit = cache_lookup(arg.unsafeGetTensorImpl().get());
            if (hit.defined()) {
                return hit;
            }
            // The cast must be differentiable (ToCopyBackward), so route it
            // through tpx::to -- the raw Tensor::to bypasses autograd and
            // would silently cut the graph at every autocast boundary.
            Tensor casted_arg = tpx::to(arg, to_type);
            cache_store(arg.unsafeGetTensorImpl().get(), arg, casted_arg);
            return casted_arg;
        } else {
            return tpx::to(arg, to_type);
        }
    } else {
        return arg;
    }
}

} // namespace autocast
} // namespace tensorplay
