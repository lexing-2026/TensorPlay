#include "autocast_mode.h"
#include "autocast_cast.h"
#include "Autograd.h"

namespace tensorplay {
namespace autocast {

Tensor cached_cast(DType to_type, const Tensor& arg, DeviceType device_type) {
    if (is_eligible(arg, device_type) && (arg.dtype() != to_type)) {
        // Heuristic: Do what Apex does, and cache lower_precision_fp casts of
        // fp32 model weights (leaves).
        bool can_try_cache =
            (to_type == get_lower_precision_fp_from_device_type(device_type) &&
             arg.dtype() == DType::Float32 && arg.requires_grad() &&
             tpx::impl::is_leaf(arg) && is_autocast_cache_enabled());

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
