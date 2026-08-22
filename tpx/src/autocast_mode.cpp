#include "autocast_mode.h"
#include "autocast_cast.h"

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
            // The cast is differentiable through `to` (ToCopyBackward), so the
            // cached tensor always carries a grad_fn and can be reused in
            // grad-enabled contexts.
            Tensor casted_arg = arg.to(to_type);
            cache_store(arg.unsafeGetTensorImpl().get(), arg, casted_arg);
            return casted_arg;
        } else {
            return arg.to(to_type);
        }
    } else {
        return arg;
    }
}

} // namespace autocast
} // namespace tensorplay
