#include "autocast_mode.h"

#include <memory>
#include <utility>

#include "Node.h"

namespace tensorplay {
namespace tpx {

// Mirrors torch's ToCopyBackward: the autocast cast participates in the
// autograd graph so gradients are cast back to the source dtype.
struct CastBackward : public Node {
    DType src_dtype_;

    explicit CastBackward(DType src_dtype) : src_dtype_(src_dtype) {}

    size_t num_inputs() const override { return 1; }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        return {inputs[0].to(src_dtype_)};
    }
};

} // namespace tpx

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
            // When caching, the cast must always carry a grad_fn so the
            // cached tensor can be reused in grad-enabled contexts (mirrors
            // torch's AutoGradMode(true) around the .to()).
            Tensor casted_arg = cast_with_grad(arg, to_type);
            cache_store(arg.unsafeGetTensorImpl().get(), arg, casted_arg);
            return casted_arg;
        } else {
            return cast_with_grad(arg, to_type);
        }
    } else {
        return arg;
    }
}

Tensor cast_with_grad(const Tensor& tensor, DType to_type) {
    bool record_grad = GradMode::is_enabled() && tensor.requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (record_grad) {
        grad_fn = std::make_shared<tpx::CastBackward>(tensor.dtype());
        grad_fn->add_next_edge_list(tpx::collect_next_edges(tensor));
    }

    Tensor result = tensor.to(to_type);
    if (record_grad && result.defined()) {
        tpx::impl::set_grad_fn(result, grad_fn);
    }
    return result;
}

} // namespace autocast
} // namespace tensorplay
