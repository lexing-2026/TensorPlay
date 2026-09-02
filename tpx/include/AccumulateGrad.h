#pragma once
#include "Node.h"
#include "AutogradMeta.h"
#include "InputBuffer.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <cstdint>
#include <iostream>

namespace tensorplay {
namespace tpx {

struct AccumulateGrad : public Node {
    // The leaf tensor this node accumulates gradients into.
    Tensor value_;

    // AccumulateGrad sets sequence_nr to the max value so it's always called
    // ASAP during backwards.
    explicit AccumulateGrad(const Tensor& t)
        : Node(UINT64_MAX), value_(t) {}

    size_t num_inputs() const override { return 1; }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {};

        Tensor grad = inputs[0];

        if (grad.device() != value_.device()) {
            TP_THROW(RuntimeError, "Expected all tensors to be on the same device, but found at least two devices, " +
                value_.device().toString() + " (param) and " + grad.device().toString() + " (grad)!");
        }

        const auto& param_sizes = value_.impl()->sizes();
        // Shape mismatch check without materializing either shape into a
        // heap vector (this node runs once per leaf per backward).
        const size_t p_dim = param_sizes.size();
        bool same_shape = grad.dim() == static_cast<int64_t>(p_dim);
        if (same_shape) {
            for (size_t i = 0; i < p_dim; ++i) {
                if (grad.size(i) != param_sizes[i]) {
                    same_shape = false;
                    break;
                }
            }
        }

        if (!same_shape) {
            const int64_t g_dim = grad.dim();
            std::vector<int64_t> g_sizes(g_dim);
            for (int64_t i = 0; i < g_dim; ++i) {
                g_sizes[static_cast<size_t>(i)] = grad.size(i);
            }
            int64_t p_dim_signed = static_cast<int64_t>(p_dim);

            if (g_dim >= p_dim_signed) {
                std::vector<int64_t> reduce_dims;
                int64_t dim_diff = g_dim - p_dim_signed;

                // 1. Reduce extra leading dimensions
                for (int64_t i = 0; i < dim_diff; ++i) {
                    reduce_dims.push_back(i);
                }

                // 2. Reduce broadcasted dimensions
                for (int64_t i = 0; i < p_dim_signed; ++i) {
                    if (param_sizes[i] == 1 && g_sizes[i + dim_diff] > 1) {
                        reduce_dims.push_back(i + dim_diff);
                    }
                }

                if (!reduce_dims.empty()) {
                    // Sum reduction
                    grad = ops::sum(grad, reduce_dims, false);
                    // Reshape to ensure correct output shape (e.g. restoring dims of size 1)
                    grad = ops::reshape(grad, param_sizes);
                }
            }
        }

        if (auto* meta = impl::get_autograd_meta(value_)) {
            // Materialize strided gradients (e.g. a .t() view produced by a
            // contiguous layout; downstream consumers (foreach optimizers,
            // .numpy()) rely on dense storage.
            if (!grad.is_contiguous()) {
                grad = grad.contiguous();
            }
            meta->accum_grad(grad);
        }
        return {};
    }
};

} // namespace tpx
} // namespace tensorplay
