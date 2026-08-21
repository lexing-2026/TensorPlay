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
    // The leaf tensor this node accumulates gradients into. Mirrors
    // torch::autograd::AccumulateGrad::variable_.
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
        Size g_shape = grad.shape();

        if (g_shape != Size(param_sizes)) {
            std::vector<int64_t> g_sizes(g_shape.begin(), g_shape.end());
            int64_t p_dim = param_sizes.size();
            int64_t g_dim = g_sizes.size();

            if (g_dim >= p_dim) {
                std::vector<int64_t> reduce_dims;
                int64_t dim_diff = g_dim - p_dim;

                // 1. Reduce extra leading dimensions
                for (int64_t i = 0; i < dim_diff; ++i) {
                    reduce_dims.push_back(i);
                }

                // 2. Reduce broadcasted dimensions
                for (int64_t i = 0; i < p_dim; ++i) {
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
            meta->accum_grad(grad);
        }
        return {};
    }
};

} // namespace tpx
} // namespace tensorplay
