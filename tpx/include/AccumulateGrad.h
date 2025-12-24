#pragma once
#include "Node.h"
#include "TPXTensor.h"
#include "AutogradMetaInterface.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <iostream>

namespace tensorplay {
namespace tpx {

struct AccumulateGrad : public Node {
    std::weak_ptr<AutogradMetaInterface> meta_;
    std::weak_ptr<tensorplay::TensorImpl> impl_;
    
    explicit AccumulateGrad(const Tensor& t) 
        : meta_(t.autograd_meta()), impl_(t.core().unsafeGetTensorImpl()) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].core().defined()) return {};
        
        if (auto meta = meta_.lock()) {
             Tensor grad = inputs[0];
             
             if (auto impl = impl_.lock()) {
                  if (grad.device() != impl->device()) {
                       TP_THROW(RuntimeError, "Expected all tensors to be on the same device, but found at least two devices, " + 
                           impl->device().toString() + " (param) and " + grad.device().toString() + " (grad)!");
                  }

                  const auto& param_sizes = impl->sizes();
                  Size g_shape = grad.shape();
                  
                  if (g_shape != Size(param_sizes)) {
                       // fprintf(stderr, "AccumulateGrad mismatch: grad %s param %s. Triggering sum.\n", g_shape.toString().c_str(), Size(param_sizes).toString().c_str());
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
             }
             
             meta->accum_grad(grad);
        }
        return {};
    }
};

} // namespace tpx
} // namespace tensorplay
