#pragma once

#include "DispatchStub.h"
#include "Tensor.h"

#include "Exception.h"

#include <optional>
#include <vector>

namespace tensorplay {
namespace cpu {

// Helper to compute output shape for reduction
inline std::vector<int64_t> compute_reduction_shape(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<bool> is_reduced(shape.size(), false);

    for (int64_t d : dims) {
        int64_t dim = d;
        if (dim < 0) dim += shape.size();
        if (dim < 0 || dim >= (int64_t)shape.size()) {
             TP_THROW(RuntimeError, "Dimension out of range");
        }
        is_reduced[dim] = true;
    }

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (is_reduced[i]) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    return out_shape;
}

using sum_fn = Tensor (*)(const Tensor&, DType);
DECLARE_DISPATCH(sum_fn, sum_stub)

using sum_dim_fn = Tensor (*)(const Tensor&, std::vector<int64_t>, bool, DType);
DECLARE_DISPATCH(sum_dim_fn, sum_dim_stub)

using max_fn = Tensor (*)(const Tensor&);
DECLARE_DISPATCH(max_fn, max_stub)

using max_dim_fn = Tensor (*)(const Tensor&, std::vector<int64_t>, bool);
DECLARE_DISPATCH(max_dim_fn, max_dim_stub)

using min_fn = Tensor (*)(const Tensor&);
DECLARE_DISPATCH(min_fn, min_stub)

using min_dim_fn = Tensor (*)(const Tensor&, std::vector<int64_t>, bool);
DECLARE_DISPATCH(min_dim_fn, min_dim_stub)

using prod_fn = Tensor (*)(const Tensor&, DType);
DECLARE_DISPATCH(prod_fn, prod_stub)

using prod_dim_fn = Tensor (*)(const Tensor&, std::vector<int64_t>, bool, DType);
DECLARE_DISPATCH(prod_dim_fn, prod_dim_stub)

using all_fn = Tensor (*)(const Tensor&);
DECLARE_DISPATCH(all_fn, all_stub)

using all_dim_fn = Tensor (*)(const Tensor&, std::vector<int64_t>, bool);
DECLARE_DISPATCH(all_dim_fn, all_dim_stub)

using any_fn = Tensor (*)(const Tensor&);
DECLARE_DISPATCH(any_fn, any_stub)

using any_dim_fn = Tensor (*)(const Tensor&, std::vector<int64_t>, bool);
DECLARE_DISPATCH(any_dim_fn, any_dim_stub)

using argmax_fn = Tensor (*)(const Tensor&, std::optional<int64_t>, bool);
DECLARE_DISPATCH(argmax_fn, argmax_stub)

using argmin_fn = Tensor (*)(const Tensor&, std::optional<int64_t>, bool);
DECLARE_DISPATCH(argmin_fn, argmin_stub)

using median_fn = Tensor (*)(const Tensor&);
DECLARE_DISPATCH(median_fn, median_stub)

} // namespace cpu
} // namespace tensorplay
