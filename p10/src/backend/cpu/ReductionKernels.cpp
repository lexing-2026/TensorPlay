#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "ReductionKernels.h"
#include <optional>
#include <vector>
#include <algorithm>
#include <limits>

namespace tensorplay {
namespace cpu {

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(sum_dim_stub);
DEFINE_DISPATCH(max_stub);
DEFINE_DISPATCH(max_dim_stub);
DEFINE_DISPATCH(min_stub);
DEFINE_DISPATCH(min_dim_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(prod_dim_stub);
DEFINE_DISPATCH(all_stub);
DEFINE_DISPATCH(all_dim_stub);
DEFINE_DISPATCH(any_stub);
DEFINE_DISPATCH(any_dim_stub);
DEFINE_DISPATCH(argmax_stub);
DEFINE_DISPATCH(argmin_stub);
DEFINE_DISPATCH(median_stub);

Tensor sum_kernel(const Tensor& self, DType dtype) {
    return sum_stub(DeviceType::CPU, self, dtype);
}

Tensor sum_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim, DType dtype) {
    return sum_dim_stub(DeviceType::CPU, self, std::move(dims), keepdim, dtype);
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
    DType out_dtype = (dtype == DType::Undefined) ? (isFloatingType(self.dtype()) ? self.dtype() : DType::Float32) : dtype;
    Tensor s = sum_kernel(self, out_dtype);
    return s / Scalar((float)self.numel());
}

Tensor mean_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim, DType dtype) {
    DType out_dtype = (dtype == DType::Undefined) ? (isFloatingType(self.dtype()) ? self.dtype() : DType::Float32) : dtype;
    Tensor s = sum_dim_kernel(self, dims, keepdim, out_dtype);

    int64_t count = 1;
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(self.shape());
    for (int64_t d : dims) {
        if (d < 0) d += shape.size();
        count *= shape[d];
    }

    return s / Scalar((float)count);
}

// Autograd helper for mean.dim.  The forward reduction already returns a
// broadcast-compatible view when keepdim=true; for keepdim=false restore the
// reduced singleton dimensions before expanding to the input shape.  This is
// intentionally a view/pointwise composition so CPU and CUDA share the exact
// same gradient semantics without another reduction kernel.
Tensor mean_dim_backward_kernel(const Tensor& grad_output, const Tensor& self,
                                const std::vector<int64_t>& dims, bool keepdim) {
    std::vector<int64_t> normalized;
    normalized.reserve(dims.size());
    for (int64_t d : dims) {
        if (d < 0) d += self.dim();
        if (d < 0 || d >= self.dim()) {
            TP_THROW(IndexError, "mean.dim backward: dimension out of range");
        }
        normalized.push_back(d);
    }
    std::sort(normalized.begin(), normalized.end());
    if (std::adjacent_find(normalized.begin(), normalized.end()) != normalized.end()) {
        TP_THROW(RuntimeError, "mean.dim backward: duplicate dimensions");
    }
    Tensor expanded = grad_output;
    if (!keepdim) {
        for (int64_t d : normalized) expanded = expanded.unsqueeze(d);
    }
    int64_t count = 1;
    for (int64_t d : normalized) count *= self.size(d);
    Tensor grad = expanded.expand(static_cast<std::vector<int64_t>>(self.shape())) /
                  Scalar(static_cast<float>(count));
    return grad.dtype() == self.dtype() ? grad : grad.to(self.dtype());
}

Tensor max_kernel(const Tensor& self) {
    return max_stub(DeviceType::CPU, self);
}

Tensor max_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim) {
    return max_dim_stub(DeviceType::CPU, self, std::move(dims), keepdim);
}

Tensor min_kernel(const Tensor& self) {
    return min_stub(DeviceType::CPU, self);
}

Tensor min_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim) {
    return min_dim_stub(DeviceType::CPU, self, std::move(dims), keepdim);
}

Tensor prod_kernel(const Tensor& self, DType dtype) {
    return prod_stub(DeviceType::CPU, self, dtype);
}

Tensor prod_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim, DType dtype) {
    return prod_dim_stub(DeviceType::CPU, self, std::move(dims), keepdim, dtype);
}

Tensor all_kernel(const Tensor& self) {
    return all_stub(DeviceType::CPU, self);
}

Tensor all_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim) {
    return all_dim_stub(DeviceType::CPU, self, std::move(dims), keepdim);
}

Tensor any_kernel(const Tensor& self) {
    return any_stub(DeviceType::CPU, self);
}

Tensor any_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim) {
    return any_dim_stub(DeviceType::CPU, self, std::move(dims), keepdim);
}

Tensor argmax_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    return argmax_stub(DeviceType::CPU, self, dim, keepdim);
}

Tensor argmin_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    return argmin_stub(DeviceType::CPU, self, dim, keepdim);
}

Tensor var_kernel(const Tensor& self, int64_t correction) {
    if (self.numel() == 0) return Tensor::empty({}, DType::Float32, self.device()).fill_(Scalar(std::numeric_limits<float>::quiet_NaN()));
    Tensor mean = self.mean();
    Tensor diff = self - mean;
    Tensor sq_diff = diff * diff;
    Tensor sum_sq = sq_diff.sum();
    int64_t n = self.numel();
    double div_val = std::max<double>(0.0, static_cast<double>(n - correction));
    Tensor result = (sum_sq / Scalar(div_val)).to(self.dtype());
    return result;
}

Tensor var_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
    if (self.numel() == 0) return Tensor::empty(compute_reduction_shape(self, dim, keepdim), DType::Float32, self.device()).fill_(Scalar(std::numeric_limits<float>::quiet_NaN()));
    std::vector<int64_t> dims = dim;
    Tensor mean = self.mean(dims, true);
    std::vector<int64_t> dims_sum = dim;
    Tensor diff = self - mean;
    Tensor sq_diff = diff * diff;
    Tensor sum_sq = sq_diff.sum(dims_sum, keepdim);
    int64_t n = 1;
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(self.shape());
    for (int64_t d : dim) {
        if (d < 0) d += shape.size();
        n *= shape[d];
    }
    double div = std::max<double>(0.0, static_cast<double>(n - correction));
    return (sum_sq / Scalar(div)).to(self.dtype());
}

Tensor std_kernel(const Tensor& self, int64_t correction) {
    return var_kernel(self, correction).sqrt();
}

Tensor std_dim_kernel(const Tensor& self, std::vector<int64_t> dim, int64_t correction, bool keepdim) {
    return var_dim_kernel(self, dim, correction, keepdim).sqrt();
}

Tensor norm_kernel(const Tensor& self, double p) {
    if (std::isinf(p)) {
        if (p > 0) return self.abs().max();
        else return self.abs().min();
    }
    return self.abs().pow(Scalar(p)).sum().pow(Scalar(1.0/p));
}

Tensor norm_dim_kernel(const Tensor& self, std::vector<int64_t> dim, double p, bool keepdim) {
    if (std::isinf(p)) {
        if (p > 0) return self.abs().max(dim, keepdim);
        else return self.abs().min(dim, keepdim);
    }
    return self.abs().pow(Scalar(p)).sum(dim, keepdim).pow(Scalar(1.0/p));
}

Tensor median_kernel(const Tensor& self) {
    return median_stub(DeviceType::CPU, self);
}

TENSORPLAY_LIBRARY_IMPL(CPU, ReductionKernels) {
    m.impl("sum", sum_kernel);
    m.impl("sum.dim_IntList", sum_dim_kernel);
    m.impl("mean", mean_kernel);
    m.impl("mean.dim", mean_dim_kernel);
    m.impl("mean_dim_backward", mean_dim_backward_kernel);
    m.impl("max", max_kernel);
    m.impl("max.dim", max_dim_kernel);
    m.impl("min", min_kernel);
    m.impl("min.dim", min_dim_kernel);
    m.impl("prod", prod_kernel);
    m.impl("prod.dim_IntList", prod_dim_kernel);
    m.impl("all", all_kernel);
    m.impl("all.dim", all_dim_kernel);
    m.impl("any", any_kernel);
    m.impl("any.dim", any_dim_kernel);
    m.impl("argmax", argmax_kernel);
    m.impl("argmin", argmin_kernel);
    m.impl("var", var_kernel);
    m.impl("var.dim", var_dim_kernel);
    m.impl("std", std_kernel);
    m.impl("std.dim", std_dim_kernel);
    m.impl("median", median_kernel);
    m.impl("norm", norm_kernel);
    m.impl("norm.dim", norm_dim_kernel);
}

} // namespace cpu
} // namespace tensorplay
