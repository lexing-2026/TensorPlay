#include "ReduceKernels.cuh"

namespace tensorplay {
namespace cuda {

namespace {


std::tuple<Tensor, Tensor> var_mean_cuda(const Tensor& self, const std::vector<int64_t>& dim,
                                         bool unbiased, bool keepdim) {
    // One Welford pass yields both; a separate mean pass would re-read the
    // whole input for a value the accumulator already holds.
    return var_mean_dim_kernel(self, dim, unbiased ? 1 : 0, keepdim);
}

std::tuple<Tensor, Tensor> std_mean_cuda(const Tensor& self, const std::vector<int64_t>& dim,
                                         bool unbiased, bool keepdim) {
    auto vm = var_mean_cuda(self, dim, unbiased, keepdim);
    return {std::get<0>(vm).sqrt(), std::get<1>(vm)};
}




// NaN-aware mean: counts the finite elements per slice and divides.
__global__ void nanmean_zero_mask_kernel(int64_t n, const float* count, float* data) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (count[i] == 0.0f) data[i] = std::numeric_limits<float>::quiet_NaN();
    }
}


Tensor nanmean_cuda(const Tensor& self, std::optional<int64_t> dim_opt, bool keepdim,
                    std::optional<DType> dtype) {
    DType acc_dt = dtype.value_or(DType::Undefined);
    if (!isFloatingType(self.dtype()) && !isComplexType(self.dtype())) {
        TP_THROW(TypeError,
                 "nanmean(): expected input to have floating point or complex dtype but got ",
                 toString(self.dtype()));
    }
    if (acc_dt != DType::Undefined && !isFloatingType(acc_dt) &&
        !isComplexType(acc_dt)) {
        TP_THROW(TypeError,
                 "nanmean(): could not infer output dtype. Optional dtype must be either a floating point or complex dtype. Got: ",
                 toString(acc_dt));
    }
    Tensor x = self;
    if (acc_dt != DType::Undefined && x.dtype() != acc_dt) {
        x = x.to(acc_dt);
    } else if (isReducedFloatingType(x.dtype()) && acc_dt == DType::Undefined) {
        x = x.to(DType::Float32);
    }
    std::vector<int64_t> dims;
    if (dim_opt.has_value()) dims.push_back(*dim_opt);
    else {
        // global reduction over every dimension
        for (int64_t i = 0; i < x.dim(); ++i) dims.push_back(i);
    }
    Tensor total = nansum_cuda2(x, dims, keepdim);
    Tensor valid = ops::isnan(x).logical_not();
    Tensor count = sum_dim_kernel(valid.to(DType::Float32), dims, keepdim, DType::Float32);
    Tensor quot = total.div(count);
    Tensor zero = count.eq(Scalar(0.0f));
    Tensor result = quot.masked_fill(zero, Scalar(std::numeric_limits<double>::quiet_NaN()));
    return result.to(acc_dt != DType::Undefined ? acc_dt : total.dtype());
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceMomentKernels) {
    m.impl("var_mean", var_mean_cuda);
    m.impl("std_mean", std_mean_cuda);
    m.impl("nanmean", nanmean_cuda);
}

} // namespace cuda
} // namespace tensorplay
