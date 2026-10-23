#include "ReduceEngine.cuh"

#include "Complex.h"
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

using namespace reduce_family;

namespace {

template <typename T, typename OutT = T>
Tensor sum_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::zeros(reduction_output_shape(input, spec, keepdim), dtype, input.device());
    }
    return run_reduction_typed<T, AccT, OutT>(
        input, spec, keepdim, dtype, SumOps<T, AccT, OutT>{}, AccT(0));
}

template <typename T, typename OutT = T>
Tensor nansum_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::zeros(reduction_output_shape(input, spec, keepdim), dtype, input.device());
    }
    return run_reduction_typed<T, AccT, OutT>(
        input, spec, keepdim, dtype, NanSumOps<T, AccT, OutT>{}, AccT(0));
}

template <typename T, typename OutT = T>
Tensor prod_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::ones(reduction_output_shape(input, spec, keepdim), dtype, input.device());
    }
    return run_reduction_typed<T, AccT, OutT>(
        input, spec, keepdim, dtype, ProdOps<T, AccT, OutT>{}, AccT(1));
}

} // namespace

// Sum
Tensor sum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isIntegralType(self.dtype(), true) ? DType::Int64 : self.dtype();
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    // Cast-and-reduce in a single kernel for the promoted dtype pairs: the
    // generic engine reads the narrow input and accumulates in the wider
    // type, so no materialized copy is needed.
    if (self.dtype() == DType::Float16 && out_dtype == DType::Float32) {
        return sum_same_dtype<Half, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::BFloat16 && out_dtype == DType::Float32) {
        return sum_same_dtype<BFloat16, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::ComplexHalf || self.dtype() == DType::BComplex32) {
        // Accumulate in complex<float> without promoting the input first.
        if (out_dtype == DType::ComplexFloat) {
            return sum_same_dtype<tensorplay::complex<Half>, tensorplay::complex<float>>(
                self, spec, keepdim, out_dtype);
        }
        return sum_same_dtype<tensorplay::complex<Half>, tensorplay::complex<Half>>(
            self, spec, keepdim, out_dtype);
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec input_spec = make_reduction_spec(input, dim);
    // Complex accumulates in its own width via the generic ops (+ only).
    if (input.dtype() == DType::ComplexFloat) {
        return sum_same_dtype<tensorplay::complex<float>>(input, input_spec, keepdim, out_dtype);
    }
    if (input.dtype() == DType::ComplexDouble) {
        return sum_same_dtype<tensorplay::complex<double>>(input, input_spec, keepdim, out_dtype);
    }
    TP_DISPATCH_REDUCTION(sum_same_dtype, input.dtype(), input, input_spec, keepdim, out_dtype);
}

Tensor sum_kernel(const Tensor& self, DType dtype) {
    return sum_dim_kernel(self, {}, false, dtype);
}

Tensor nansum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                         bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isFloatingOrComplexType(self.dtype()) ? self.dtype() : DType::Int64;
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    if (self.dtype() == DType::Float16 && out_dtype == DType::Float32) {
        return nansum_same_dtype<Half, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::BFloat16 && out_dtype == DType::Float32) {
        return nansum_same_dtype<BFloat16, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::ComplexHalf || self.dtype() == DType::BComplex32) {
        if (out_dtype == DType::ComplexFloat) {
            return nansum_same_dtype<tensorplay::complex<Half>, tensorplay::complex<float>>(
                self, spec, keepdim, out_dtype);
        }
        return nansum_same_dtype<tensorplay::complex<Half>, tensorplay::complex<Half>>(
            self, spec, keepdim, out_dtype);
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec input_spec = make_reduction_spec(input, dim);
    if (input.dtype() == DType::ComplexFloat) {
        return nansum_same_dtype<tensorplay::complex<float>>(
            input, input_spec, keepdim, out_dtype);
    }
    if (input.dtype() == DType::ComplexDouble) {
        return nansum_same_dtype<tensorplay::complex<double>>(
            input, input_spec, keepdim, out_dtype);
    }
    TP_DISPATCH_REDUCTION(nansum_same_dtype, input.dtype(), input, input_spec, keepdim, out_dtype);
}

// Prod
Tensor prod_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isIntegralType(self.dtype(), true) ? DType::Int64 : self.dtype();
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    if (self.dtype() == DType::Float16 && out_dtype == DType::Float32) {
        return prod_same_dtype<Half, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::BFloat16 && out_dtype == DType::Float32) {
        return prod_same_dtype<BFloat16, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::ComplexHalf || self.dtype() == DType::BComplex32) {
        if (out_dtype == DType::ComplexFloat) {
            return prod_same_dtype<tensorplay::complex<Half>, tensorplay::complex<float>>(
                self, spec, keepdim, out_dtype);
        }
        return prod_same_dtype<tensorplay::complex<Half>, tensorplay::complex<Half>>(
            self, spec, keepdim, out_dtype);
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec input_spec = make_reduction_spec(input, dim);
    if (input.dtype() == DType::ComplexFloat) {
        return prod_same_dtype<tensorplay::complex<float>>(input, input_spec, keepdim, out_dtype);
    }
    if (input.dtype() == DType::ComplexDouble) {
        return prod_same_dtype<tensorplay::complex<double>>(input, input_spec, keepdim, out_dtype);
    }
    TP_DISPATCH_REDUCTION(prod_same_dtype, input.dtype(), input, input_spec, keepdim, out_dtype);
}

Tensor prod_kernel(const Tensor& self, DType dtype) {
    return prod_dim_kernel(self, {}, false, dtype);
}

Tensor nansum_cuda2(const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    return nansum_dim_kernel(self, dim_in, keepdim, DType::Undefined);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceSumProdKernels) {
    m.impl("nansum", nansum_cuda2);
    m.impl("sum", sum_kernel);
    m.impl("sum.dim_IntList", sum_dim_kernel);
    m.impl("prod", prod_kernel);
    m.impl("prod.dim_IntList", prod_dim_kernel);
}

} // namespace cuda
} // namespace tensorplay
