// Pointwise CUDA kernels: maxmin/ldexp family.
#include "PointwiseCommon.cuh"

namespace tensorplay {
namespace cuda {

template <bool Maximum, typename T>
__device__ inline T maximum_minimum_elem(T a, T b) {
    if (a != a) return a;
    if (b != b) return b;
    if constexpr (Maximum) {
        return a < b ? b : a;
    } else {
        return a < b ? a : b;
    }
}

template <bool Maximum>
Tensor maximum_minimum_cuda_impl(const Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    if (isComplexType(common_dtype)) {
        TP_THROW(RuntimeError, "maximum/minimum is not implemented for complex tensors");
    }
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    if (result.numel() == 0) return result;
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(a)
        .add_input(b)
        .build();

    #define MAXMIN_CASE(ctype, name) \
        case DType::name: \
            gpu_kernel(iter, [] __host__ __device__(ctype lhs, ctype rhs) -> ctype { \
                return maximum_minimum_elem<Maximum>(lhs, rhs); \
            }); \
            break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(MAXMIN_CASE)
        default: TP_THROW(NotImplementedError, "CUDA maximum/minimum: unsupported dtype");
    }
    #undef MAXMIN_CASE
    return result;
}

Tensor maximum_cuda(const Tensor& self, const Tensor& other) {
    return maximum_minimum_cuda_impl<true>(self, other);
}

Tensor minimum_cuda(const Tensor& self, const Tensor& other) {
    return maximum_minimum_cuda_impl<false>(self, other);
}

// fmax/fmin: a NaN operand yields the other input verbatim; two NaNs stay NaN.
// Integral types have no NaN and reduce to the plain comparison; Half/BFloat16
// are evaluated through the float overloads below.  double keeps its own
// overload so the comparison never narrows to float.
template <bool Maximum, typename T>
__device__ inline T fmaxfmin_elem(T a, T b) {
    if constexpr (Maximum) {
        return a < b ? b : a;
    } else {
        return a < b ? a : b;
    }
}

template <bool Maximum>
__device__ inline float fmaxfmin_elem(float a, float b) {
    if (a != a) return b;
    if (b != b) return a;
    if constexpr (Maximum) return a < b ? b : a;
    else return a < b ? a : b;
}

template <bool Maximum>
__device__ inline double fmaxfmin_elem(double a, double b) {
    if (a != a) return b;
    if (b != b) return a;
    if constexpr (Maximum) return a < b ? b : a;
    else return a < b ? a : b;
}

template <bool Maximum>
__device__ inline Half fmaxfmin_elem(Half a, Half b) {
    return Half(fmaxfmin_elem<Maximum>(static_cast<float>(a), static_cast<float>(b)));
}

template <bool Maximum>
__device__ inline BFloat16 fmaxfmin_elem(BFloat16 a, BFloat16 b) {
    return BFloat16(fmaxfmin_elem<Maximum>(static_cast<float>(a), static_cast<float>(b)));
}

template <bool Maximum>
Tensor fmaxfmin_cuda_impl(const Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    if (isComplexType(common_dtype)) {
        TP_THROW(RuntimeError, "fmax/fmin is not implemented for complex tensors");
    }
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    if (result.numel() == 0) return result;
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(a)
        .add_input(b)
        .build();

    #define FMAXMIN_CASE(ctype, name) \
        case DType::name: \
            gpu_kernel(iter, [] __host__ __device__(ctype lhs, ctype rhs) -> ctype { \
                return fmaxfmin_elem<Maximum>(lhs, rhs); \
            }); \
            break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(FMAXMIN_CASE)
        default: TP_THROW(NotImplementedError, "CUDA fmax/fmin: unsupported dtype");
    }
    #undef FMAXMIN_CASE
    return result;
}

Tensor fmax_cuda(const Tensor& self, const Tensor& other) {
    return fmaxfmin_cuda_impl<true>(self, other);
}

Tensor fmin_cuda(const Tensor& self, const Tensor& other) {
    return fmaxfmin_cuda_impl<false>(self, other);
}

Tensor ldexp_cuda(const Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    const int64_t n = result.numel();
    if (n == 0) return result;
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_const_input(a)
        .add_const_input(b)
        .build();

    #define LDEXP_CASE(ctype, name) \
        case DType::name: \
            gpu_kernel(iter, [] __host__ __device__(ctype x, ctype exponent) -> ctype { \
                return ldexp_element(x, exponent); \
            }); \
            break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(LDEXP_CASE)
        default: TP_THROW(NotImplementedError, "CUDA ldexp: unsupported dtype");
    }
    #undef LDEXP_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, PointwiseKernels) {
    m.impl("maximum", maximum_cuda);
    m.impl("minimum", minimum_cuda);

    m.impl("fmax", fmax_cuda);
    m.impl("fmin", fmin_cuda);
    m.impl("ldexp", ldexp_cuda);
    m.impl("ldexp.Tensor", ldexp_cuda);
}

} // namespace cuda
} // namespace tensorplay
