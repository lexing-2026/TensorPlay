// Pointwise CUDA kernels: comparison/where family.
#include "PointwiseCommon.cuh"

namespace tensorplay {
namespace cuda {

template <bool AllowFloat8, typename Functor>
Tensor comparison_op_kernel(const Tensor& self, const Tensor& other, Functor functor) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, DType::Bool, self.device());
    Tensor a = (self.dtype() == common_dtype) ? self : self.to(common_dtype);
    Tensor b = (other.dtype() == common_dtype) ? other : other.to(common_dtype);
    if (result.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(result)
        .add_input(a)
        .add_input(b)
        .build();

    #define COMP_CASE(ctype, name) \
    case DType::name: \
        gpu_kernel(iter, [functor] __host__ __device__(ctype lhs, ctype rhs) -> bool { \
            return functor(lhs, rhs); \
        }); \
        break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(COMP_CASE)
        default:
            if constexpr (AllowFloat8) {
                switch (common_dtype) {
                    TENSORPLAY_FORALL_FP8_TYPES(COMP_CASE)
                    default: TP_THROW(TypeError, "CUDA comparison: Unsupported dtype");
                }
            } else {
                TP_THROW(TypeError, "CUDA comparison: Unsupported dtype");
            }
    }
    #undef COMP_CASE
    return result;
}

template <bool AllowFloat8, typename Functor>
Tensor comparison_scalar_op_kernel(const Tensor& self, Scalar other, Functor functor) {
    DType common = result_type_with_scalar_cuda(self, other);
    Tensor in = (self.dtype() == common) ? self : self.to(common);
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(in.shape()), DType::Bool, self.device());
    if (in.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(result)
        .add_input(in)
        .build();

    #define COMP_SCALAR_CASE(ctype, name) \
    case DType::name: { \
        const ctype rhs = other.to<ctype>(); \
        gpu_kernel(iter, [functor, rhs] __host__ __device__(ctype lhs) -> bool { \
            return functor(lhs, rhs); \
        }); \
        break; \
    }
    switch (common) {
        TENSORPLAY_FORALL_SCALAR_TYPES(COMP_SCALAR_CASE)
        default:
            if constexpr (AllowFloat8) {
                switch (common) {
                    TENSORPLAY_FORALL_FP8_TYPES(COMP_SCALAR_CASE)
                    default: TP_THROW(TypeError, "CUDA comparison: Unsupported dtype");
                }
            } else {
                TP_THROW(TypeError, "CUDA comparison: Unsupported dtype");
            }
    }
    #undef COMP_SCALAR_CASE
    return result;
}

struct EqFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a == b; } };
struct NeFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a != b; } };
struct LtFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a < b; } };
struct LeFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a <= b; } };
struct GtFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a > b; } };
struct GeFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a >= b; } };

template <typename complex_t, typename math_t, typename Functor>
Tensor complex_comparison_kernel(const Tensor& self, const Tensor& other,
                                 Functor functor) {
    const DType rd = promoteTypes(self.dtype(), other.dtype());
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    Tensor result = Tensor::empty(out_shape, DType::Bool, self.device());
    if (result.numel() == 0) return result;
    Tensor a = self.to(rd);
    Tensor b = other.to(rd);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(result)
        .add_const_input(a)
        .add_const_input(b)
        .build();
    gpu_kernel(iter, [functor] __host__ __device__(complex_t lhs, complex_t rhs) -> bool {
        return functor(static_cast<math_t>(lhs), static_cast<math_t>(rhs));
    });
    return result;
}

template <typename Functor>
Tensor complex_comparison_kernel(const Tensor& self, const Tensor& other,
                                 Functor functor) {
    const DType rd = promoteTypes(self.dtype(), other.dtype());
    switch (rd) {
        case DType::ComplexHalf:
            return complex_comparison_kernel<tensorplay::complex<Half>,
                                              tensorplay::complex<float>>(
                self, other, functor);
        case DType::ComplexFloat:
            return complex_comparison_kernel<tensorplay::complex<float>,
                                              tensorplay::complex<float>>(
                self, other, functor);
        case DType::ComplexDouble:
            return complex_comparison_kernel<tensorplay::complex<double>,
                                              tensorplay::complex<double>>(
                self, other, functor);
        case DType::BComplex32:
            return complex_comparison_kernel<tensorplay::complex<BFloat16>,
                                              tensorplay::complex<float>>(
                self, other, functor);
        default:
            TP_THROW(NotImplementedError, "CUDA complex comparison: unsupported dtype");
    }
}

Tensor eq_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (isComplexType(promoteTypes(self.dtype(), other.dtype())))
        return complex_comparison_kernel(self, other, EqFunctor{});
    return comparison_op_kernel<true>(self, other, EqFunctor());
}
Tensor ne_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (isComplexType(promoteTypes(self.dtype(), other.dtype())))
        return complex_comparison_kernel(self, other, NeFunctor{});
    return comparison_op_kernel<true>(self, other, NeFunctor());
}
Tensor lt_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel<false>(self, other, LtFunctor()); }
Tensor le_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel<false>(self, other, LeFunctor()); }
Tensor gt_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel<false>(self, other, GtFunctor()); }
Tensor ge_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel<false>(self, other, GeFunctor()); }

Tensor eq_scalar_kernel_cuda(const Tensor& self, Scalar other) {
    if (other.isComplex()) {
        DType rd = isComplexType(self.dtype())
            ? self.dtype()
            : (isFloatingType(self.dtype())
                   ? promoteTypes(toComplexType(self.dtype()), other.dtype())
                   : promoteTypes(DType::ComplexFloat, other.dtype()));
        Tensor o = Tensor::full({}, other, rd, self.device());
        return eq_kernel_cuda(self.to(rd), o);
    }
    return comparison_scalar_op_kernel<true>(self, other, EqFunctor());
}
Tensor ne_scalar_kernel_cuda(const Tensor& self, Scalar other) {
    if (other.isComplex()) {
        DType rd = isComplexType(self.dtype())
            ? self.dtype()
            : (isFloatingType(self.dtype())
                   ? promoteTypes(toComplexType(self.dtype()), other.dtype())
                   : promoteTypes(DType::ComplexFloat, other.dtype()));
        Tensor o = Tensor::full({}, other, rd, self.device());
        return ne_kernel_cuda(self.to(rd), o);
    }
    return comparison_scalar_op_kernel<true>(self, other, NeFunctor());
}
Tensor lt_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel<false>(self, other, LtFunctor()); }
Tensor le_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel<false>(self, other, LeFunctor()); }
Tensor gt_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel<false>(self, other, GtFunctor()); }
Tensor ge_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel<false>(self, other, GeFunctor()); }

template <typename T>
void where_loop(TensorIterator& iter) {
    gpu_kernel(iter, [] __host__ __device__(bool condition, T self_value, T other_value) -> T {
        return condition ? self_value : other_value;
    });
}

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

Tensor where_cuda(const Tensor& condition, const Tensor& self, const Tensor& other) {
    if (condition.dtype() != DType::Bool) {
        TP_THROW(TypeError, "where condition must be a boolean tensor");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(condition.shape()),
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    if (result.numel() == 0) return result;
    Tensor self_casted = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor other_casted = other.dtype() == common_dtype ? other : other.to(common_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(result)
        .add_const_input(condition)
        .add_const_input(self_casted)
        .add_const_input(other_casted)
        .build();

#define WHERE_CASE(ctype, name) \
    case DType::name: where_loop<ctype>(iter); break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(WHERE_CASE)
        case DType::ComplexHalf:
            where_loop<tensorplay::complex<Half>>(iter);
            break;
        case DType::ComplexFloat:
            where_loop<tensorplay::complex<float>>(iter);
            break;
        case DType::ComplexDouble:
            where_loop<tensorplay::complex<double>>(iter);
            break;
        case DType::BComplex32:
            where_loop<tensorplay::complex<BFloat16>>(iter);
            break;
        default: TP_THROW(NotImplementedError, "CUDA where: unsupported dtype");
    }
    #undef WHERE_CASE
    return result;
}

Tensor where_scalar_self_cuda(const Tensor& condition, Scalar self, const Tensor& other) {
    DType common_dtype = result_type(self, other.dtype());
    return where_cuda(condition, Tensor::full({}, self, common_dtype, other.device()), other);
}

Tensor where_scalar_other_cuda(const Tensor& condition, const Tensor& self, Scalar other) {
    DType common_dtype = result_type(other, self.dtype());
    return where_cuda(condition, self, Tensor::full({}, other, common_dtype, self.device()));
}

static DType where_scalar_dtype(const Scalar& self, const Scalar& other) {
    if (self.isComplex() || other.isComplex()) {
        return promoteTypes(self.dtype(), other.dtype());
    }
    if (self.isFloatingPoint() || other.isFloatingPoint()) {
        return self.dtype() == DType::Float64 || other.dtype() == DType::Float64
            ? DType::Float64 : DType::Float32;
    }
    return DType::Int64;
}

Tensor where_scalar_scalar_cuda(const Tensor& condition, Scalar self, Scalar other) {
    DType common_dtype = where_scalar_dtype(self, other);
    return where_cuda(
        condition,
        Tensor::full({}, self, common_dtype, condition.device()),
        Tensor::full({}, other, common_dtype, condition.device()));
}

TENSORPLAY_LIBRARY_IMPL(CUDA, PointwiseKernels) {
    m.impl("eq.Tensor", eq_kernel_cuda);
    m.impl("ne.Tensor", ne_kernel_cuda);
    m.impl("lt.Tensor", lt_kernel_cuda);
    m.impl("le.Tensor", le_kernel_cuda);
    m.impl("gt.Tensor", gt_kernel_cuda);
    m.impl("ge.Tensor", ge_kernel_cuda);

    m.impl("eq.Scalar", eq_scalar_kernel_cuda);
    m.impl("ne.Scalar", ne_scalar_kernel_cuda);
    m.impl("lt.Scalar", lt_scalar_kernel_cuda);
    m.impl("le.Scalar", le_scalar_kernel_cuda);
    m.impl("gt.Scalar", gt_scalar_kernel_cuda);
    m.impl("ge.Scalar", ge_scalar_kernel_cuda);

        m.impl("where.self", where_cuda);
    m.impl("where.ScalarSelf", where_scalar_self_cuda);
    m.impl("where.ScalarOther", where_scalar_other_cuda);
    m.impl("where.Scalar", where_scalar_scalar_cuda);
}

} // namespace cuda
} // namespace tensorplay
