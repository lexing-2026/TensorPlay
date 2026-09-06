#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "TensorIteratorOps.h"
#include "Exception.h"
#include "Parallel.h"
#include "cpu/VecUnary.h"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>

namespace tensorplay {
namespace cpu {

#if defined(__x86_64__) || defined(__i386__)
namespace {

__attribute__((target("avx512f")))
inline void where_f32_avx512(const bool* condition, const float* self,
                             const float* other, float* result,
                             int64_t begin, int64_t end) {
    int64_t i = begin;
    const __m128i zero = _mm_setzero_si128();
    for (; i + 16 <= end; i += 16) {
        const __m128i c = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(condition + i));
        const int zero_bits = _mm_movemask_epi8(_mm_cmpeq_epi8(c, zero));
        const __mmask16 mask = static_cast<__mmask16>(~zero_bits);
        const __m512 a = _mm512_loadu_ps(self + i);
        const __m512 b = _mm512_loadu_ps(other + i);
        _mm512_storeu_ps(result + i, _mm512_mask_blend_ps(mask, b, a));
    }
    for (; i < end; ++i) {
        result[i] = condition[i] ? self[i] : other[i];
    }
}

__attribute__((target("avx512f")))
inline void where_f64_avx512(const bool* condition, const double* self,
                             const double* other, double* result,
                             int64_t begin, int64_t end) {
    int64_t i = begin;
    const __m128i zero = _mm_setzero_si128();
    for (; i + 8 <= end; i += 8) {
        const __m128i c = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(condition + i));
        const int zero_bits = _mm_movemask_epi8(_mm_cmpeq_epi8(c, zero));
        const __mmask8 mask = static_cast<__mmask8>(~zero_bits);
        const __m512d a = _mm512_loadu_pd(self + i);
        const __m512d b = _mm512_loadu_pd(other + i);
        _mm512_storeu_pd(result + i, _mm512_mask_blend_pd(mask, b, a));
    }
    for (; i < end; ++i) {
        result[i] = condition[i] ? self[i] : other[i];
    }
}

} // namespace
#endif

// (result_type(int_tensor, 2.5) == Float32), and the scalar must never be
// truncated into the tensor's dtype before comparing.
static DType result_type_with_scalar(const Tensor& t, const Scalar& s) {
    DType td = t.dtype();
    if (s.dtype() == DType::Bool) return td;
    if (isComplexType(s.dtype())) {
        // float64 widens to complex128.
        if (isFloatingOrComplexType(td)) return td;
        return promoteTypes(td, DType::ComplexFloat);
    }
    if (isFloatingType(s.dtype())) {
        if (isFloatingType(td)) return td;   // half/bf16 stay reduced
        return DType::Float32;
    }
    // int scalar: float tensors keep their dtype; integral tensors keep theirs
    return td;
}

// Helper for comparison ops.
// kEquality=false: ordering ops (lt/le/gt/ge) are undefined over complex.
template<bool kEquality, typename Op>
Tensor comparison_kernel_impl(const Tensor& self, const Tensor& other, Op op) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));

    // Result is always Bool
    Tensor result = Tensor::empty(out_shape, DType::Bool, self.device());

    // For comparison, we usually don't promote types to a common type for the operation,
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());

    if (!kEquality && isComplexType(common_dtype)) {
        TP_THROW(NotImplementedError, "comparison not implemented for '", toString(common_dtype), "'");
    }

    Tensor self_casted = (self.dtype() == common_dtype) ? self : self.to(common_dtype);
    Tensor other_casted = (other.dtype() == common_dtype) ? other : other.to(common_dtype);

    // TensorIterator owns broadcast/reorder/coalesce/parallelism; no
    // materialized expansion needed.
    if constexpr (kEquality) {
        ti_apply_equality(result, self_casted, other_casted, common_dtype, op);
    } else {
        ti_apply_compare(result, self_casted, other_casted, common_dtype, op);
    }

    return result;
}

Tensor eq_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<true>(self, other, [](auto a, auto b) { return a == b; });
}

Tensor ne_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<true>(self, other, [](auto a, auto b) { return a != b; });
}

Tensor lt_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<false>(self, other, [](auto a, auto b) { return a < b; });
}

Tensor le_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<false>(self, other, [](auto a, auto b) { return a <= b; });
}

Tensor gt_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<false>(self, other, [](auto a, auto b) { return a > b; });
}

Tensor ge_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<false>(self, other, [](auto a, auto b) { return a >= b; });
}

// Scalar versions: promote the scalar with weak-scalar rules instead of
// casting it into self.dtype() (which truncated e.g. eq(2.5) on int tensors)
#define DEFINE_CMP_SCALAR_KERNEL(NAME) \
Tensor NAME##_scalar_kernel(const Tensor& self, Scalar other) { \
    DType common = result_type_with_scalar(self, other); \
    Tensor other_t = Tensor::full({}, other, common, self.device()); \
    return NAME##_tensor_kernel(self.to(common), other_t); \
}

DEFINE_CMP_SCALAR_KERNEL(eq)
DEFINE_CMP_SCALAR_KERNEL(ne)
DEFINE_CMP_SCALAR_KERNEL(lt)
DEFINE_CMP_SCALAR_KERNEL(le)
DEFINE_CMP_SCALAR_KERNEL(gt)
DEFINE_CMP_SCALAR_KERNEL(ge)
#undef DEFINE_CMP_SCALAR_KERNEL

template <typename Op>
Tensor where_kernel_impl(const Tensor& condition, const Tensor& self,
                         const Tensor& other, Op op) {
    if (condition.dtype() != DType::Bool) {
        TP_THROW(TypeError, "where condition must be a boolean tensor");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(condition.shape()),
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    Tensor self_casted = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor other_casted = other.dtype() == common_dtype ? other : other.to(common_dtype);

    // No-broadcast contiguous case: one flat select loop, parallelized and
    // left to the auto-vectorizer (compare + blend lowers to cmov/blend ops).
    const bool flat = out_shape == static_cast<std::vector<int64_t>>(condition.shape()) &&
                      out_shape == static_cast<std::vector<int64_t>>(self_casted.shape()) &&
                      out_shape == static_cast<std::vector<int64_t>>(other_casted.shape()) &&
                      condition.is_contiguous() && self_casted.is_contiguous() &&
                      other_casted.is_contiguous() && result.is_contiguous();
    if (flat) {
        const int64_t n = result.numel();
        const bool* cond = condition.data_ptr<bool>();
#if defined(__x86_64__) || defined(__i386__)
        if (vecunary::avx512_available() && common_dtype == DType::Float32) {
            const float* a = self_casted.data_ptr<float>();
            const float* b = other_casted.data_ptr<float>();
            float* o = result.data_ptr<float>();
            tensorplay::parallel::parallel_for(0, n, 8192,
                [&](int64_t begin, int64_t end) {
                    where_f32_avx512(cond, a, b, o, begin, end);
                });
            return result;
        }
        if (vecunary::avx512_available() && common_dtype == DType::Float64) {
            const double* a = self_casted.data_ptr<double>();
            const double* b = other_casted.data_ptr<double>();
            double* o = result.data_ptr<double>();
            tensorplay::parallel::parallel_for(0, n, 8192,
                [&](int64_t begin, int64_t end) {
                    where_f64_avx512(cond, a, b, o, begin, end);
                });
            return result;
        }
#endif
        bool done = false;
        #define TP_WHERE_FLAT_CASE(ctype, name) \
        case DType::name: { \
            const ctype* a = self_casted.data_ptr<ctype>(); \
            const ctype* b = other_casted.data_ptr<ctype>(); \
            ctype* o = result.data_ptr<ctype>(); \
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t lo, int64_t hi) { \
                for (int64_t i = lo; i < hi; ++i) o[i] = cond[i] ? a[i] : b[i]; \
            }); \
            done = true; \
            break; \
        }
        switch (common_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_WHERE_FLAT_CASE)
            default: break;
        }
        #undef TP_WHERE_FLAT_CASE
        if (done) return result;
    }

    auto condition_strides = broadcast_strides(condition, out_shape);
    auto self_strides = broadcast_strides(self_casted, out_shape);
    auto other_strides = broadcast_strides(other_casted, out_shape);

    #define WHERE_CASE(ctype, name) \
        case DType::name: { \
            apply_ternary_op_recursive_mixed<ctype, bool, ctype>( \
                result.data_ptr<ctype>(), result.strides(), condition, condition_strides, \
                self_casted, self_strides, other_casted, other_strides, \
                0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(WHERE_CASE)
        case DType::ComplexFloat: {
            apply_ternary_op_recursive_mixed<tensorplay::complex<float>, bool, tensorplay::complex<float>>( \
                result.data_ptr<tensorplay::complex<float>>(), result.strides(), condition, condition_strides, \
                self_casted, self_strides, other_casted, other_strides, \
                0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
        case DType::ComplexDouble: {
            apply_ternary_op_recursive_mixed<tensorplay::complex<double>, bool, tensorplay::complex<double>>( \
                result.data_ptr<tensorplay::complex<double>>(), result.strides(), condition, condition_strides, \
                self_casted, self_strides, other_casted, other_strides, \
                0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
        default: TP_THROW(TypeError, "where: unsupported dtype");
    }
    #undef WHERE_CASE
    return result;
}

Tensor where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other) {
    return where_kernel_impl(condition, self, other,
        [](bool select_self, auto a, auto b) { return select_self ? a : b; });
}

Tensor where_scalar_self_cpu(const Tensor& condition, Scalar self, const Tensor& other) {
    DType common_dtype = result_type(self, other.dtype());
    Tensor self_tensor = Tensor::full({}, self, common_dtype, other.device());
    return where_cpu(condition, self_tensor, other);
}

Tensor where_scalar_other_cpu(const Tensor& condition, const Tensor& self, Scalar other) {
    DType common_dtype = result_type(other, self.dtype());
    Tensor other_tensor = Tensor::full({}, other, common_dtype, self.device());
    return where_cpu(condition, self, other_tensor);
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

Tensor where_scalar_scalar_cpu(const Tensor& condition, Scalar self, Scalar other) {
    DType common_dtype = where_scalar_dtype(self, other);
    Tensor self_tensor = Tensor::full({}, self, common_dtype, condition.device());
    Tensor other_tensor = Tensor::full({}, other, common_dtype, condition.device());
    return where_cpu(condition, self_tensor, other_tensor);
}

template <typename Op>
Tensor maximum_minimum_kernel_impl(const Tensor& self, const Tensor& other, Op op) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    if (isComplexType(common_dtype)) {
        TP_THROW(RuntimeError, "maximum/minimum is not implemented for complex tensors");
    }
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
    ti_apply_binary(result, a, b, op);
    return result;
}

struct MaximumOp {
    template <typename T>
    T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(a)) return a;
            if (std::isnan(b)) return b;
        }
        return a < b ? b : a;
    }
};

struct MinimumOp {
    template <typename T>
    T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(a)) return a;
            if (std::isnan(b)) return b;
        }
        return a < b ? a : b;
    }
};

Tensor maximum_cpu(const Tensor& self, const Tensor& other) {
    return maximum_minimum_kernel_impl(self, other, MaximumOp());
}

Tensor minimum_cpu(const Tensor& self, const Tensor& other) {
    return maximum_minimum_kernel_impl(self, other, MinimumOp());
}

// fmax/fmin: a NaN operand yields the other input verbatim (IEEE fmax/fmin
// semantics); two NaNs stay NaN.  Integral types have no NaN and reduce to
// the plain max/min comparison.
template <bool Maximum>
struct FmaxFminOp {
    template <typename T>
    T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(a)) return b;
            if (std::isnan(b)) return a;
        }
        if constexpr (Maximum) {
            return a < b ? b : a;
        } else {
            return a < b ? a : b;
        }
    }
};

Tensor fmax_cpu(const Tensor& self, const Tensor& other) {
    return maximum_minimum_kernel_impl(self, other, FmaxFminOp<true>());
}

Tensor fmin_cpu(const Tensor& self, const Tensor& other) {
    return maximum_minimum_kernel_impl(self, other, FmaxFminOp<false>());
}

TENSORPLAY_LIBRARY_IMPL(CPU, ComparisonKernels) {
    m.impl("eq.Tensor", eq_tensor_kernel);
    m.impl("eq.Scalar", eq_scalar_kernel);
    m.impl("ne.Tensor", ne_tensor_kernel);
    m.impl("ne.Scalar", ne_scalar_kernel);
    m.impl("lt.Tensor", lt_tensor_kernel);
    m.impl("lt.Scalar", lt_scalar_kernel);
    m.impl("le.Tensor", le_tensor_kernel);
    m.impl("le.Scalar", le_scalar_kernel);
    m.impl("gt.Tensor", gt_tensor_kernel);
    m.impl("gt.Scalar", gt_scalar_kernel);
    m.impl("ge.Tensor", ge_tensor_kernel);
    m.impl("ge.Scalar", ge_scalar_kernel);
    m.impl("where.self", where_cpu);
    m.impl("where.ScalarSelf", where_scalar_self_cpu);
    m.impl("where.ScalarOther", where_scalar_other_cpu);
    m.impl("where.Scalar", where_scalar_scalar_cpu);
    m.impl("maximum", maximum_cpu);
    m.impl("minimum", minimum_cpu);
    m.impl("fmax", fmax_cpu);
    m.impl("fmin", fmin_cpu);
}

} // namespace cpu
} // namespace tensorplay
