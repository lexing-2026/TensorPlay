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
#include <cstring>
#include <algorithm>
#include <type_traits>

namespace tensorplay {
namespace cpu {

// One tag per ordering, so the predicate is fixed before the loop and the
// compare folds into a single instruction instead of a branch per vector.
enum class NarrowCmp { Eq, Ne, Lt, Le, Gt, Ge };

template <NarrowCmp C>
inline bool narrow_cmp(float x, float y) {
    switch (C) {
        case NarrowCmp::Eq: return x == y;
        case NarrowCmp::Ne: return x != y;
        case NarrowCmp::Lt: return x < y;
        case NarrowCmp::Le: return x <= y;
        case NarrowCmp::Gt: return x > y;
        case NarrowCmp::Ge: return x >= y;
    }
    return false;
}

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

// Half's widening to float walks a branchy subnormal path, so a comparison
// written over the storage never vectorizes: every lane of it stays scalar.
// Widening a whole vector at a time and comparing in float reaches the same
// domain the per-element loop does, and costs one convert pair per eight lanes.
// Only the Half side needs the convert instruction itself; BFloat16 widens with
// a shift.
inline bool narrow_compare_ready(bool half) {
    if (!vecunary::avx2_available()) return false;
    return !half || vecunary::f16c_available();
}

// Equality is ordered-and-equal, so a NaN answers false; the inequality has to
// count unordered pairs as unequal for the same NaN to answer true.  The
// predicate is picked as an immediate, so the compare itself stays inside the
// function that carries the vector target.
template <NarrowCmp C>
constexpr int narrow_cmp_imm() {
    switch (C) {
        case NarrowCmp::Eq: return _CMP_EQ_OQ;
        case NarrowCmp::Ne: return _CMP_NEQ_UQ;
        case NarrowCmp::Lt: return _CMP_LT_OQ;
        case NarrowCmp::Le: return _CMP_LE_OQ;
        case NarrowCmp::Gt: return _CMP_GT_OQ;
        case NarrowCmp::Ge: return _CMP_GE_OQ;
    }
    return _CMP_EQ_OQ;
}

// Eight storages widened into the float domain the comparison runs in.  Half
// converts through f16c; the other narrow type only needs its bits moved up,
// which is a zero-extend plus a shift rather than the branchy per-element
// widening that kept the loop scalar.
template <bool kHalf>
__attribute__((target("avx2,fma,f16c"), always_inline))
inline __m256 narrow_widen8(const uint16_t* p) {
    if (kHalf) {
        return _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    }
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(p))), 16));
}

template <bool kHalf>
__attribute__((target("avx2,fma,f16c"), always_inline))
inline float narrow_widen1(const uint16_t p) {
    if (kHalf) return _cvtsh_ss(p);
    const uint32_t bits = static_cast<uint32_t>(p) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

template <NarrowCmp C, bool kHalf, bool kBcastB>
__attribute__((target("avx2,fma,f16c")))
inline void narrow_compare_avx2(const uint16_t* a, const uint16_t* b, bool* out,
                                int64_t begin, int64_t end) {
    // A compare lane answers all-zero or all-one, so its low byte is already
    // the 0 or 1 the Bool storage holds; gathering the four low bytes of each
    // half and concatenating them writes eight answers.
    const __m256i pack_pick = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // A single-element right operand is read once and carried in every lane.
    const __m256 vb = kBcastB ? _mm256_set1_ps(narrow_widen1<kHalf>(b[0]))
                              : _mm256_setzero_ps();
    int64_t i = begin;
    for (; i + 8 <= end; i += 8) {
        const __m256 x = narrow_widen8<kHalf>(a + i);
        const __m256 y = kBcastB ? vb : narrow_widen8<kHalf>(b + i);
        const __m256i answers = _mm256_and_si256(
            _mm256_castps_si256(_mm256_cmp_ps(x, y, narrow_cmp_imm<C>())),
            _mm256_set1_epi32(1));
        const __m256i packed = _mm256_shuffle_epi8(answers, pack_pick);
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(out + i),
            _mm_unpacklo_epi32(_mm256_castsi256_si128(packed),
                               _mm256_extracti128_si256(packed, 1)));
    }
    const float only = kBcastB ? narrow_widen1<kHalf>(b[0]) : 0.f;
    for (; i < end; ++i) {
        out[i] = narrow_cmp<C>(narrow_widen1<kHalf>(a[i]),
                               kBcastB ? only : narrow_widen1<kHalf>(b[i]));
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
Tensor dequantize_self_cpu(const Tensor& self);

// A flat, contiguous narrow storage compared against a same-shape storage or
// against one broadcast value takes one masked compare per eight lanes.
// Everything else -- a shaped broadcast, a strided storage, a host without the
// convert pair -- reports failure and stays with the iterator.
template <NarrowCmp C>
bool narrow_compare_path(const Tensor& result, const Tensor& a, const Tensor& b,
                         DType common) {
#if defined(__x86_64__) || defined(__i386__)
    const bool half = common == DType::Float16;
    if (!half && common != DType::BFloat16) return false;
    if (!narrow_compare_ready(half)) return false;
    if (!a.is_contiguous() || !result.is_contiguous()) return false;
    const bool bcast = (b.numel() == 1);
    if (!bcast && (!b.is_contiguous() || b.sizes() != result.sizes())) {
        return false;
    }
    if (a.sizes() != result.sizes()) return false;
    const int64_t n = result.numel();
    bool* out = result.data_ptr<bool>();
    if (half) {
        const uint16_t* pa = reinterpret_cast<const uint16_t*>(a.data_ptr<Half>());
        const uint16_t* pb = reinterpret_cast<const uint16_t*>(b.data_ptr<Half>());
        tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t lo, int64_t hi) {
            if (bcast) {
                narrow_compare_avx2<C, true, true>(pa, pb, out, lo, hi);
            } else {
                narrow_compare_avx2<C, true, false>(pa, pb, out, lo, hi);
            }
        });
    } else {
        const uint16_t* pa =
            reinterpret_cast<const uint16_t*>(a.data_ptr<BFloat16>());
        const uint16_t* pb =
            reinterpret_cast<const uint16_t*>(b.data_ptr<BFloat16>());
        tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t lo, int64_t hi) {
            if (bcast) {
                narrow_compare_avx2<C, false, true>(pa, pb, out, lo, hi);
            } else {
                narrow_compare_avx2<C, false, false>(pa, pb, out, lo, hi);
            }
        });
    }
    return true;
#else
    return false;
#endif
}

// Comparator ops accept quantized tensors by dequantizing both operands into
// the float domain; the comparison itself runs on the dequantized values.
template<bool kEquality, NarrowCmp C, typename Op>
Tensor comparison_kernel_impl(const Tensor& self, const Tensor& other, Op op) {
    if (isQuantizedType(self.dtype()) || isQuantizedType(other.dtype())) {
        Tensor self_f = isQuantizedType(self.dtype()) ? dequantize_self_cpu(self) : self;
        Tensor other_f = isQuantizedType(other.dtype()) ? dequantize_self_cpu(other) : other;
        return comparison_kernel_impl<kEquality, C>(self_f, other_f, op);
    }
    std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());

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
    // materialized expansion needed.  The narrow storages go through the masked
    // compare first, because their widening to float is not something the
    // per-element loop can run several lanes at a time.
    if (narrow_compare_path<C>(result, self_casted, other_casted, common_dtype)) {
        return result;
    }
    if constexpr (kEquality) {
        ti_apply_equality(result, self_casted, other_casted, common_dtype, op);
    } else {
        ti_apply_compare(result, self_casted, other_casted, common_dtype, op);
    }

    return result;
}

Tensor eq_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<true, NarrowCmp::Eq>(self, other, [](auto a, auto b) { return a == b; });
}

Tensor ne_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<true, NarrowCmp::Ne>(self, other, [](auto a, auto b) { return a != b; });
}

Tensor lt_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<false, NarrowCmp::Lt>(self, other, [](auto a, auto b) { return a < b; });
}

Tensor le_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<false, NarrowCmp::Le>(self, other, [](auto a, auto b) { return a <= b; });
}

Tensor gt_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<false, NarrowCmp::Gt>(self, other, [](auto a, auto b) { return a > b; });
}

Tensor ge_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl<false, NarrowCmp::Ge>(self, other, [](auto a, auto b) { return a >= b; });
}

// Scalar versions: promote the scalar with weak-scalar rules instead of
// casting it into self.dtype() (which truncated e.g. eq(2.5) on int tensors)
#define DEFINE_CMP_SCALAR_KERNEL(NAME) \
Tensor NAME##_scalar_kernel(const Tensor& self, const Scalar& other) { \
    Tensor self_f = isQuantizedType(self.dtype()) ? dequantize_self_cpu(self) : self; \
    DType common = result_type_with_scalar(self_f, other); \
    Tensor other_t = Tensor::full({}, other, common, self_f.device()); \
    return NAME##_tensor_kernel(self_f.to(common), other_t); \
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
    std::vector<int64_t> out_shape = broadcast_shapes(condition.sizes(), self.sizes(), other.sizes());
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

Tensor where_scalar_self_cpu(const Tensor& condition, const Scalar& self, const Tensor& other) {
    DType common_dtype = result_type(self, other.dtype());
    Tensor self_tensor = Tensor::full({}, self, common_dtype, other.device());
    return where_cpu(condition, self_tensor, other);
}

Tensor where_scalar_other_cpu(const Tensor& condition, const Tensor& self, const Scalar& other) {
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

Tensor where_scalar_scalar_cpu(const Tensor& condition, const Scalar& self, const Scalar& other) {
    DType common_dtype = where_scalar_dtype(self, other);
    Tensor self_tensor = Tensor::full({}, self, common_dtype, condition.device());
    Tensor other_tensor = Tensor::full({}, other, common_dtype, condition.device());
    return where_cpu(condition, self_tensor, other_tensor);
}

template <typename Op>
Tensor maximum_minimum_kernel_impl(const Tensor& self, const Tensor& other, Op op) {
    std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), other.sizes());
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
