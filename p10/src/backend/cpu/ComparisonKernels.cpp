#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "OutWrite.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "TensorIteratorOps.h"
#include "Exception.h"
#include "Parallel.h"
#include "cpu/PairMath.h"
#include "cpu/VecUnary.h"
#include "cpu/SelectVecKernels.h"
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

// 256-bit twins.  The select runs on the sign bit of a mask that is widened
// from the byte-per-element condition, so eight lanes leave the condition
// array in one load and no lane is branched on.
__attribute__((target("avx2")))
inline void where_f32_avx2(const bool* condition, const float* self,
                           const float* other, float* result,
                           int64_t begin, int64_t end) {
    int64_t i = begin;
    const __m128i zero = _mm_setzero_si128();
    for (; i + 8 <= end; i += 8) {
        const __m128i c = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(condition + i));
        const __m256i m = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(c, zero));
        const __m256 a = _mm256_loadu_ps(self + i);
        const __m256 b = _mm256_loadu_ps(other + i);
        _mm256_storeu_ps(result + i, _mm256_blendv_ps(a, b, _mm256_castsi256_ps(m)));
    }
    for (; i < end; ++i) {
        result[i] = condition[i] ? self[i] : other[i];
    }
}

__attribute__((target("avx2")))
inline void where_f64_avx2(const bool* condition, const double* self,
                           const double* other, double* result,
                           int64_t begin, int64_t end) {
    int64_t i = begin;
    const __m128i zero = _mm_setzero_si128();
    for (; i + 4 <= end; i += 4) {
        const __m128i c = _mm_cvtsi32_si128(
            *reinterpret_cast<const int*>(condition + i));
        const __m256i m = _mm256_cvtepi8_epi64(_mm_cmpeq_epi8(c, zero));
        const __m256d a = _mm256_loadu_pd(self + i);
        const __m256d b = _mm256_loadu_pd(other + i);
        _mm256_storeu_pd(result + i, _mm256_blendv_pd(a, b, _mm256_castsi256_pd(m)));
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
Tensor comparison_kernel_impl(const Tensor& self, const Tensor& other, Op op);

// The pass once the destination, the operands and their common width are
// settled.  The narrow storages go through the masked compare first, because
// their widening to float is not something the per-element loop can run several
// lanes at a time; everything else is left to the iterator, which owns
// broadcast, reorder, coalesce, parallelism and the overlap judgment over the
// destination.
// The scalar comparison matching the lane predicate, for the sweep tails.
#if defined(__x86_64__) || defined(__i386__)
template <typename T, int PRED>
inline bool op_pred(T x, T y) {
    switch (PRED) {
        case _CMP_EQ_OQ: return x == y;
        case _CMP_NEQ_UQ: return x != y;
        case _CMP_LT_OQ: return x < y;
        case _CMP_LE_OQ: return x <= y;
        case _CMP_GT_OQ: return x > y;
        default: return x >= y;
    }
}

// One sweep of the f32/f64 comparison against a broadcast scalar: the same
// lane compare as the pair form, with the scalar held in a broadcast
// register.  The predicate instantiates per comparison kind.
template <int PRED, typename T>
__attribute__((target("avx2,fma"), noinline))
static void compare_scalar_sweep(const T* xp, T val, bool* rp, int64_t n) {
    if constexpr (std::is_same_v<T, float>) {
        const __m256 v = _mm256_set1_ps(val);
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 x = _mm256_loadu_ps(xp + i);
            const int mask = _mm256_movemask_ps(_mm256_cmp_ps(x, v, PRED));
            alignas(8) unsigned char tmp[8];
            for (int k = 0; k < 8; ++k) tmp[k] = (unsigned char)((mask >> k) & 1);
            std::memcpy(rp + i, tmp, 8);
        }
        for (; i < n; ++i) rp[i] = op_pred<T, PRED>(xp[i], val);
    } else {
        const __m256d v = _mm256_set1_pd(val);
        int64_t i = 0;
        for (; i + 4 <= n; i += 4) {
            const __m256d x = _mm256_loadu_pd(xp + i);
            const int mask = _mm256_movemask_pd(_mm256_cmp_pd(x, v, PRED));
            alignas(8) unsigned char tmp[4];
            for (int k = 0; k < 4; ++k) tmp[k] = (unsigned char)((mask >> k) & 1);
            std::memcpy(rp + i, tmp, 4);
        }
        for (; i < n; ++i) rp[i] = op_pred<T, PRED>(xp[i], val);
    }
}

template <int PRED, typename T>
__attribute__((target("avx2,fma"), noinline))
static void compare_sweep(const T* xp, const T* yp, bool* rp, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(xp + i);
        const __m256 y = _mm256_loadu_ps(yp + i);
        const int mask = _mm256_movemask_ps(_mm256_cmp_ps(x, y, PRED));
        alignas(8) unsigned char tmp[8];
        for (int k = 0; k < 8; ++k) tmp[k] = (unsigned char)((mask >> k) & 1);
        std::memcpy(rp + i, tmp, 8);
    }
    for (; i < n; ++i) rp[i] = op_pred<T, PRED>(xp[i], yp[i]);
}

template <int PRED, typename T>
__attribute__((target("avx2,fma"), noinline))
static void compare_sweep_pd(const T* xp, const T* yp, bool* rp, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d x = _mm256_loadu_pd(xp + i);
        const __m256d y = _mm256_loadu_pd(yp + i);
        const int mask = _mm256_movemask_pd(_mm256_cmp_pd(x, y, PRED));
        alignas(8) unsigned char tmp[4];
        for (int k = 0; k < 4; ++k) tmp[k] = (unsigned char)((mask >> k) & 1);
        std::memcpy(rp + i, tmp, 4);
    }
    for (; i < n; ++i) rp[i] = op_pred<T, PRED>(xp[i], yp[i]);
}
#endif  // x86

template <bool kEquality, NarrowCmp C, typename Op>
void comparison_pass(Tensor& result, const Tensor& self, const Tensor& other,
                     DType common_dtype, Op op) {
    Tensor self_casted = (self.dtype() == common_dtype) ? self : self.to(common_dtype);
    Tensor other_casted = (other.dtype() == common_dtype) ? other : other.to(common_dtype);
    // A scalar other broadcasts over the base: the lane compare runs with
    // the scalar in a broadcast register.  The predicate instantiates per
    // comparison kind.
#if defined(__x86_64__) || defined(__i386__)
    if ((common_dtype == DType::Float32 || common_dtype == DType::Float64) &&
        self_casted.is_contiguous() && other_casted.dim() == 0 &&
        self_casted.numel() == result.numel()) {
        const int64_t n = self_casted.numel();
        if (common_dtype == DType::Float32) {
            const float val = other_casted.data_ptr<float>()[0];
            switch (C) {
                case NarrowCmp::Eq: compare_scalar_sweep<_CMP_EQ_OQ>(self_casted.data_ptr<float>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Ne: compare_scalar_sweep<_CMP_NEQ_UQ>(self_casted.data_ptr<float>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Lt: compare_scalar_sweep<_CMP_LT_OQ>(self_casted.data_ptr<float>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Le: compare_scalar_sweep<_CMP_LE_OQ>(self_casted.data_ptr<float>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Gt: compare_scalar_sweep<_CMP_GT_OQ>(self_casted.data_ptr<float>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Ge: compare_scalar_sweep<_CMP_GE_OQ>(self_casted.data_ptr<float>(), val, result.data_ptr<bool>(), n); return;
            }
        } else {
            const double val = other_casted.data_ptr<double>()[0];
            switch (C) {
                case NarrowCmp::Eq: compare_scalar_sweep<_CMP_EQ_OQ>(self_casted.data_ptr<double>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Ne: compare_scalar_sweep<_CMP_NEQ_UQ>(self_casted.data_ptr<double>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Lt: compare_scalar_sweep<_CMP_LT_OQ>(self_casted.data_ptr<double>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Le: compare_scalar_sweep<_CMP_LE_OQ>(self_casted.data_ptr<double>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Gt: compare_scalar_sweep<_CMP_GT_OQ>(self_casted.data_ptr<double>(), val, result.data_ptr<bool>(), n); return;
                case NarrowCmp::Ge: compare_scalar_sweep<_CMP_GE_OQ>(self_casted.data_ptr<double>(), val, result.data_ptr<bool>(), n); return;
            }
        }
    }
    // Contiguous same-shape float pairs compare in one lane sweep; the
    // predicate instantiates per comparison kind.
    if ((common_dtype == DType::Float32 || common_dtype == DType::Float64) &&
        self_casted.is_contiguous() && other_casted.is_contiguous() &&
        self_casted.numel() == other_casted.numel() &&
        self_casted.numel() == result.numel()) {
        if (common_dtype == DType::Float32) {
            switch (C) {
                case NarrowCmp::Eq: compare_sweep<_CMP_EQ_OQ>(self_casted.data_ptr<float>(), other_casted.data_ptr<float>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Ne: compare_sweep<_CMP_NEQ_UQ>(self_casted.data_ptr<float>(), other_casted.data_ptr<float>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Lt: compare_sweep<_CMP_LT_OQ>(self_casted.data_ptr<float>(), other_casted.data_ptr<float>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Le: compare_sweep<_CMP_LE_OQ>(self_casted.data_ptr<float>(), other_casted.data_ptr<float>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Gt: compare_sweep<_CMP_GT_OQ>(self_casted.data_ptr<float>(), other_casted.data_ptr<float>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Ge: compare_sweep<_CMP_GE_OQ>(self_casted.data_ptr<float>(), other_casted.data_ptr<float>(), result.data_ptr<bool>(), self_casted.numel()); return;
            }
        } else {
            switch (C) {
                case NarrowCmp::Eq: compare_sweep_pd<_CMP_EQ_OQ>(self_casted.data_ptr<double>(), other_casted.data_ptr<double>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Ne: compare_sweep_pd<_CMP_NEQ_UQ>(self_casted.data_ptr<double>(), other_casted.data_ptr<double>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Lt: compare_sweep_pd<_CMP_LT_OQ>(self_casted.data_ptr<double>(), other_casted.data_ptr<double>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Le: compare_sweep_pd<_CMP_LE_OQ>(self_casted.data_ptr<double>(), other_casted.data_ptr<double>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Gt: compare_sweep_pd<_CMP_GT_OQ>(self_casted.data_ptr<double>(), other_casted.data_ptr<double>(), result.data_ptr<bool>(), self_casted.numel()); return;
                case NarrowCmp::Ge: compare_sweep_pd<_CMP_GE_OQ>(self_casted.data_ptr<double>(), other_casted.data_ptr<double>(), result.data_ptr<bool>(), self_casted.numel()); return;
            }
        }
    }
#endif  // x86
    if (narrow_compare_path<C>(result, self_casted, other_casted, common_dtype)) {
        return;
    }
    if constexpr (kEquality) {
        ti_apply_equality(result, self_casted, other_casted, common_dtype, op);
    } else {
        ti_apply_compare(result, self_casted, other_casted, common_dtype, op);
    }
}

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

    comparison_pass<kEquality, C>(result, self, other, common_dtype, op);
    return result;
}

// A destination the caller supplied takes the pass when it is the tensor the
// allocating form would have made; anything that needs a cast or a resize, a
// quantized pair and a complex pair the comparator refuses keep the value that
// form made.
template<bool kEquality, NarrowCmp C, typename Op>
Tensor& comparison_out_kernel(const Tensor& self, const Tensor& other,
                              Tensor& out, Op op) {
    const bool plain = !isQuantizedType(self.dtype()) && !isQuantizedType(other.dtype());
    const DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    const std::vector<int64_t> out_shape =
        broadcast_shapes(self.sizes(), other.sizes());
    // The pass walks the destination as a flat array, so it only serves a
    // packed destination; a strided one is written element by element by the
    // allocating form's copy instead.  An operand that overlaps the
    // destination is refused for the same reason: only a window read at the
    // width it is stored may be overwritten in place.
    if (plain && (kEquality || !isComplexType(common_dtype)) &&
        destination_takes_flat(out, self.device(), DType::Bool, out_shape,
                               self, other)) {
        comparison_pass<kEquality, C>(out, self, other, common_dtype, op);
        return out;
    }
    return write_out(out, comparison_kernel_impl<kEquality, C>(self, other, op));
}

#define DEFINE_CMP_TENSOR_OUT_KERNEL(NAME, EQUALITY, ORDER, PREDICATE)              \
    Tensor& NAME##_tensor_out_kernel(const Tensor& self, const Tensor& other,       \
                                     Tensor& out) {                                 \
        return comparison_out_kernel<EQUALITY, ORDER>(self, other, out, PREDICATE); \
    }

DEFINE_CMP_TENSOR_OUT_KERNEL(eq, true, NarrowCmp::Eq, [](auto a, auto b) { return a == b; })
DEFINE_CMP_TENSOR_OUT_KERNEL(ne, true, NarrowCmp::Ne, [](auto a, auto b) { return a != b; })
DEFINE_CMP_TENSOR_OUT_KERNEL(lt, false, NarrowCmp::Lt, [](auto a, auto b) { return a < b; })
DEFINE_CMP_TENSOR_OUT_KERNEL(le, false, NarrowCmp::Le, [](auto a, auto b) { return a <= b; })
DEFINE_CMP_TENSOR_OUT_KERNEL(gt, false, NarrowCmp::Gt, [](auto a, auto b) { return a > b; })
DEFINE_CMP_TENSOR_OUT_KERNEL(ge, false, NarrowCmp::Ge, [](auto a, auto b) { return a >= b; })
#undef DEFINE_CMP_TENSOR_OUT_KERNEL

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

// The same promotion for the spelling that answers into a caller's tensor: the
// scalar becomes a one-element operand and the comparison then judges the
// destination against the shape of the tensor it was given.
#define DEFINE_CMP_SCALAR_OUT_KERNEL(NAME)                                        \
    Tensor& NAME##_scalar_out_kernel(const Tensor& self, const Scalar& other,      \
                                     Tensor& out) {                                \
        Tensor self_f = isQuantizedType(self.dtype()) ? dequantize_self_cpu(self) : self; \
        DType common = result_type_with_scalar(self_f, other);                     \
        Tensor other_t = Tensor::full({}, other, common, self_f.device());         \
        return NAME##_tensor_out_kernel(self_f.to(common), other_t, out);          \
    }

DEFINE_CMP_SCALAR_OUT_KERNEL(eq)
DEFINE_CMP_SCALAR_OUT_KERNEL(ne)
DEFINE_CMP_SCALAR_OUT_KERNEL(lt)
DEFINE_CMP_SCALAR_OUT_KERNEL(le)
DEFINE_CMP_SCALAR_OUT_KERNEL(gt)
DEFINE_CMP_SCALAR_OUT_KERNEL(ge)
#undef DEFINE_CMP_SCALAR_OUT_KERNEL

// The long spellings answer from the same pass as their short counterparts: an
// out variant here only forwards, so whatever the destination takes in one
// spelling it takes in the other, and a quantized or complex operand meets the
// same handling either way.
#define DEFINE_CMP_OUT_ALIAS(NAME, TARGET)                                    \
    Tensor& NAME##_tensor_out_kernel(const Tensor& self, const Tensor& other, \
                                     Tensor& out) {                           \
        return TARGET##_tensor_out_kernel(self, other, out);                  \
    }                                                                         \
    Tensor& NAME##_scalar_out_kernel(const Tensor& self, const Scalar& other, \
                                     Tensor& out) {                           \
        return TARGET##_scalar_out_kernel(self, other, out);                  \
    }

DEFINE_CMP_OUT_ALIAS(not_equal, ne)
DEFINE_CMP_OUT_ALIAS(less, lt)
DEFINE_CMP_OUT_ALIAS(less_equal, le)
DEFINE_CMP_OUT_ALIAS(greater, gt)
DEFINE_CMP_OUT_ALIAS(greater_equal, ge)
#undef DEFINE_CMP_OUT_ALIAS

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
        if (common_dtype == DType::Float32 && vecunary::avx512_available()) {
            const float* a = self_casted.data_ptr<float>();
            const float* b = other_casted.data_ptr<float>();
            float* o = result.data_ptr<float>();
            tensorplay::parallel::parallel_for(0, n, 8192,
                [&](int64_t begin, int64_t end) {
                    where_f32_avx512(cond, a, b, o, begin, end);
                });
            return result;
        }
        if (common_dtype == DType::Float32 && vecunary::avx2_available()) {
            const float* a = self_casted.data_ptr<float>();
            const float* b = other_casted.data_ptr<float>();
            float* o = result.data_ptr<float>();
            tensorplay::parallel::parallel_for(0, n, 8192,
                [&](int64_t begin, int64_t end) {
                    where_f32_avx2(cond, a, b, o, begin, end);
                });
            return result;
        }
        if (common_dtype == DType::Float64 && vecunary::avx512_available()) {
            const double* a = self_casted.data_ptr<double>();
            const double* b = other_casted.data_ptr<double>();
            double* o = result.data_ptr<double>();
            tensorplay::parallel::parallel_for(0, n, 8192,
                [&](int64_t begin, int64_t end) {
                    where_f64_avx512(cond, a, b, o, begin, end);
                });
            return result;
        }
        if (common_dtype == DType::Float64 && vecunary::avx2_available()) {
            const double* a = self_casted.data_ptr<double>();
            const double* b = other_casted.data_ptr<double>();
            double* o = result.data_ptr<double>();
            tensorplay::parallel::parallel_for(0, n, 8192,
                [&](int64_t begin, int64_t end) {
                    where_f64_avx2(cond, a, b, o, begin, end);
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

// The chooser answers into the caller's tensor whenever that tensor is already
// the one the allocating form would have made; a destination that needs a cast
// or a resize, and a complex pair that has no answer at all, still go through
// that form.
template <typename Op>
Tensor& maximum_minimum_out_kernel(const Tensor& self, const Tensor& other,
                                   Tensor& out, Op op) {
    const DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    const std::vector<int64_t> out_shape =
        broadcast_shapes(self.sizes(), other.sizes());
    if (!isComplexType(common_dtype) &&
        destination_takes(out, self.device(), common_dtype, out_shape)) {
        Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
        Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
        ti_apply_binary(out, a, b, op);
        return out;
    }
    return write_out(out, maximum_minimum_kernel_impl(self, other, op));
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

#if defined(_M_X64)
// The same packed same-width pair the pair walk would take, handed to the tier
// chunk walk instead: on MSVC the vecbinary bodies do not exist, so the flat
// map (a NaN test per element) was the only answer before this path.
inline bool select_pair_takes(const Tensor& a, const Tensor& b) {
    if (a.dtype() != b.dtype()) return false;
    if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64) return false;
    return a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() &&
           a.numel() > 0;
}

inline Tensor select_pair_alloc(int op, const Tensor& a, const Tensor& b) {
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(a.shape()), a.dtype(), a.device());
    const int64_t n = result.numel();
    const int dtype = static_cast<int>(a.dtype());
    if (a.dtype() == DType::Float32) {
        const float* x = a.data_ptr<float>();
        const float* y = b.data_ptr<float>();
        float* d = result.data_ptr<float>();
        parallel::parallel_for_balanced(0, n, parallel::GRAIN_SIZE,
                                        [&](int64_t begin, int64_t end) {
            select_vec_stub(DeviceType::CPU, x + begin, y + begin, d + begin,
                            end - begin, op, 0.0, 0.0, dtype);
        });
    } else {
        const double* x = a.data_ptr<double>();
        const double* y = b.data_ptr<double>();
        double* d = result.data_ptr<double>();
        parallel::parallel_for_balanced(0, n, parallel::GRAIN_SIZE,
                                        [&](int64_t begin, int64_t end) {
            select_vec_stub(DeviceType::CPU, x + begin, y + begin, d + begin,
                            end - begin, op, 0.0, 0.0, dtype);
        });
    }
    return result;
}
#endif

// A floating pair of one width, in one shape and packed, is answered a vector
// at a time; everything else keeps the flat map.
Tensor maximum_cpu(const Tensor& self, const Tensor& other) {
#if defined(_M_X64)
    if (select_pair_takes(self, other)) return select_pair_alloc(1, self, other);
#endif
    if (pair_vector_takes(self, other)) {
        return pair_math_alloc<vecbinary::BOp::Maximum>(self, other);
    }
    return maximum_minimum_kernel_impl(self, other, MaximumOp());
}

Tensor minimum_cpu(const Tensor& self, const Tensor& other) {
#if defined(_M_X64)
    if (select_pair_takes(self, other)) return select_pair_alloc(2, self, other);
#endif
    if (pair_vector_takes(self, other)) {
        return pair_math_alloc<vecbinary::BOp::Minimum>(self, other);
    }
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

Tensor& maximum_out_cpu(const Tensor& self, const Tensor& other, Tensor& out) {
    if (pair_math_takes(self, other, out)) {
        return pair_math_out_kernel<vecbinary::BOp::Maximum>(
            self, other, out, MaximumOp(), "maximum");
    }
    return maximum_minimum_out_kernel(self, other, out, MaximumOp());
}

Tensor& minimum_out_cpu(const Tensor& self, const Tensor& other, Tensor& out) {
    if (pair_math_takes(self, other, out)) {
        return pair_math_out_kernel<vecbinary::BOp::Minimum>(
            self, other, out, MinimumOp(), "minimum");
    }
    return maximum_minimum_out_kernel(self, other, out, MinimumOp());
}

Tensor& fmax_out_cpu(const Tensor& self, const Tensor& other, Tensor& out) {
    return maximum_minimum_out_kernel(self, other, out, FmaxFminOp<true>());
}

Tensor& fmin_out_cpu(const Tensor& self, const Tensor& other, Tensor& out) {
    return maximum_minimum_out_kernel(self, other, out, FmaxFminOp<false>());
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
    m.impl("maximum.out", maximum_out_cpu);
    m.impl("minimum.out", minimum_out_cpu);
    m.impl("max.out", maximum_out_cpu);
    m.impl("min.out", minimum_out_cpu);
    m.impl("fmax.out", fmax_out_cpu);
    m.impl("fmin.out", fmin_out_cpu);
    m.impl("eq.Tensor_out", eq_tensor_out_kernel);
    m.impl("eq.Scalar_out", eq_scalar_out_kernel);
    m.impl("ne.Tensor_out", ne_tensor_out_kernel);
    m.impl("ne.Scalar_out", ne_scalar_out_kernel);
    m.impl("lt.Tensor_out", lt_tensor_out_kernel);
    m.impl("lt.Scalar_out", lt_scalar_out_kernel);
    m.impl("le.Tensor_out", le_tensor_out_kernel);
    m.impl("le.Scalar_out", le_scalar_out_kernel);
    m.impl("gt.Tensor_out", gt_tensor_out_kernel);
    m.impl("gt.Scalar_out", gt_scalar_out_kernel);
    m.impl("ge.Tensor_out", ge_tensor_out_kernel);
    m.impl("ge.Scalar_out", ge_scalar_out_kernel);
    m.impl("not_equal.Tensor_out", not_equal_tensor_out_kernel);
    m.impl("not_equal.Scalar_out", not_equal_scalar_out_kernel);
    m.impl("less.Tensor_out", less_tensor_out_kernel);
    m.impl("less.Scalar_out", less_scalar_out_kernel);
    m.impl("less_equal.Tensor_out", less_equal_tensor_out_kernel);
    m.impl("less_equal.Scalar_out", less_equal_scalar_out_kernel);
    m.impl("greater.Tensor_out", greater_tensor_out_kernel);
    m.impl("greater.Scalar_out", greater_scalar_out_kernel);
    m.impl("greater_equal.Tensor_out", greater_equal_tensor_out_kernel);
    m.impl("greater_equal.Scalar_out", greater_equal_scalar_out_kernel);
}

} // namespace cpu
} // namespace tensorplay
