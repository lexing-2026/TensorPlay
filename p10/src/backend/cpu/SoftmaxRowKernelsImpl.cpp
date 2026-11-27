// Tier-compiled last-dimension softmax row walk (f32/f64).
//
// Each CPU capability tier compiles its own copy; vec::maximum and the
// exponential member resolve to that tier's width.  The DEFAULT copy is the
// emulated width, still correct on a host without AVX2.
//
// The row is read twice in log mode (max, then the shifted values again after
// the log of the sum), which is the same traffic the scalar row makes.
//
// f32 on AVX-512 / AVX-2 takes the explicit intrinsic walk below.  Three
// things in it are not expressible through the portable vector type and are
// what the reference runtime's softmax does instead of the generic walk:
//
//   * the exponential is a range-reduced Horner form (2^f with f = x - n*ln2
//     split into an exact 12-bit hi part and a lo part, degree-5 polynomial
//     for 2^f, 2^n built from the exponent field), about half the arithmetic
//     of the u10 library exponential and accurate to one ulp, which is all a
//     softmax needs since the argument is x - max <= 0;
//   * the scale pass (and the log pass in log mode) runs over vectors, where
//     the previous walk multiplied and logged element by element;
//   * the lane fold of the max and the sum goes through the intrinsic
//     reductions instead of a store-and-scalar loop.
//
// NaN handling follows the generic walk: a NaN input makes the row max a NaN
// and therefore every output of the row, and -inf inputs give exact zeros
// like the library exponential does.

#include "cpu/SoftmaxRowKernels.h"

#include "DType.h"
#include "Exception.h"
#include "Parallel.h"
#include "cpu/vec/vec.h"
#include "cpu/vec/SleefShims.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tensorplay {
namespace cpu {

namespace {

template <typename T>
void softmax_rows_generic(const T* in, T* out, int64_t rows, int64_t size,
                          bool log_mode) {
    using V = vec::Vectorized<T>;
    constexpr int64_t W = V::size();

    parallel::parallel_for(0, rows, 1, [&](int64_t rb, int64_t re) {
        // One scratch per worker: the lane fold of a row is written here and
        // read back as scalars, so it must not be shared across the pool.
        alignas(64) T buf[V::size()];
        for (int64_t r = rb; r < re; ++r) {
            const T* row = in + r * size;
            T* orow = out + r * size;
            int64_t j = 0;
            T mx = row[0];
            if (size >= W) {
                V m = V::loadu(row);
                for (j = W; j + W <= size; j += W) {
                    m = vec::maximum(m, V::loadu(row + j));
                }
                m.store(buf);
                mx = buf[0];
                for (int64_t l = 1; l < W; ++l) mx = std::max(mx, buf[l]);
            }
            for (; j < size; ++j) mx = std::max(mx, row[j]);

            const V vmx(mx);
            V sv(T(0));
            j = 0;
            if (size >= W) {
                for (; j + W <= size; j += W) {
                    const V e = (V::loadu(row + j) - vmx).exp();
                    if (!log_mode) e.store(orow + j);
                    sv = sv + e;
                }
                sv.store(buf);
            }
            T sum = size >= W ? buf[0] : T(0);
            if (size >= W) {
                for (int64_t l = 1; l < W; ++l) sum += buf[l];
            }
            for (; j < size; ++j) {
                const T e = std::exp(row[j] - mx);
                if (!log_mode) orow[j] = e;
                sum += e;
            }

            if (log_mode) {
                const T lse = std::log(sum);
                const V vlse(lse);
                for (int64_t k = 0; k + W <= size; k += W) {
                    ((V::loadu(row + k) - vmx) - vlse).store(orow + k);
                }
                for (int64_t k = size - size % W; k < size; ++k) {
                    orow[k] = (row[k] - mx) - lse;
                }
            } else {
                const V vinv(T(1) / sum);
                for (int64_t k = 0; k + W <= size; k += W) {
                    (V::loadu(orow + k) * vinv).store(orow + k);
                }
                const T inv = T(1) / sum;
                for (int64_t k = size - size % W; k < size; ++k) orow[k] *= inv;
            }
        }
    });
}

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)

// The primitives the f32 walk is written against.  AVX512 / AVX2 specialise
// the types; the mask is whatever the compare returns in that ISA.
#if defined(CPU_CAPABILITY_AVX512)
struct Tag512 {};
using RowTag = Tag512;
inline __m512 rv_set1(float v) { return _mm512_set1_ps(v); }
inline __m512 rv_load(const float* p) { return _mm512_loadu_ps(p); }
inline void rv_store(float* p, __m512 v) { _mm512_storeu_ps(p, v); }
inline __m512 rv_max(__m512 a, __m512 b) { return _mm512_max_ps(a, b); }
inline __m512 rv_add(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }
inline __m512 rv_sub(__m512 a, __m512 b) { return _mm512_sub_ps(a, b); }
inline __m512 rv_mul(__m512 a, __m512 b) { return _mm512_mul_ps(a, b); }
inline __m512 rv_fmadd(__m512 a, __m512 b, __m512 c) { return _mm512_fmadd_ps(a, b, c); }
inline __m512 rv_log(__m512 a) { return tensorplay::tpsleef::log(a); }
inline __mmask16 rv_unord(__m512 a) { return _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q); }
inline __mmask16 rv_lt(__m512 a, float lim) { return _mm512_cmp_ps_mask(a, rv_set1(lim), _CMP_LT_OQ); }
inline __mmask16 rv_or(__mmask16 a, __mmask16 b) { return _kor_mask16(a, b); }
inline bool rv_any(__mmask16 m) { return m != 0; }
inline __m512 rv_zero_where(__mmask16 m, __m512 v) { return _mm512_mask_mov_ps(v, m, _mm512_setzero_ps()); }
inline float rv_max_lane(__m512 v) { return _mm512_reduce_max_ps(v); }
inline float rv_sum_lane(__m512 v) { return _mm512_reduce_add_ps(v); }
inline __m512i rv_rint(__m512 x) { return _mm512_cvtps_epi32(_mm512_mul_ps(x, rv_set1(1.4426950408889634f))); }
inline __m512 rv_int_to_float(__m512i n) { return _mm512_cvtepi32_ps(n); }
inline __m512 rv_scale_pow2(__m512 p, __m512i n) {
    const __m512i e = _mm512_slli_epi32(_mm512_add_epi32(n, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(e));
}
constexpr int kRvLanes = 16;
#else
struct Tag256 {};
using RowTag = Tag256;
inline __m256 rv_set1(float v) { return _mm256_set1_ps(v); }
inline __m256 rv_load(const float* p) { return _mm256_loadu_ps(p); }
inline void rv_store(float* p, __m256 v) { _mm256_storeu_ps(p, v); }
inline __m256 rv_max(__m256 a, __m256 b) { return _mm256_max_ps(a, b); }
inline __m256 rv_add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
inline __m256 rv_sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
inline __m256 rv_mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
inline __m256 rv_fmadd(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }
inline __m256 rv_log(__m256 a) { return tensorplay::tpsleef::log(a); }
inline __m256 rv_unord(__m256 a) { return _mm256_cmp_ps(a, a, _CMP_UNORD_Q); }
inline __m256 rv_lt(__m256 a, float lim) { return _mm256_cmp_ps(a, rv_set1(lim), _CMP_LT_OQ); }
inline __m256 rv_or(__m256 a, __m256 b) { return _mm256_or_ps(a, b); }
inline bool rv_any(__m256 m) { return _mm256_movemask_ps(m) != 0; }
inline __m256 rv_zero_where(__m256 m, __m256 v) { return _mm256_blendv_ps(v, _mm256_setzero_ps(), m); }
inline float rv_max_lane(__m256 v) {
    __m128 l = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    l = _mm_max_ps(l, _mm_movehl_ps(l, l));
    l = _mm_max_ps(l, _mm_movehdup_ps(l));
    return _mm_cvtss_f32(l);
}
inline float rv_sum_lane(__m256 v) {
    const __m128 l = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 h = _mm_add_ps(l, _mm_movehl_ps(l, l));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)));
}
inline __m256i rv_rint(__m256 x) { return _mm256_cvtps_epi32(_mm256_mul_ps(x, rv_set1(1.4426950408889634f))); }
inline __m256 rv_int_to_float(__m256i n) { return _mm256_cvtepi32_ps(n); }
inline __m256 rv_scale_pow2(__m256 p, __m256i n) {
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}
constexpr int kRvLanes = 8;
#endif

using RVf = decltype(rv_set1(0.0f));

// exp(x) for x <= 0: n = rint(x / ln2), f = x - n*ln2 in [-ln2/2, ln2/2]
// (hi part exact in 12 bits so the first fma is exact), e^f by a degree-6
// polynomial (2.6e-9 relative, fitted on that interval), 2^n from the exponent
// field.  Arguments below -87.3 are clamped so n never leaves the normal
// range, and anything below -104 (where the float exponential is 0) is zeroed
// after the fact.  The clamp is written with the constant on the left because
// a max whose second operand is NaN returns that NaN, which keeps a NaN input
// a NaN output the way the library exponential does.
inline RVf rv_exp_fast(RVf x) {
    const RVf neg_ln2_hi = rv_set1(-0.693359375f);
    const RVf neg_ln2_lo = rv_set1(2.1219444005469058277e-4f);
    const auto underflow = rv_lt(x, -103.97f);
    x = rv_max(rv_set1(-87.3f), x);
    const auto n = rv_rint(x);
    const auto nf = rv_int_to_float(n);
    RVf f = rv_fmadd(nf, neg_ln2_hi, x);
    f = rv_fmadd(nf, neg_ln2_lo, f);
    RVf p = rv_set1(0.00139485808f);
    p = rv_fmadd(p, f, rv_set1(0.00837512889f));
    p = rv_fmadd(p, f, rv_set1(0.0416662183f));
    p = rv_fmadd(p, f, rv_set1(0.166664155f));
    p = rv_fmadd(p, f, rv_set1(0.500000011f));
    p = rv_fmadd(p, f, rv_set1(1.0f));
    p = rv_fmadd(p, f, rv_set1(1.0f));
    const RVf r = rv_scale_pow2(p, n);
    return rv_zero_where(underflow, r);
}

void softmax_rows_f32(const float* in, float* out, int64_t rows, int64_t size,
                      bool log_mode) {
    constexpr int64_t W = kRvLanes;
    parallel::parallel_for(0, rows, 1, [&](int64_t rb, int64_t re) {
        for (int64_t r = rb; r < re; ++r) {
            const float* row = in + r * size;
            float* orow = out + r * size;
            int64_t j = 0;
            float mx = row[0];
            if (size >= W) {
                auto m = rv_load(row);
                auto nan = rv_unord(m);
                for (j = W; j + W <= size; j += W) {
                    const auto x = rv_load(row + j);
                    nan = rv_or(nan, rv_unord(x));
                    m = rv_max(m, x);
                }
                mx = rv_max_lane(m);
                if (rv_any(nan)) mx = std::numeric_limits<float>::quiet_NaN();
            }
            for (; j < size; ++j) {
                const float v = row[j];
                if (std::isnan(v) || std::isnan(mx)) {
                    mx = std::numeric_limits<float>::quiet_NaN();
                } else if (v > mx) {
                    mx = v;
                }
            }

            const auto vmx = rv_set1(mx);
            auto sv = rv_set1(0.0f);
            j = 0;
            if (size >= W) {
                for (; j + W <= size; j += W) {
                    const auto e = rv_exp_fast(rv_sub(rv_load(row + j), vmx));
                    if (!log_mode) rv_store(orow + j, e);
                    sv = rv_add(sv, e);
                }
            }
            float sum = size >= W ? rv_sum_lane(sv) : 0.0f;
            for (; j < size; ++j) {
                const float e = std::exp(row[j] - mx);
                if (!log_mode) orow[j] = e;
                sum += e;
            }

            if (log_mode) {
                const float lse = std::log(sum);
                const auto vlse = rv_set1(lse);
                for (int64_t k = 0; k + W <= size; k += W) {
                    rv_store(orow + k, rv_sub(rv_sub(rv_load(row + k), vmx), vlse));
                }
                for (int64_t k = size - size % W; k < size; ++k) {
                    orow[k] = (row[k] - mx) - lse;
                }
            } else {
                const auto vinv = rv_set1(1.0f / sum);
                for (int64_t k = 0; k + W <= size; k += W) {
                    rv_store(orow + k, rv_mul(rv_load(orow + k), vinv));
                }
                const float inv = 1.0f / sum;
                for (int64_t k = size - size % W; k < size; ++k) orow[k] *= inv;
            }
        }
    });
}
#endif  // AVX-512 / AVX-2

void softmax_row_impl(const void* in, void* out, int64_t rows,
                      int64_t size, bool log_mode, int dtype) {
    if (rows <= 0 || size <= 0) {
        return;
    }
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)
            softmax_rows_f32(static_cast<const float*>(in),
                             static_cast<float*>(out), rows, size, log_mode);
#else
            softmax_rows_generic<float>(static_cast<const float*>(in),
                                        static_cast<float*>(out), rows, size,
                                        log_mode);
#endif
            return;
        case DType::Float64:
            softmax_rows_generic<double>(static_cast<const double*>(in),
                                         static_cast<double*>(out), rows, size,
                                         log_mode);
            return;
        default:
            TP_THROW(TypeError, "softmax_row: unsupported dtype");
    }
}

} // namespace

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(softmax_row_stub, &softmax_row_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(softmax_row_stub, &softmax_row_impl);
#endif

} // namespace cpu
} // namespace tensorplay
