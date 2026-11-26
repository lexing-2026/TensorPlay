// Tier-compiled walk for the selection maps (f32/f64): clamp, maximum,
// minimum.
//
// Every tier compiles its own copy and registers the slot declared in
// cpu/SelectVecKernels.h; `vec::Vectorized<T>` resolves to that tier's width,
// and `vec::maximum` / `vec::minimum` / `vec::clamp` are the IEEE-201X
// variants (a NaN operand reaches the result), which is what the scalar
// reference in ComparisonKernels.cpp answers as well.
//
// The tail rounds each element the way those references do, so the last
// partial vector agrees with the flat loop it replaces.

#include "cpu/SelectVecKernels.h"

#include "DType.h"
#include "Exception.h"
#include "cpu/vec/vec.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace tensorplay {
namespace cpu {

namespace {

template <typename T>
void select_chunk(const T* a, const T* b, T* d, int64_t n, int op,
                  double p0, double p1) {
    using V = vec::Vectorized<T>;
    constexpr int64_t W = V::size();
    int64_t i = 0;

    if (op == 0) {
        // ATen's shape for the bounded pair: max with the lower bound first,
        // then min with the upper.  `max` and `min` answer their *second*
        // operand when either input is a NaN, and the value sits second in
        // both, so a NaN lane survives -- the same contract as vec::clamp
        // below, in two instructions instead of the compare/blend ladder.
#if defined(__AVX512F__) || defined(__AVX2__)
        if constexpr (std::is_same<T, float>::value) {
#if defined(__AVX512F__)
            const __m512 vlo = _mm512_set1_ps(static_cast<float>(p0));
            const __m512 vhi = _mm512_set1_ps(static_cast<float>(p1));
            constexpr int64_t K = 16;
            for (; i + K <= n; i += K) {
                const __m512 v = _mm512_loadu_ps(a + i);
                _mm512_storeu_ps(d + i,
                                 _mm512_min_ps(vhi, _mm512_max_ps(vlo, v)));
            }
#else
            const __m256 vlo = _mm256_set1_ps(static_cast<float>(p0));
            const __m256 vhi = _mm256_set1_ps(static_cast<float>(p1));
            constexpr int64_t K = 8;
            for (; i + K <= n; i += K) {
                const __m256 v = _mm256_loadu_ps(a + i);
                _mm256_storeu_ps(d + i,
                                 _mm256_min_ps(vhi, _mm256_max_ps(vlo, v)));
            }
#endif
            for (; i < n; ++i) {
                float v = a[i];
                if (v < static_cast<float>(p0)) v = static_cast<float>(p0);
                if (v > static_cast<float>(p1)) v = static_cast<float>(p1);
                d[i] = v;
            }
            return;
        }
        if constexpr (std::is_same<T, double>::value) {
#if defined(__AVX512F__)
            const __m512d vlo = _mm512_set1_pd(p0);
            const __m512d vhi = _mm512_set1_pd(p1);
            constexpr int64_t K = 8;
            for (; i + K <= n; i += K) {
                const __m512d v = _mm512_loadu_pd(a + i);
                _mm512_storeu_pd(d + i,
                                 _mm512_min_pd(vhi, _mm512_max_pd(vlo, v)));
            }
#else
            const __m256d vlo = _mm256_set1_pd(p0);
            const __m256d vhi = _mm256_set1_pd(p1);
            constexpr int64_t K = 4;
            for (; i + K <= n; i += K) {
                const __m256d v = _mm256_loadu_pd(a + i);
                _mm256_storeu_pd(d + i,
                                 _mm256_min_pd(vhi, _mm256_max_pd(vlo, v)));
            }
#endif
            for (; i < n; ++i) {
                double v = a[i];
                if (v < p0) v = p0;
                if (v > p1) v = p1;
                d[i] = v;
            }
            return;
        }
#endif
        const V lo(static_cast<T>(p0));
        const V hi(static_cast<T>(p1));
        for (; i + W <= n; i += W) {
            vec::clamp(V::loadu(a + i), lo, hi).store(d + i);
        }
        for (; i < n; ++i) {
            T v = a[i];
            if (v < static_cast<T>(p0)) v = static_cast<T>(p0);
            if (v > static_cast<T>(p1)) v = static_cast<T>(p1);
            d[i] = v;
        }
        return;
    }

    const T nan = std::numeric_limits<T>::quiet_NaN();
    if (op == 1) {
        for (; i + W <= n; i += W) {
            vec::maximum(V::loadu(a + i), V::loadu(b + i)).store(d + i);
        }
        for (; i < n; ++i) {
            const T x = a[i];
            const T y = b[i];
            d[i] = (x != x || y != y) ? nan : (x < y ? y : x);
        }
        return;
    }

    for (; i + W <= n; i += W) {
        vec::minimum(V::loadu(a + i), V::loadu(b + i)).store(d + i);
    }
    for (; i < n; ++i) {
        const T x = a[i];
        const T y = b[i];
        d[i] = (x != x || y != y) ? nan : (x < y ? x : y);
    }
}

void select_vec_impl(const void* a, const void* b, void* dst, int64_t n,
                     int op, double p0, double p1, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            select_chunk(static_cast<const float*>(a),
                         static_cast<const float*>(b),
                         static_cast<float*>(dst), n, op, p0, p1);
            return;
        case DType::Float64:
            select_chunk(static_cast<const double*>(a),
                         static_cast<const double*>(b),
                         static_cast<double*>(dst), n, op, p0, p1);
            return;
        default:
            TP_THROW(TypeError, "select_vec: unsupported dtype");
    }
}

} // namespace

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(select_vec_stub, &select_vec_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(select_vec_stub, &select_vec_impl);
#endif

} // namespace cpu
} // namespace tensorplay
