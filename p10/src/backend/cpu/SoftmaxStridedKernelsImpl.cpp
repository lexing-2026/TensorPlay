// The softmax over a dimension that is not the last one.
//
// The reduced dimension is walked one row at a time and the values of the
// inner dimension are handled several at a time, the same shape the
// reference runtime walks.  The walk is written against the vector layer, so
// every CPU_CAPABILITY tier brings its own width and its own exponentials;
// DispatchStub picks the best registered tier at runtime.
// See p10/CMakeLists.txt Note [CPU_CAPABILITY namespace] and DispatchStub.h.

#include "cpu/SoftmaxStridedKernels.h"
#include "Parallel.h"
#include "cpu/vec/vec.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace tensorplay {
namespace cpu {
namespace {
using namespace vec;
// parallel_for / GRAIN_SIZE moved under tensorplay::parallel.
using namespace tensorplay::parallel;

// One unit of the walk: the values of two vectors of the inner dimension,
// reduced across the strided dimension.  A unit covers a cache line, so the
// rows of the reduced dimension are read once.
template <typename T, typename V>
void strided_unit(const T* base, T* obase, int64_t size, int64_t stride,
                  bool log_mode) {
    const int64_t lanes = V::size();
    V m0 = V::loadu(base);
    V m1 = V::loadu(base + lanes);
    for (int64_t k = 1; k < size; ++k) {
        const T* row = base + k * stride;
        m0 = vec::maximum(m0, V::loadu(row));
        m1 = vec::maximum(m1, V::loadu(row + lanes));
    }
    V s0(T(0));
    V s1(T(0));
    for (int64_t k = 0; k < size; ++k) {
        const T* row = base + k * stride;
        const V e0 = (V::loadu(row) - m0).exp();
        const V e1 = (V::loadu(row + lanes) - m1).exp();
        s0 = s0 + e0;
        s1 = s1 + e1;
        if (!log_mode) {
            e0.store(obase + k * stride);
            e1.store(obase + k * stride + lanes);
        }
    }
    if (log_mode) {
        // The logarithmic form answers from the row itself, so the
        // exponentials are not stored on the way.
        alignas(64) T b0[V::size()];
        alignas(64) T b1[V::size()];
        s0.store(b0);
        s1.store(b1);
        for (int64_t l = 0; l < V::size(); ++l) {
            b0[l] = std::log(b0[l]);
            b1[l] = std::log(b1[l]);
        }
        const V l0 = V::loadu(b0);
        const V l1 = V::loadu(b1);
        for (int64_t k = 0; k < size; ++k) {
            const T* row = base + k * stride;
            (V::loadu(row) - m0 - l0).store(obase + k * stride);
            (V::loadu(row + lanes) - m1 - l1).store(obase + k * stride + lanes);
        }
    } else {
        const V i0 = V(T(1)) / s0;
        const V i1 = V(T(1)) / s1;
        for (int64_t k = 0; k < size; ++k) {
            T* drow = obase + k * stride;
            (V::loadu(drow) * i0).store(drow);
            (V::loadu(drow + lanes) * i1).store(drow + lanes);
        }
    }
}

// The values that do not fill a unit are walked on their own.  Each of them
// is a column of the reduced dimension, reduced on its own.
template <typename T>
void strided_tail(const T* src, T* dst, int64_t size, int64_t stride,
                  int64_t before, int64_t offset, int64_t count,
                  bool log_mode) {
    parallel_for(0, before * count, 1, [&](int64_t ub, int64_t ue) {
        for (int64_t u = ub; u < ue; ++u) {
            const int64_t a = u / count;
            const int64_t j = u - a * count;
            const T* col = src + a * size * stride + offset + j;
            T* ocol = dst + a * size * stride + offset + j;
            T m = col[0];
            for (int64_t k = 1; k < size; ++k) m = std::max(m, col[k * stride]);
            T sum = T(0);
            for (int64_t k = 0; k < size; ++k) {
                const T e = std::exp(col[k * stride] - m);
                if (!log_mode) ocol[k * stride] = e;
                sum += e;
            }
            if (log_mode) {
                const T lse = std::log(sum);
                for (int64_t k = 0; k < size; ++k) {
                    ocol[k * stride] = (col[k * stride] - m) - lse;
                }
            } else {
                const T inv = T(1) / sum;
                for (int64_t k = 0; k < size; ++k) ocol[k * stride] *= inv;
            }
        }
    });
}

template <typename T, typename V>
void softmax_strided_walk(const T* src, T* dst, int64_t size, int64_t stride,
                          int64_t before, int64_t after, bool log_mode) {
    const int64_t lanes = 2 * V::size();
    if (after < lanes) {
        strided_tail(src, dst, size, stride, before, 0, after, log_mode);
        return;
    }
    const int64_t nblk = after / lanes;
    parallel_for(0, before * nblk, 1, [&](int64_t ub, int64_t ue) {
        for (int64_t u = ub; u < ue; ++u) {
            const int64_t a = u / nblk;
            const int64_t iv0 = (u - a * nblk) * lanes;
            strided_unit<T, V>(src + a * size * stride + iv0,
                               dst + a * size * stride + iv0,
                               size, stride, log_mode);
        }
    });
    const int64_t rest = after - nblk * lanes;
    if (rest > 0) {
        strided_tail(src, dst, size, stride, before, nblk * lanes, rest,
                     log_mode);
    }
}

void softmax_strided_f32_impl(const float* src, float* dst, int64_t size,
                              int64_t stride, int64_t before, int64_t after,
                              bool log_mode) {
    softmax_strided_walk<float, vec::Vectorized<float>>(src, dst, size, stride,
                                                        before, after, log_mode);
}

void softmax_strided_f64_impl(const double* src, double* dst, int64_t size,
                              int64_t stride, int64_t before, int64_t after,
                              bool log_mode) {
    softmax_strided_walk<double, vec::Vectorized<double>>(src, dst, size, stride,
                                                          before, after, log_mode);
}

} // anonymous namespace

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot.
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(softmax_strided_f32_stub, &softmax_strided_f32_impl);
REGISTER_DISPATCH(softmax_strided_f64_stub, &softmax_strided_f64_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(softmax_strided_f32_stub, &softmax_strided_f32_impl);
ALSO_REGISTER_AVX512_DISPATCH(softmax_strided_f64_stub, &softmax_strided_f64_impl);
#endif

} // namespace cpu
} // namespace tensorplay
