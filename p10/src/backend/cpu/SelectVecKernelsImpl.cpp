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
