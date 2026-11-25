// Tier-compiled walks for the contiguous wrapped-scalar arithmetic forms.
//
// Each CPU capability tier compiles its own copy of this file (see
// TP_CPU_KERNEL_SRCS in p10/CMakeLists.txt) and registers the slot declared in
// cpu/ScalarArithVecKernels.h; DispatchStub picks the widest copy at run time.
// The DEFAULT copy is the emulated vec::Vectorized loop, so a host without
// AVX2 keeps a correct, if scalar, walk.
//
// The per-lane arithmetic stays in the operand width.  add.Scalar folds
// alpha * other in the result dtype before it gets here, and sub.Scalar hands
// the negated product to the add walk, so both modes round exactly like the
// generic recursive functor in ArithmeticKernels.cpp.

#include "cpu/ScalarArithVecKernels.h"
#include "cpu/vec/vec.h"

#include "DType.h"
#include "Exception.h"

#include <cstdint>

namespace tensorplay {
namespace cpu {

namespace {

enum : int { kAdd = 0, kMul = 1, kDiv = 2 };

template <typename T>
void scalar_vec_typed(const T* a, T* y, int64_t n, double scalar, int code) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec vs(static_cast<T>(scalar));
    int64_t i = 0;
    for (; i + W <= n; i += W) {
        const Vec x = Vec::loadu(a + i);
        const Vec r = code == kAdd ? x + vs : (code == kMul ? x * vs : x / vs);
        r.store(y + i);
    }
    const T s = static_cast<T>(scalar);
    for (; i < n; ++i) {
        y[i] = code == kAdd ? static_cast<T>(a[i] + s)
             : code == kMul ? static_cast<T>(a[i] * s)
                            : static_cast<T>(a[i] / s);
    }
}

void scalar_vec_impl(const void* a, void* y, int64_t n, double scalar, int code,
                     int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            scalar_vec_typed(static_cast<const double*>(a),
                             static_cast<double*>(y), n, scalar, code);
            return;
        case DType::Float32:
            scalar_vec_typed(static_cast<const float*>(a),
                             static_cast<float*>(y), n, scalar, code);
            return;
        default:
            TP_THROW(TypeError, "scalar_vec: unsupported dtype");
    }
}

} // namespace

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(scalar_vec_stub, &scalar_vec_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(scalar_vec_stub, &scalar_vec_impl);
#endif

} // namespace cpu
} // namespace tensorplay
