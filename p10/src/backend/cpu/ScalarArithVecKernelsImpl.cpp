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
//
// The walk for a code is selected once, outside the loop, through a template
// parameter.  A runtime `code == ... ? a / b : a * b` inside the loop lets
// MSVC evaluate the division for every vector of the multiply walk as well,
// which showed up as mul.Scalar running at half the rate of mul.Tensor over
// 1M elements (vdivps is 8 cycles a vector on Zen 4 against 0.5 for vmulps).

#include "cpu/ScalarArithVecKernels.h"
#include "cpu/vec/vec.h"

#include "DType.h"
#include "Exception.h"

#include <cstdint>

namespace tensorplay {
namespace cpu {

namespace {

enum : int { kAdd = 0, kMul = 1, kDiv = 2 };
// Codes of the tensor OP tensor walk (binary_vec_impl); the scalar walk keeps
// its own numbering above because the wrapped-scalar kernels pass it through.
enum : int { bAdd = 0, bSub = 1, bMul = 2, bDiv = 3 };

// Scalar arithmetic selected at compile time: ScalarOp<kAdd>{}(x, s) is x + s.
template <int Code>
struct ScalarOp {
    template <typename T>
    T operator()(const T& x, const T& s) const {
        if constexpr (Code == kAdd) return x + s;
        else if constexpr (Code == kMul) return x * s;
        else return x / s;
    }
};

template <typename T, int Code>
void scalar_vec_loop(const T* a, T* y, int64_t n, double scalar) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    ScalarOp<Code> op;
    const Vec vs(static_cast<T>(scalar));
    int64_t i = 0;
    for (; i + W <= n; i += W) {
        op(Vec::loadu(a + i), vs).store(y + i);
    }
    const T s = static_cast<T>(scalar);
    for (; i < n; ++i) {
        y[i] = static_cast<T>(op(a[i], s));
    }
}

template <typename T>
void scalar_vec_typed(const T* a, T* y, int64_t n, double scalar, int code) {
    switch (code) {
        case kAdd: scalar_vec_loop<T, kAdd>(a, y, n, scalar); return;
        case kMul: scalar_vec_loop<T, kMul>(a, y, n, scalar); return;
        default:   scalar_vec_loop<T, kDiv>(a, y, n, scalar); return;
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

// Same-shape tensor OP tensor.  Both operands are loaded before the store of a
// vector, so an output that aliases either input (the in-place forms) still
// reads the values that were there before the pass reached that vector.
template <int Code>
struct BinaryOp {
    template <typename T>
    T operator()(const T& x, const T& z) const {
        if constexpr (Code == bAdd) return x + z;
        else if constexpr (Code == bSub) return x - z;
        else if constexpr (Code == bMul) return x * z;
        else return x / z;
    }
};

template <typename T, int Code>
void binary_vec_loop(const T* a, const T* b, T* y, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    BinaryOp<Code> op;
    int64_t i = 0;
    for (; i + W <= n; i += W) {
        const Vec x = Vec::loadu(a + i);
        const Vec z = Vec::loadu(b + i);
        op(x, z).store(y + i);
    }
    for (; i < n; ++i) {
        y[i] = static_cast<T>(op(a[i], b[i]));
    }
}

template <typename T>
void binary_vec_typed(const T* a, const T* b, T* y, int64_t n, int code) {
    switch (code) {
        case bAdd: binary_vec_loop<T, bAdd>(a, b, y, n); return;
        case bSub: binary_vec_loop<T, bSub>(a, b, y, n); return;
        case bMul: binary_vec_loop<T, bMul>(a, b, y, n); return;
        default:   binary_vec_loop<T, bDiv>(a, b, y, n); return;
    }
}

void binary_vec_impl(const void* a, const void* b, void* y, int64_t n, int code,
                     int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            binary_vec_typed(static_cast<const double*>(a),
                             static_cast<const double*>(b),
                             static_cast<double*>(y), n, code);
            return;
        case DType::Float32:
            binary_vec_typed(static_cast<const float*>(a),
                             static_cast<const float*>(b),
                             static_cast<float*>(y), n, code);
            return;
        default:
            TP_THROW(TypeError, "binary_vec: unsupported dtype");
    }
}

} // namespace

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(scalar_vec_stub, &scalar_vec_impl);
REGISTER_DISPATCH(binary_vec_stub, &binary_vec_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(scalar_vec_stub, &scalar_vec_impl);
ALSO_REGISTER_AVX512_DISPATCH(binary_vec_stub, &binary_vec_impl);
#endif

} // namespace cpu
} // namespace tensorplay
