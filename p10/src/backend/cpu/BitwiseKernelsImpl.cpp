// Vectorized bitwise cores, compiled once per CPU capability tier (see
// TP_CPU_KERNEL_SRCS in p10/CMakeLists.txt).  Each copy lands in the
// CPU_CAPABILITY inline namespace and registers its own slot on the stubs
// declared in cpu/BitwiseKernels.h; DispatchStub picks the best tier at
// runtime.  The bitwise entry points, broadcasting, dtype promotion and the
// boolean special cases stay in the base tier (BitwiseKernels.cpp).

#include "cpu/BitwiseKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"

#include "cpu/vec/vec.h"

#include <cstdint>
#include <type_traits>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

// Shift semantics: shift amounts outside [0, bit width) produce a defined
// fallback value instead of hardware-dependent results.  Left shifts of an
// invalid amount yield zero; right shifts yield zero for unsigned operands
// and the sign extension (all ones for negative values) for signed ones.
template <typename T, bool kLeft>
inline T bitwise_shift_value(T value, T shift) {
    using S = typename std::make_signed<T>::type;
    using U = typename std::make_unsigned<T>::type;
    constexpr U kBits = static_cast<U>(sizeof(T) * 8);
    const bool invalid = static_cast<S>(shift) < 0 || static_cast<U>(shift) >= kBits;
    if constexpr (kLeft) {
        if (invalid) return T(0);
        return static_cast<T>(static_cast<U>(value) << static_cast<U>(shift));
    }
    if (invalid) {
        if constexpr (std::is_signed_v<T>) return value < 0 ? T(-1) : T(0);
        return T(0);
    }
    return static_cast<T>(value >> static_cast<U>(shift));
}

template <typename T, typename ScalarOp, typename VectorOp>
inline void parallel_binary_vectorized(
        const T* lhs, const T* rhs, T* output, int64_t n,
        ScalarOp scalar_op, VectorOp vector_op) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t width = Vec::size();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t index = begin;
        const int64_t vector_end = begin + ((end - begin) / width) * width;
        for (; index < vector_end; index += width) {
            vector_op(Vec::loadu(lhs + index), Vec::loadu(rhs + index))
                .store(output + index);
        }
        for (; index < end; ++index) {
            output[index] = scalar_op(lhs[index], rhs[index]);
        }
    });
}

template <typename T, typename ScalarOp, typename VectorOp>
inline void parallel_scalar_vectorized(
        const T* input, T scalar, T* output, int64_t n,
        ScalarOp scalar_op, VectorOp vector_op) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t width = Vec::size();
    const Vec vector_scalar(scalar);
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t index = begin;
        const int64_t vector_end = begin + ((end - begin) / width) * width;
        for (; index < vector_end; index += width) {
            vector_op(Vec::loadu(input + index), vector_scalar)
                .store(output + index);
        }
        for (; index < end; ++index) {
            output[index] = scalar_op(input[index], scalar);
        }
    });
}

template <typename T, typename ScalarOp, typename VectorOp>
inline void parallel_unary_vectorized(
        const T* input, T* output, int64_t n,
        ScalarOp scalar_op, VectorOp vector_op) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t width = Vec::size();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t index = begin;
        const int64_t vector_end = begin + ((end - begin) / width) * width;
        for (; index < vector_end; index += width) {
            vector_op(Vec::loadu(input + index)).store(output + index);
        }
        for (; index < end; ++index) {
            output[index] = scalar_op(input[index]);
        }
    });
}

// The scalar shift path clamps invalid amounts through bitwise_shift_value;
// the vector path applies the raw lane-wise shift, so both paths keep their
// pre-existing semantics for out-of-range shift amounts.
template <typename T>
void bitwise_binary_typed(const T* lhs, const T* rhs, T* out, int64_t n,
                          int op) {
    using Vec = tensorplay::vec::Vectorized<T>;
    switch (op) {
        case static_cast<int>(BitwiseOp::kAnd):
            parallel_binary_vectorized<T>(lhs, rhs, out, n,
                [](T a, T b) { return static_cast<T>(a & b); },
                [](Vec a, Vec b) { return a & b; });
            break;
        case static_cast<int>(BitwiseOp::kOr):
            parallel_binary_vectorized<T>(lhs, rhs, out, n,
                [](T a, T b) { return static_cast<T>(a | b); },
                [](Vec a, Vec b) { return a | b; });
            break;
        case static_cast<int>(BitwiseOp::kXor):
            parallel_binary_vectorized<T>(lhs, rhs, out, n,
                [](T a, T b) { return static_cast<T>(a ^ b); },
                [](Vec a, Vec b) { return a ^ b; });
            break;
        case static_cast<int>(BitwiseOp::kLshift):
            parallel_binary_vectorized<T>(lhs, rhs, out, n,
                [](T v, T sh) { return bitwise_shift_value<T, true>(v, sh); },
                [](Vec v, Vec sh) { return v << sh; });
            break;
        case static_cast<int>(BitwiseOp::kRshift):
            parallel_binary_vectorized<T>(lhs, rhs, out, n,
                [](T v, T sh) { return bitwise_shift_value<T, false>(v, sh); },
                [](Vec v, Vec sh) { return v >> sh; });
            break;
        default:
            TP_THROW(TypeError, "bitwise: unsupported operation");
    }
}

template <typename T>
void bitwise_scalar_typed(const T* lhs, int64_t scalar, T* out, int64_t n,
                          int op) {
    using Vec = tensorplay::vec::Vectorized<T>;
    const T s = static_cast<T>(scalar);
    switch (op) {
        case static_cast<int>(BitwiseOp::kAnd):
            parallel_scalar_vectorized<T>(lhs, s, out, n,
                [](T a, T b) { return static_cast<T>(a & b); },
                [](Vec a, Vec b) { return a & b; });
            break;
        case static_cast<int>(BitwiseOp::kOr):
            parallel_scalar_vectorized<T>(lhs, s, out, n,
                [](T a, T b) { return static_cast<T>(a | b); },
                [](Vec a, Vec b) { return a | b; });
            break;
        case static_cast<int>(BitwiseOp::kXor):
            parallel_scalar_vectorized<T>(lhs, s, out, n,
                [](T a, T b) { return static_cast<T>(a ^ b); },
                [](Vec a, Vec b) { return a ^ b; });
            break;
        case static_cast<int>(BitwiseOp::kLshift):
            parallel_scalar_vectorized<T>(lhs, s, out, n,
                [](T v, T sh) { return bitwise_shift_value<T, true>(v, sh); },
                [](Vec v, Vec sh) { return v << sh; });
            break;
        case static_cast<int>(BitwiseOp::kRshift):
            parallel_scalar_vectorized<T>(lhs, s, out, n,
                [](T v, T sh) { return bitwise_shift_value<T, false>(v, sh); },
                [](Vec v, Vec sh) { return v >> sh; });
            break;
        default:
            TP_THROW(TypeError, "bitwise: unsupported operation");
    }
}

void bitwise_binary_impl(const void* lhs, const void* rhs, void* out,
                         int64_t n, int dtype, int op) {
#define TP_BITWISE_BIN_CASE(ctype, name_)                        \
    case DType::name_:                                           \
        bitwise_binary_typed<ctype>(                             \
            static_cast<const ctype*>(lhs),                      \
            static_cast<const ctype*>(rhs),                      \
            static_cast<ctype*>(out), n, op);                    \
        break;
    switch (static_cast<DType>(dtype)) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BITWISE_BIN_CASE)
        default:
            TP_THROW(TypeError, "bitwise: unsupported dtype");
    }
#undef TP_BITWISE_BIN_CASE
}

void bitwise_scalar_impl(const void* lhs, int64_t scalar, void* out,
                         int64_t n, int dtype, int op) {
#define TP_BITWISE_SCALAR_CASE(ctype, name_)                     \
    case DType::name_:                                           \
        bitwise_scalar_typed<ctype>(                             \
            static_cast<const ctype*>(lhs), scalar,              \
            static_cast<ctype*>(out), n, op);                    \
        break;
    switch (static_cast<DType>(dtype)) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BITWISE_SCALAR_CASE)
        default:
            TP_THROW(TypeError, "bitwise: unsupported dtype");
    }
#undef TP_BITWISE_SCALAR_CASE
}

void bitwise_not_impl(const void* lhs, void* out, int64_t n, int dtype) {
#define TP_BITWISE_NOT_CASE(ctype, name_)                        \
    case DType::name_:                                           \
        parallel_unary_vectorized<ctype>(                        \
            static_cast<const ctype*>(lhs),                      \
            static_cast<ctype*>(out), n,                         \
            [](ctype value) { return static_cast<ctype>(~value); }, \
            [](tensorplay::vec::Vectorized<ctype> value) {       \
                return ~value;                                   \
            });                                                  \
        break;
    switch (static_cast<DType>(dtype)) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BITWISE_NOT_CASE)
        default:
            TP_THROW(TypeError, "bitwise_not: unsupported dtype");
    }
#undef TP_BITWISE_NOT_CASE
}

} // namespace

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(bitwise_binary_stub, &bitwise_binary_impl);
REGISTER_DISPATCH(bitwise_scalar_stub, &bitwise_scalar_impl);
REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(bitwise_binary_stub, &bitwise_binary_impl);
ALSO_REGISTER_AVX512_DISPATCH(bitwise_scalar_stub, &bitwise_scalar_impl);
ALSO_REGISTER_AVX512_DISPATCH(bitwise_not_stub, &bitwise_not_impl);
#endif

} // namespace cpu
} // namespace tensorplay
