// TensorIteratorOps.h -- thin dtype-dispatched helpers on top of
// binary functor; iteration (reorder / coalesce / parallel chunks /
// byte-stride addressing) is owned by TensorIterator.
#pragma once

#include "TensorIterator.h"
#include "Complex.h"

namespace tensorplay {
namespace cpu {

/// Run op(a_i, b_i) -> out_i over the broadcast of a and b, writing into
/// the pre-allocated `out` (dtype must equal the promoted common dtype).
/// `op` must be generic over the element type (auto lambda or template
/// functor), matching TENSORPLAY_FORALL_SCALAR_TYPES instantiation.
template <typename Op>
inline void ti_apply_binary(Tensor& out, const Tensor& a, const Tensor& b,
                            Op op) {
    TensorIterator iter = TensorIterator::binary_op(out, a, b);
    switch (out.dtype()) {
#define TP_TI_CASE(ctype, name)                                            \
        case DType::name: {                                                \
            iter.for_each([&op](char** data, const int64_t* strides,       \
                                int64_t n) {                               \
                char* r = data[0];                                         \
                const char* x = data[1];                                   \
                const char* y = data[2];                                   \
                constexpr int64_t SZ = static_cast<int64_t>(sizeof(ctype)); \
                if (strides[0] == SZ && strides[1] == SZ &&                \
                    strides[2] == SZ) {                                    \
                    ctype* rp = reinterpret_cast<ctype*>(r);               \
                    const ctype* xp = reinterpret_cast<const ctype*>(x);   \
                    const ctype* yp = reinterpret_cast<const ctype*>(y);   \
                    for (int64_t i = 0; i < n; ++i) rp[i] = op(xp[i], yp[i]); \
                } else {                                                   \
                    for (int64_t i = 0; i < n; ++i) {                      \
                        *reinterpret_cast<ctype*>(r + i * strides[0]) = op(\
                            *reinterpret_cast<const ctype*>(x + i * strides[1]),\
                            *reinterpret_cast<const ctype*>(y + i * strides[2]));\
                    }                                                      \
                }                                                          \
            });                                                            \
            break;                                                         \
        }
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_TI_CASE)
#undef TP_TI_CASE
        default:
            TP_THROW(TypeError, "ti_apply_binary: unsupported dtype");
    }
}

/// Complex-capable flavor of ti_apply_binary for pure arithmetic functors
/// (+, -, *, /, pow): dispatch additionally covers the four complex
/// dtypes.  Call sites whose functor relies on operator< / fmod / casts to
/// real types (clamp, gcd, remainder, ...) must stay on ti_apply_binary --
template <typename Op>
inline void ti_apply_arith(Tensor& out, const Tensor& a, const Tensor& b,
                           Op op) {
    TensorIterator iter = TensorIterator::binary_op(out, a, b);
    switch (out.dtype()) {
#define TP_TI_CASE(ctype, name)                                            \
        case DType::name: {                                                \
            iter.for_each([&op](char** data, const int64_t* strides,       \
                                int64_t n) {                               \
                char* r = data[0];                                         \
                const char* x = data[1];                                   \
                const char* y = data[2];                                   \
                constexpr int64_t SZ = static_cast<int64_t>(sizeof(ctype)); \
                if (strides[0] == SZ && strides[1] == SZ &&                \
                    strides[2] == SZ) {                                    \
                    ctype* rp = reinterpret_cast<ctype*>(r);               \
                    const ctype* xp = reinterpret_cast<const ctype*>(x);   \
                    const ctype* yp = reinterpret_cast<const ctype*>(y);   \
                    for (int64_t i = 0; i < n; ++i) rp[i] = op(xp[i], yp[i]); \
                } else {                                                   \
                    for (int64_t i = 0; i < n; ++i) {                      \
                        *reinterpret_cast<ctype*>(r + i * strides[0]) = op(\
                            *reinterpret_cast<const ctype*>(x + i * strides[1]),\
                            *reinterpret_cast<const ctype*>(y + i * strides[2]));\
                    }                                                      \
                }                                                          \
            });                                                            \
            break;                                                         \
        }
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_TI_CASE)
#undef TP_TI_CASE
        // Full-width complexes dispatch like any scalar type.
#define TP_TI_CX(ctype, name)                                          \
        case DType::name: {                                                \
            iter.for_each([&op](char** data, const int64_t* strides,       \
                                int64_t n) {                               \
                char* r = data[0];                                         \
                const char* x = data[1];                                   \
                const char* y = data[2];                                   \
                constexpr int64_t SZ = static_cast<int64_t>(sizeof(ctype)); \
                if (strides[0] == SZ && strides[1] == SZ &&                \
                    strides[2] == SZ) {                                    \
                    ctype* rp = reinterpret_cast<ctype*>(r);               \
                    const ctype* xp = reinterpret_cast<const ctype*>(x);   \
                    const ctype* yp = reinterpret_cast<const ctype*>(y);   \
                    for (int64_t i = 0; i < n; ++i) rp[i] = op(xp[i], yp[i]); \
                } else {                                                   \
                    for (int64_t i = 0; i < n; ++i) {                      \
                        *reinterpret_cast<ctype*>(r + i * strides[0]) = op(\
                            *reinterpret_cast<const ctype*>(x + i * strides[1]),\
                            *reinterpret_cast<const ctype*>(y + i * strides[2]));\
                    }                                                      \
                }                                                          \
            });                                                            \
            break;                                                         \
        }
        TP_TI_CX(complex<float>, ComplexFloat)
        TP_TI_CX(complex<double>, ComplexDouble)
#undef TP_TI_CX
#define TP_TI_CX_RED_CASE(halftype, name)                                  \
        case DType::name: {                                                \
            using cxh_t = complex<halftype>;                          \
            iter.for_each([&op](char** data, const int64_t* strides,       \
                                int64_t n) {                               \
                char* r = data[0];                                         \
                const char* x = data[1];                                   \
                const char* y = data[2];                                   \
                constexpr int64_t SZ = static_cast<int64_t>(sizeof(cxh_t)); \
                for (int64_t i = 0; i < n; ++i) {                          \
                    const cxh_t& xi = *reinterpret_cast<const cxh_t*>(x + i * strides[1]); \
                    const cxh_t& yi = *reinterpret_cast<const cxh_t*>(y + i * strides[2]); \
                    complex<float> xo(static_cast<float>(xi.real()),   \
                                           static_cast<float>(xi.imag()));  \
                    complex<float> yo(static_cast<float>(yi.real()),   \
                                           static_cast<float>(yi.imag()));  \
                    complex<float> ro = op(xo, yo);                    \
                    *reinterpret_cast<cxh_t*>(r + i * strides[0]) = cxh_t(  \
                        static_cast<halftype>(ro.real()),                   \
                        static_cast<halftype>(ro.imag()));                  \
                }                                                          \
            });                                                            \
            break;                                                         \
        }
        TP_TI_CX_RED_CASE(Half, ComplexHalf)
        TP_TI_CX_RED_CASE(BFloat16, BComplex32)
#undef TP_TI_CX_RED_CASE
        default:
            TP_THROW(TypeError, "ti_apply_arith: unsupported dtype");
    }
}

/// defines over complex tensors (component-wise), so they get their own
/// applier instead of teaching the ordering-only ti_apply_compare about
/// complex dtypes (complex has no operator<).
template <typename Op>
inline void ti_apply_equality(Tensor& out, const Tensor& a, const Tensor& b,
                              DType common, Op op) {
    TensorIterator iter = TensorIterator::binary_op(out, a, b);
    switch (common) {
#define TP_TI_EQ(ctype, name)                                              \
        case DType::name: {                                                \
            iter.for_each([&op](char** data, const int64_t* strides,       \
                                int64_t n) {                               \
                char* r = data[0];                                         \
                const char* x = data[1];                                   \
                const char* y = data[2];                                   \
                constexpr int64_t SZ = static_cast<int64_t>(sizeof(ctype)); \
                if (strides[1] == SZ && strides[2] == SZ &&                \
                    strides[0] == static_cast<int64_t>(sizeof(bool))) {    \
                    bool* rp = reinterpret_cast<bool*>(r);                 \
                    const ctype* xp = reinterpret_cast<const ctype*>(x);   \
                    const ctype* yp = reinterpret_cast<const ctype*>(y);   \
                    for (int64_t i = 0; i < n; ++i) rp[i] = op(xp[i], yp[i]); \
                } else {                                                   \
                    for (int64_t i = 0; i < n; ++i) {                      \
                        *reinterpret_cast<bool*>(r + i * strides[0]) = op( \
                            *reinterpret_cast<const ctype*>(x + i * strides[1]),\
                            *reinterpret_cast<const ctype*>(y + i * strides[2]));\
                    }                                                      \
                }                                                          \
            });                                                            \
            break;                                                         \
        }
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_TI_EQ)
        TP_TI_EQ(complex<Half>, ComplexHalf)
        TP_TI_EQ(complex<float>, ComplexFloat)
        TP_TI_EQ(complex<double>, ComplexDouble)
        TP_TI_EQ(complex<BFloat16>, BComplex32)
#undef TP_TI_EQ
        default:
            TP_THROW(TypeError, "ti_apply_equality: unsupported dtype");
    }
}

/// Comparison flavor: inputs share `common`; out is pre-allocated Bool.
template <typename Op>
inline void ti_apply_compare(Tensor& out, const Tensor& a, const Tensor& b,
                             DType common, Op op) {
    TensorIterator iter = TensorIterator::binary_op(out, a, b);
    switch (common) {
#define TP_TI_CMP(ctype, name)                                                     case DType::name: {                                                            iter.for_each([&op](char** data, const int64_t* strides,                                       int64_t n) {                                               char* r = data[0];                                                         const char* x = data[1];                                                   const char* y = data[2];                                                   constexpr int64_t SZ = static_cast<int64_t>(sizeof(ctype));                 if (strides[1] == SZ && strides[2] == SZ &&                                    strides[0] == static_cast<int64_t>(sizeof(bool))) {                        bool* rp = reinterpret_cast<bool*>(r);                                     const ctype* xp = reinterpret_cast<const ctype*>(x);                       const ctype* yp = reinterpret_cast<const ctype*>(y);                       for (int64_t i = 0; i < n; ++i) rp[i] = op(xp[i], yp[i]);                 } else {                                                                       for (int64_t i = 0; i < n; ++i) {                                              *reinterpret_cast<bool*>(r + i * strides[0]) = op(                             *reinterpret_cast<const ctype*>(x + i * strides[1]),                            *reinterpret_cast<const ctype*>(y + i * strides[2]));                    }                                                                      }                                                                      });                                                                        break;                                                                 }
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_TI_CMP)
#undef TP_TI_CMP
        default:
            TP_THROW(TypeError, "ti_apply_compare: unsupported dtype");
    }
}

}  // namespace cpu
}  // namespace tensorplay
