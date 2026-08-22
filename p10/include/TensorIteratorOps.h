// TensorIteratorOps.h -- thin dtype-dispatched helpers on top of
// TensorIterator for elementwise kernels.  Mirrors ATen's cpu_kernel_vec role: kernels hand in an output tensor plus broadcast inputs and a
// binary functor; iteration (reorder / coalesce / parallel chunks /
// byte-stride addressing) is owned by TensorIterator.
#pragma once

#include "TensorIterator.h"

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
