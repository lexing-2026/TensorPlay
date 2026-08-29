// Shared helpers for the backend-neutral composite kernels that live beside
// those translation units -- nothing is exported through the public p10 API.
#pragma once

#include "Tensor.h"

#include <cstdint>

namespace tensorplay {
namespace composite {

// MemoryFormat::Contiguous tag consumed by ops::clone / ops::contiguous.
constexpr int64_t kContiguous = 0;

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (ndim == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range (expected to be in range of [-1, 0], but got ",
                     dim, ")");
        }
        return 0;
    }
    if (dim < -ndim || dim > ndim - 1) {
        TP_THROW(IndexError,
                 "Dimension out of range (expected to be in range of [", -ndim,
                 ", ", ndim - 1, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

// Natural dtype of a scalar as materialized by scalar_tensor (int -> int64,
inline DType scalar_natural_dtype(const Scalar& s) {
    if (s.isComplex()) return DType::ComplexFloat;
    if (s.isFloatingPoint()) return DType::Float32;
    if (s.isBoolean()) return DType::Bool;
    return DType::Int64;
}

} // namespace composite
} // namespace tensorplay
