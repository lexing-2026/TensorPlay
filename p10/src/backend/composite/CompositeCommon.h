// Shared helpers for the backend-neutral composite kernels that live beside
// those translation units -- nothing is exported through the public p10 API.
#pragma once

#include "Tensor.h"
#include "DispatchKey.h"
#include "Dispatcher.h"

#include <cstdint>
#include <string>

namespace tensorplay {
namespace composite {

// MemoryFormat::Contiguous tag consumed by ops::clone / ops::contiguous.
constexpr int64_t kContiguous = 0;

// True when the tensor carries an active transform level (vmap/grad): its
// public metadata describes the unbatched view while the payload lives in the
// transform wrapper, so plain backend kernels must not touch it.
inline bool under_active_transform(const Tensor& t) {
    const auto impl = t.unsafeGetTensorImpl();
    return impl && impl->is_batched();
}

// Refuse tensors inside an active transform layer.  Generated callers collapse
// vmap keys to their backend component before the composite fallthrough, so a
// composite that re-dispatches with the plain ops:: wrappers would run backend
// kernels on the wrapper and corrupt downstream batching.  Decomposition-style
// composites route around this through redispatch_below_transform; everything
// else fails the same way it did before the op was registered.
inline void reject_active_transform(const Tensor& t, const char* op) {
    if (under_active_transform(t)) {
        TP_THROW(NotImplementedError, std::string(op),
                 " is not supported for tensors inside an active transform "
                 "(vmap/grad) layer");
    }
}

// Decomposition-style re-dispatch: forward `op` with the key derived from the
// arguments kept intact, so an inner call from a composite lands on the
// transform layer's own batch rules instead of the raw backend kernel.  The
// caller must have established that a transform is active.
template <typename Return, typename... Args>
Return redispatch_below_transform(const char* op, Args... args) {
    DispatchKey key = dispatchKeyForTensorArgs(args...);
    TP_CHECK(is_vmap_key(key),
             op, ": expected an active transform key for decomposition");
    return DispatchStub<Return, Args...>::call(
        std::string(op), key, std::forward<Args>(args)...);
}

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
