// Shared helpers for the shape-only kernels of the meta device.
//
// A meta tensor records sizes, strides, dtype and device but owns no element
// data, so every kernel here returns an unbacked allocation created straight
// through TensorImpl -- never through the dispatcher, which would redispatch
// back into the same-named op and recurse.  The helpers also carry the small
// dtype and shape rules (promotion, reduction extents) that the per-category
// kernel files share.

#pragma once

#include "Tensor.h"
#include "TensorImpl.h"
#include "Context.h"
#include "DType.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "Utils.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace meta {

inline Tensor empty_meta(std::vector<int64_t> size, DType dtype,
                         const Device& device) {
    return Tensor(make_intrusive<TensorImpl>(size, dtype, device));
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

inline std::vector<int64_t> broadcast2(const Tensor& a, const Tensor& b) {
    return broadcast_shapes(shape_of(a), shape_of(b));
}

inline void check_all_meta(const Tensor& a, const Tensor& b) {
    if (!a.device().is_meta() || !b.device().is_meta()) {
        TP_THROW(DeviceMismatchError,
                 "expected both operands on the meta device but got ",
                 a.device().toString(), " and ", b.device().toString());
    }
}

// An in-place operand must fit the target: right-aligned, every source
// dimension is either 1 or matches the destination.
inline void check_inplace_fits(const Tensor& self, const Tensor& other) {
    const int64_t nd = self.dim();
    const int64_t od = other.dim();
    for (int64_t i = 1; i <= od; ++i) {
        const int64_t src = other.size(od - i);
        const int64_t dst = i <= nd ? self.size(nd - i) : 1;
        if (src != 1 && src != dst) {
            TP_THROW(RuntimeError, "the shape of the operand (",
                     other.size(od - i), " at position ", od - i,
                     ") cannot be applied in place to a destination of size ",
                     dst);
        }
    }
}

// Whether an element type takes part in the elementwise kernels' dispatch:
// every integral, boolean and non-fp8 floating type.
inline bool is_scalar_dtype(DType dtype) {
    return dtype == DType::Bool || isIntegralType(dtype) ||
           (isFloatingType(dtype) && !isFloat8Type(dtype));
}

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim, ")");
    }
    return dim;
}

inline Device resolve_device(const std::optional<Device>& device) {
    return device.has_value() ? *device : globalContext().defaultDevice();
}

inline DType resolve_factory_dtype(const std::optional<DType>& dtype) {
    return (dtype.has_value() && *dtype != DType::Undefined)
               ? *dtype
               : globalContext().defaultDType();
}

inline DType resolve_like_dtype(DType dtype, const Tensor& self) {
    return dtype == DType::Undefined ? self.dtype() : dtype;
}

inline Device resolve_like_device(const std::optional<Device>& device,
                                  const Tensor& self) {
    return device.has_value() ? *device : self.device();
}

// A float-only pointwise op answers an integral or boolean operand in single
// precision, because the operation has no integral form to answer in; a
// floating or complex operand keeps its own type.
inline DType unary_float_dtype(DType dtype) {
    if (isIntegralType(dtype, /*includeBool=*/true)) {
        return DType::Float32;
    }
    return dtype;
}

// The float-math sweeps compute at floating precision and have no complex
// form anywhere: a floating operand keeps its width, every other operand --
// integers, booleans, complex planes -- is answered in single precision.
inline DType float_math_dtype(DType dtype) {
    return isFloatingType(dtype) ? dtype : DType::Float32;
}

// The real-number sweeps (logarithms, exponentials, roots, trigonometry)
// keep a complex lane of their own: a whole-number operand answers in single
// precision, a boolean or 8-bit float operand has no form to run in, and a
// floating or complex operand keeps its width.
inline DType real_math_dtype(DType dtype) {
    if (dtype == DType::Bool || isFloat8Type(dtype)) {
        TP_THROW(TypeError, "Unsupported dtype (expected float)");
    }
    return isIntegralType(dtype) ? DType::Float32 : dtype;
}

// Result dtype for add/sub-style arithmetic: promote the operands, and a
// floating alpha lifts an integer pair to at least float.
inline DType add_like_dtype(DType self_dtype, DType other_dtype,
                            const Scalar& alpha) {
    DType result = promoteTypes(self_dtype, other_dtype);
    if (alpha.isFloatingPoint() && !isFloatingType(result)) {
        result = promoteTypes(result, DType::Float32);
    }
    return result;
}

inline DType add_like_scalar_dtype(DType self_dtype, const Scalar& other,
                                   const Scalar& alpha) {
    return add_like_dtype(self_dtype, result_type(other, self_dtype), alpha);
}

// True division always falls back to float when the promoted pair is integral.
inline DType div_like_dtype(DType self_dtype, DType other_dtype) {
    DType result = promoteTypes(self_dtype, other_dtype);
    if (isIntegralType(result, /*include_bool=*/true)) {
        result = DType::Float32;
    }
    return result;
}

inline DType div_like_scalar_dtype(DType self_dtype, const Scalar& other) {
    return div_like_dtype(self_dtype, result_type(other, self_dtype));
}

// The extents kept by a reduction over `dims` with keepdim, shared by sum,
// mean, var, norm and the index-reduction family.  An empty list reduces
// every dimension (the one virtual dimension of a rank-0 operand).
inline std::vector<int64_t> reduce_out_shape(const Tensor& self,
                                             const std::vector<int64_t>& dims,
                                             bool keepdim) {
    const int64_t nd = self.dim();
    std::vector<bool> reduced(static_cast<size_t>(nd), false);
    if (dims.empty()) {
        if (nd == 0) {
            reduced.push_back(true);
        } else {
            for (int64_t i = 0; i < nd; ++i) {
                reduced[static_cast<size_t>(i)] = true;
            }
        }
    }
    for (int64_t d : dims) {
        int64_t dim = d;
        if (nd == 0) {
            if (dim != 0 && dim != -1) {
                TP_THROW(IndexError, "reduction dimension ", dim,
                         " out of range for a rank-", nd, " tensor");
            }
            reduced.push_back(true);
            continue;
        }
        if (dim < 0) dim += nd;
        if (dim < 0 || dim >= nd) {
            TP_THROW(IndexError, "reduction dimension ", dim,
                     " out of range for a rank-", nd, " tensor");
        }
        reduced[static_cast<size_t>(dim)] = true;
    }
    std::vector<int64_t> out;
    out.reserve(reduced.size());
    for (size_t i = 0; i < reduced.size(); ++i) {
        if (!reduced[i]) {
            out.push_back(nd == 0 ? 0 : self.size(static_cast<int64_t>(i)));
        } else if (keepdim) {
            out.push_back(1);
        }
    }
    return out;
}

} // namespace meta
} // namespace tensorplay