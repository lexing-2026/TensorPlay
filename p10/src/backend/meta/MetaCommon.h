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