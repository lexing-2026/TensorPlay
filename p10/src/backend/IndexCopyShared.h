#pragma once

// The rules every index_copy entry point enforces, whatever the device.
//
// The kernels read the source as a dense stack of slices, one slice per index
// entry, laid out as (outer, n_idx, inner) around the copied dimension.  A
// source whose slice extents disagree with the destination would walk past the
// end of its own buffer, so the geometry is rejected up front and a scalar
// source is materialized into that dense layout.

#include "Tensor.h"
#include "Exception.h"

#include <vector>

namespace tensorplay {
namespace index_copy_detail {

// `dim` must already be wrapped against self's rank.
inline void check_index_copy_geometry(const Tensor& self, int64_t dim,
                                      const Tensor& index,
                                      const Tensor& source,
                                      const char* op) {
    if (index.dim() > 1) {
        TP_THROW(RuntimeError, op, ": index must be 0- or 1-dimensional, got ",
                 index.dim(), " dimension(s)");
    }
    if (source.dim() == 0) {
        if (index.numel() != 1) {
            TP_THROW(RuntimeError, op,
                     ": a scalar source needs exactly one index, got ",
                     index.numel());
        }
    } else if (self.dim() != 0 && source.dim() != self.dim()) {
        TP_THROW(RuntimeError, op,
                 ": source and destination must have the same dimensionality, got ",
                 source.dim(), " and ", self.dim());
    }
    if (self.dtype() != source.dtype()) {
        TP_THROW(RuntimeError, op,
                 ": source and destination must share a dtype, got ",
                 toString(source.dtype()), " and ", toString(self.dtype()));
    }
    if (self.device() != source.device() || self.device() != index.device()) {
        TP_THROW(RuntimeError, op,
                 ": self, index and source must live on one device, self is on ",
                 self.device().toString());
    }
    if (self.dim() != 0 && source.dim() != 0) {
        bool same = true;
        for (int64_t i = 0; same && i < self.dim(); ++i) {
            if (i != dim && self.size(i) != source.size(i)) same = false;
        }
        if (!same) {
            TP_THROW(RuntimeError, op,
                     ": source slice shape must match the destination slice shape "
                     "away from dim ", dim);
        }
        if (index.numel() != source.size(dim)) {
            TP_THROW(RuntimeError, op, ": number of indices (", index.numel(),
                     ") must equal source.size(", dim, ") (",
                     source.size(dim), ")");
        }
    }
}

// A scalar source stands for one slice broadcast across the remaining
// extents; expand it into the dense (outer, n_idx, inner) layout the kernels
// address, so no device path reads past its single element.
inline Tensor expand_scalar_source(const Tensor& source, const Tensor& self,
                                   int64_t dim, int64_t n_idx) {
    if (source.dim() != 0) return source;
    if (self.dim() == 0) return source.reshape({1});
    std::vector<int64_t> sizes;
    sizes.reserve(static_cast<size_t>(self.dim()));
    for (int64_t i = 0; i < self.dim(); ++i) {
        sizes.push_back(i == dim ? n_idx : self.size(i));
    }
    return source.reshape({1}).expand(sizes).contiguous();
}

} // namespace index_copy_detail
} // namespace tensorplay
