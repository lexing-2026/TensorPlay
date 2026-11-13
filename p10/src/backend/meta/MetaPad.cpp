// Shape-only kernels for the non-constant padding modes on the meta device.
//
// Reflection, replication and circular padding only touch the trailing
// dimensions named by the pad list, so the answer is the input geometry with
// each named dimension grown by its pad pair.  The shape validation runs
// before the dtype gate, matching the order a computing backend reports
// failures; the backward answers the input geometry again, with the
// stricter floating-only dtype gate of the scatter.
//
// The rank-specific spellings share the nd kernel, whose name also reports
// the errors; circular padding has no per-rank spelling.

#include "MetaCommon.h"

#include "Exception.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

enum class PadIndexMode { Reflect, Replicate, Circular };

void pad_mode_check(const char* name, const Tensor& self,
                    const std::vector<int64_t>& pad, PadIndexMode mode) {
    if (pad.size() % 2 != 0)
        TP_THROW(ValueError, name, ": length of pad must be even but it equals ",
                 pad.size());
    const int64_t k = static_cast<int64_t>(pad.size()) / 2;
    if (k > self.dim())
        TP_THROW(ValueError, name,
                 ": padding length too large for input dimensionality");
    for (int64_t i = 0; i < k; ++i) {
        const int64_t dim = self.dim() - 1 - i;
        if (mode == PadIndexMode::Reflect) {
            // Reflection without repeating an edge sample needs at least one
            // interior point on each side of the source extent.
            if (pad[2 * i] >= self.size(dim) || pad[2 * i + 1] >= self.size(dim))
                TP_THROW(ValueError, name,
                         ": padding size should be less than the corresponding "
                         "input dimension");
        } else if (mode == PadIndexMode::Circular) {
            // A wrap may land on each source position at most once.
            if (pad[2 * i] > self.size(dim) || pad[2 * i + 1] > self.size(dim))
                TP_THROW(ValueError, name,
                         ": padding value causes wrapping around more than once");
        }
    }
}

std::vector<int64_t> pad_out_sizes(const Tensor& self,
                                   const std::vector<int64_t>& pad) {
    std::vector<int64_t> out = shape_of(self);
    const int64_t k = static_cast<int64_t>(pad.size()) / 2;
    for (int64_t i = 0; i < k; ++i) {
        out[static_cast<size_t>(out.size() - 1 - i)] += pad[2 * i] + pad[2 * i + 1];
    }
    return out;
}

bool pad_forward_dtype(DType dtype) {
    switch (dtype) {
        case DType::Int8:
        case DType::UInt8:
        case DType::Int32:
        case DType::Float32:
        case DType::Float64:
        case DType::Float16:
        case DType::BFloat16:
            return true;
        default:
            return false;
    }
}

bool pad_backward_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float64 ||
           dtype == DType::Float16 || dtype == DType::BFloat16;
}

Tensor pad_forward_meta(const char* name, const Tensor& self,
                        const std::vector<int64_t>& pad, PadIndexMode mode) {
    pad_mode_check(name, self, pad, mode);
    if (!pad_forward_dtype(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "non-constant padding does not support this dtype");
    }
    return empty_meta(pad_out_sizes(self, pad), self.dtype(), self.device());
}

Tensor pad_backward_meta(const char* name, const Tensor& grad_output,
                         const Tensor& self, const std::vector<int64_t>& pad,
                         PadIndexMode mode) {
    (void)grad_output;
    pad_mode_check(name, self, pad, mode);
    if (!pad_backward_dtype(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "non-constant padding only supports floating dtypes");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor reflection_pad_nd_kernel(const Tensor& self, const std::vector<int64_t>& pad) {
    return pad_forward_meta("reflection_pad_nd", self, pad, PadIndexMode::Reflect);
}

Tensor reflection_pad_nd_backward_kernel(const Tensor& grad_output,
                                         const Tensor& self,
                                         const std::vector<int64_t>& pad) {
    return pad_backward_meta("reflection_pad_nd_backward", grad_output, self, pad,
                             PadIndexMode::Reflect);
}

Tensor replication_pad_nd_kernel(const Tensor& self, const std::vector<int64_t>& pad) {
    return pad_forward_meta("replication_pad_nd", self, pad, PadIndexMode::Replicate);
}

Tensor replication_pad_nd_backward_kernel(const Tensor& grad_output,
                                          const Tensor& self,
                                          const std::vector<int64_t>& pad) {
    return pad_backward_meta("replication_pad_nd_backward", grad_output, self, pad,
                             PadIndexMode::Replicate);
}

Tensor circular_pad_nd_kernel(const Tensor& self, const std::vector<int64_t>& pad) {
    return pad_forward_meta("circular_pad_nd", self, pad, PadIndexMode::Circular);
}

Tensor circular_pad_nd_backward_kernel(const Tensor& grad_output,
                                       const Tensor& self,
                                       const std::vector<int64_t>& pad) {
    return pad_backward_meta("circular_pad_nd_backward", grad_output, self, pad,
                             PadIndexMode::Circular);
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaPad) {
    m.impl("reflection_pad_nd", reflection_pad_nd_kernel);
    m.impl("reflection_pad_nd_backward", reflection_pad_nd_backward_kernel);
    m.impl("reflection_pad1d", reflection_pad_nd_kernel);
    m.impl("reflection_pad2d", reflection_pad_nd_kernel);
    m.impl("reflection_pad3d", reflection_pad_nd_kernel);
    m.impl("reflection_pad1d_backward", reflection_pad_nd_backward_kernel);
    m.impl("reflection_pad2d_backward", reflection_pad_nd_backward_kernel);
    m.impl("reflection_pad3d_backward", reflection_pad_nd_backward_kernel);
    m.impl("replication_pad_nd", replication_pad_nd_kernel);
    m.impl("replication_pad_nd_backward", replication_pad_nd_backward_kernel);
    m.impl("replication_pad1d", replication_pad_nd_kernel);
    m.impl("replication_pad2d", replication_pad_nd_kernel);
    m.impl("replication_pad3d", replication_pad_nd_kernel);
    m.impl("replication_pad1d_backward", replication_pad_nd_backward_kernel);
    m.impl("replication_pad2d_backward", replication_pad_nd_backward_kernel);
    m.impl("replication_pad3d_backward", replication_pad_nd_backward_kernel);
    m.impl("circular_pad_nd", circular_pad_nd_kernel);
    m.impl("circular_pad_nd_backward", circular_pad_nd_backward_kernel);
}

} // namespace meta
} // namespace tensorplay
