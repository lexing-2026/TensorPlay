// Misc composite kernels: scalar-overload bridges, view wrappers, dtype
// routing for batched matmul, and the contiguous() family, expressed through
// already-registered dispatcher ops.

#include "Tensor.h"
#include "Scalar.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "tensorplay/ops/TensorRedispatchGenerated.h"

#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

Tensor wrapped_scalar(const Scalar& s, const Tensor& like) {
    return ops::full({}, s, like.dtype(), like.device());
}

} // namespace

// ---- copysign: scalar overloads lift the scalar to a 0-d tensor of the
//      operand's dtype, then reuse the registered tensor-tensor kernel.
Tensor copysign_scalar(const Tensor& self, const Scalar& other) {
    return ops::copysign(self, wrapped_scalar(other, self));
}

Tensor& copysign__scalar(Tensor& self, const Scalar& other) {
    return ops::copysign_(self, wrapped_scalar(other, self));
}

Tensor& copysign_scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
    return ops::copysign(self, wrapped_scalar(other, self), out);
}

Tensor& copysign__tensor(Tensor& self, const Tensor& other) {
    ops::copy_(self, ops::copysign(self, other));
    return self;
}

Tensor& copysign_tensor_out(const Tensor& self, const Tensor& other, Tensor& out) {
    Tensor r = ops::copysign(self, other);
    ops::copy_(out, r);
    return out;
}

// ---- clamp/clip with tensor bounds: promote each bound to a tensor and fall
//      back to the registered Scalar-based clamp when only one side is given.
Tensor clamp_tensor(const Tensor& self, const std::optional<Tensor>& min,
                    const std::optional<Tensor>& max) {
    if (min.has_value() && max.has_value()) {
        Tensor lo = *min;
        Tensor hi = *max;
        // element-wise clamp via minimum(maximum)
        Tensor t = ops::maximum(self, ops::mul(lo, ops::ones({}, lo.dtype(), lo.device())));
        (void)t;
        // broadcast-friendly form: clamp(min=lo) then clamp(max=hi)
        Scalar lo_s = lo.item();
        Scalar hi_s = hi.item();
        if (lo.numel() == 1 && hi.numel() == 1) {
            return ops::clamp(self, lo_s, hi_s);
        }
    }
    if (min.has_value() && min->numel() == 1 && !max.has_value()) {
        return ops::clamp_max(self, min->item());
    }
    if (max.has_value() && max->numel() == 1 && !min.has_value()) {
        return ops::clamp_min(self, max->item());
    }
    if (min.has_value() && max.has_value()) {
        // general tensor-bounds clamp: max(min(x, hi), lo) elementwise
        Tensor hi = *max;
        Tensor lo = *min;
        // route through where-based composition on the registered kernels
        Tensor capped = ops::where(ops::gt(self, hi), hi, self);
        return ops::where(ops::lt(capped, lo), lo, capped);
    }
    TP_THROW(RuntimeError, "clamp: at least one of 'min' or 'max' must be specified");
}

Tensor& clamp__tensor(Tensor& self, const std::optional<Tensor>& min,
                      const std::optional<Tensor>& max) {
    ops::copy_(self, clamp_tensor(self, min, max));
    return self;
}

Tensor& clamp_tensor_out(const Tensor& self, const std::optional<Tensor>& min,
                         const std::optional<Tensor>& max, Tensor& out) {
    ops::copy_(out, clamp_tensor(self, min, max));
    return out;
}

Tensor clip_tensor(const Tensor& self, const std::optional<Tensor>& min,
                   const std::optional<Tensor>& max) {
    return clamp_tensor(self, min, max);
}

Tensor& clip__tensor(Tensor& self, const std::optional<Tensor>& min,
                     const std::optional<Tensor>& max) {
    return clamp__tensor(self, min, max);
}

Tensor& clip_tensor_out(const Tensor& self, const std::optional<Tensor>& min,
                        const std::optional<Tensor>& max, Tensor& out) {
    return clamp_tensor_out(self, min, max, out);
}

// ---- _conj / _neg_view: view-style wrappers over the physical kernels
Tensor _conj_view(const Tensor& self) {
    Tensor r = self;
    // The logical conjugate view aliases storage; TensorPlay keeps the
    // concrete conjugate kernel for correctness under autograd.
    return ops::conj(self);
}

Tensor _neg_view(const Tensor& self) {
    return ops::neg(self);
}

Tensor _conj_physical_impl(const Tensor& self) {
    return ops::conj_physical(self);
}

Tensor& _conj_physical__(Tensor& self) {
    ops::copy_(self, ops::conj_physical(self));
    return self;
}

// ---- copy: explicit dtype/device-copy entry shared by Tensor.clone paths
Tensor copy_impl(const Tensor& self, bool non_blocking) {
    (void)non_blocking;
    return self.clone();
}

Tensor _copy_from_impl(const Tensor& src, const Tensor& dst) {
    TP_THROW(RuntimeError,
             "_copy_from is an internal autograd-view helper and should not be called directly");
}

Tensor _copy_from_and_resize_impl(const Tensor& src, const Tensor& dst) {
    TP_THROW(RuntimeError,
             "_copy_from_and_resize is an internal resize-view helper and should not be called directly");
}

// ---- contiguous: memory-format aware wrappers over the registered kernel
Tensor contiguous_default(const Tensor& self) {
    return ops::contiguous(self);
}

Tensor contiguous_format(const Tensor& self, int64_t memory_format) {
    if (self.is_contiguous(static_cast<MemoryFormat>(memory_format))) {
        return self;
    }
    if (memory_format == 2) {  // channels_last
        return ops::contiguous(self);
    }
    return ops::contiguous(self);
}

// ---- chalf: reinterpret/cast to the complex half dtype
Tensor chalf_impl(const Tensor& self, int64_t memory_format) {
    (void)memory_format;
    return self.to(DType::ComplexHalf);
}

// ---- _shape_as_tensor: sizes as an int64 tensor
Tensor _shape_as_tensor_impl(const Tensor& self) {
    std::vector<int64_t> s(self.dim());
    for (int64_t i = 0; i < self.dim(); ++i) {
        s[i] = self.size(i);
    }
    // assign each element through a one-element view
    Tensor r2 = ops::empty({static_cast<int64_t>(s.size())}, DType::Int64,
                           self.device(), false, false);
    for (int64_t i = 0; i < static_cast<int64_t>(s.size()); ++i) {
        Tensor cell = r2.select(0, i);
        ops::fill_(cell, Scalar(s[static_cast<size_t>(i)]));
    }
    return r2;
}

Tensor _dim_arange_impl(const Tensor& like, int64_t dim, int64_t device_index) {
    (void)device_index;
    return ops::arange(Scalar(like.size(dim)), DType::Int64, like.device());
}

// ---- _masked_scale
Tensor _masked_scale_impl(const Tensor& self, const Tensor& mask, double scale) {
    return ops::mul(self, ops::where(mask.to(DType::Bool),
                                     ops::full({}, Scalar(scale), self.dtype(), self.device()),
                                     ops::full({}, Scalar(1.0), self.dtype(), self.device())));
}

// ---- _mkldnn_transpose / _to_sparse bridges route to the registered kernels
Tensor _to_sparse_impl(const Tensor& self) {
    return ops::to_sparse(self);
}

TENSORPLAY_LIBRARY_IMPL(Composite, MiscBridgeComposites) {
    m.impl("copysign_.Scalar", copysign__scalar);
    m.impl("copysign.Scalar_out", copysign_scalar_out);
    m.impl("clamp.Tensor", clamp_tensor);
    m.impl("clamp_.Tensor", clamp__tensor);
    m.impl("clip.Tensor", clip_tensor);
    m.impl("clip_.Tensor", clip__tensor);
    m.impl("clip.out", clip_tensor_out);
    m.impl("_conj", _conj_view);
    m.impl("_neg_view", _neg_view);
    m.impl("_conj_physical", _conj_physical_impl);
    m.impl("_conj_physical_", _conj_physical__);
    m.impl("copy", copy_impl);
    m.impl("contiguous", contiguous_default);
    m.impl("contiguous.memory_format", contiguous_format);
    m.impl("chalf", chalf_impl);
    m.impl("_shape_as_tensor", _shape_as_tensor_impl);
    m.impl("_dim_arange", _dim_arange_impl);
    m.impl("_masked_scale", _masked_scale_impl);
    m.impl("_to_sparse", _to_sparse_impl);
}

} // namespace composite
} // namespace tensorplay
