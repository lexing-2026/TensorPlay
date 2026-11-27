// Misc composite kernels: scalar-overload bridges, view wrappers, dtype
// routing for batched matmul, and the contiguous() family, expressed through
// already-registered dispatcher ops.

#include "Tensor.h"
#include "Scalar.h"
#include "Exception.h"
#include "GradMode.h"
#include "InferenceMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "tensorplay/ops/TensorRedispatchGenerated.h"
#include "OutWrite.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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

Tensor& write_broadcast_out(const char* op, Tensor value, Tensor& out) {
    if (!out.defined()) {
        out = value;
        return out;
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 op, ": output must be on the same device as the result");
    }
    out.resize_(static_cast<std::vector<int64_t>>(value.shape()));
    out.copy_(value);
    return out;
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
    return write_broadcast_out("copysign", ops::copysign(self, other), out);
}

// ---- clamp/clip with tensor bounds: promote each bound to a tensor and fall
//      back to the registered Scalar-based clamp when only one side is given.
Tensor clamp_tensor(const Tensor& self, const std::optional<Tensor>& min_arg,
                    const std::optional<Tensor>& max_arg) {
    std::optional<Tensor> min = min_arg;
    std::optional<Tensor> max = max_arg;
    // .item() is only legal on 1-element tensors; any other bound shape goes
    // through the elementwise maximum/minimum composition below.
    const auto is_scalar_bound = [](const std::optional<Tensor>& b) {
        return !b.has_value() || b->numel() == 1;
    };
    if (is_scalar_bound(min) && is_scalar_bound(max)) {
        const std::optional<Scalar> lo = min.has_value() ? std::optional<Scalar>(min->item())
                                                         : std::nullopt;
        const std::optional<Scalar> hi = max.has_value() ? std::optional<Scalar>(max->item())
                                                         : std::nullopt;
        return ops::clamp(self, lo, hi);
    }
    // Elementwise bounds: clamp(min=lo) then clamp(max=hi) via the maximum /
    // minimum kernels, which broadcast like any other binary op.
    Tensor cur = self;
    if (min.has_value()) {
        cur = ops::maximum(cur, *min);
    }
    if (max.has_value()) {
        cur = ops::minimum(cur, *max);
    }
    return cur;
}

Tensor& clamp__tensor(Tensor& self, const std::optional<Tensor>& min,
                      const std::optional<Tensor>& max) {
    ops::copy_(self, clamp_tensor(self, min, max));
    return self;
}

Tensor& clamp_tensor_out(const Tensor& self, const std::optional<Tensor>& min,
                         const std::optional<Tensor>& max, Tensor& out) {
    return write_broadcast_out("clamp", clamp_tensor(self, min, max), out);
}

Tensor clip_tensor(const Tensor& self, const std::optional<Tensor>& min,
                   const std::optional<Tensor>& max) {
    return ops::clamp(self, min, max);
}

Tensor& clip__tensor(Tensor& self, const std::optional<Tensor>& min,
                     const std::optional<Tensor>& max) {
    return clamp__tensor(self, min, max);
}

Tensor& clip_scalar_out(const Tensor& self, const std::optional<Scalar>& min,
                        const std::optional<Scalar>& max, Tensor& out) {
    return write_out(out, ops::clamp(self, min, max));
}

// ---- _conj / _neg_view: view-style wrappers over the physical kernels
Tensor _conj_view(const Tensor& self) {
    Tensor r = self;
    // The logical conjugate view aliases storage; TensorPlay keeps the
    // concrete conjugate kernel for correctness under autograd.
    return ops::conj(self);
}

Tensor _neg_view(const Tensor& self) {
    // A negated view carries its sign as metadata: the bytes stay untouched and
    // the sign is folded in when the values are first reached.  Handing the
    // bytes to a second owner launches no kernel and allocates nothing, so
    // building the view is free; the first access to either side then pays at
    // most one copy plus one negation, and a view dropped before anyone reaches
    // it pays nothing at all.
    const TensorImpl* impl = self.unsafeGetTensorImpl().get();
    if (impl == nullptr || self.is_sparse() || self.is_batched()) {
        return ops::neg(self);
    }
    const Storage& storage = impl->storage();
    // Four cases keep the eager kernel: bytes the storage does not own (they
    // cannot be copied privately later), a sign that is already deferred (two
    // negations cancel, which the eager kernel expresses directly), a backend
    // that publishes raw storage handles instead of data pointers, and an input
    // that is differentiable in this mode, whose backward pass needs a node.
    if (!storage.defined() || storage.allocator() == nullptr ||
        storage.is_negated() || impl->has_quantizer() ||
        !(self.device().is_cpu() || self.device().is_cuda()) ||
        (self.requires_grad() && GradMode::is_enabled() && !InferenceMode::is_enabled())) {
        return ops::neg(self);
    }
    Storage deferred = storage.share_bytes();
    deferred.mark_negated();
    return Tensor(make_intrusive<TensorImpl>(
        std::move(deferred),
        static_cast<std::vector<int64_t>>(self.shape()),
        self.strides(),
        impl->dtype(),
        impl->storage_offset()));
}

Tensor _conj_physical_impl(const Tensor& self) {
    return ops::conj_physical(self);
}

Tensor& _conj_physical__(Tensor& self) {
    ops::copy_(self, ops::conj_physical(self));
    return self;
}

// ---- copy: the functional form of copy_ -- self's metadata, src's values
Tensor copy_impl(const Tensor& self, const Tensor& src, bool non_blocking) {
    const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    // Without storage there is nothing to clone; copy_ overwrites every
    // element anyway.
    Tensor r = self.impl()->storage().nbytes() == 0
        ? ops::empty_strided(sizes, self.strides(), self.dtype(), self.device())
        : ops::clone(self, /*Preserve=*/int64_t(1));
    r.copy_(src, non_blocking);
    return r;
}

Tensor _copy_from_impl(const Tensor& src, const Tensor& dst) {
    TP_THROW(RuntimeError,
             "_copy_from is an internal autograd-view helper and should not be called directly");
}

Tensor _copy_from_and_resize_impl(const Tensor& src, const Tensor& dst) {
    TP_THROW(RuntimeError,
             "_copy_from_and_resize is an internal resize-view helper and should not be called directly");
}

// ---- contiguous: self when already laid out in memory_format, else a copy
Tensor contiguous_format(const Tensor& self, int64_t memory_format) {
    if (memory_format == 1) {  // Preserve
        TP_THROW(RuntimeError,
                 "preserve memory format is unsupported by the contiguous operator");
    }
    if (self.is_contiguous(static_cast<MemoryFormat>(memory_format))) {
        return self;
    }
    return ops::clone(self, memory_format);
}

// ---- chalf: reinterpret/cast to the complex half dtype
Tensor chalf_impl(const Tensor& self, std::optional<int64_t> memory_format) {
    return ops::to(self, DType::ComplexHalf, false, false, memory_format);
}

// ---- _shape_as_tensor: sizes as a CPU int64 tensor
Tensor _shape_as_tensor_impl(const Tensor& self) {
    const IntArrayRef sizes = self.sizes();
    // One 1-D buffer of int64 holds the extents.  The answer is a fresh
    // allocator block every call, so the values are copied in as one run
    // rather than element by element.
    static const Device cpu_device(DeviceType::CPU);
    Tensor shape({static_cast<int64_t>(sizes.size())}, DType::Int64, cpu_device);
    if (!sizes.empty()) {
        std::memcpy(shape.data_ptr<int64_t>(), sizes.data(),
                    sizes.size() * sizeof(int64_t));
    }
    return shape;
}

Tensor _dim_arange_impl(const Tensor& like, int64_t dim) {
    return ops::arange(Scalar(like.size(dim)), DType::Int64, like.device());
}

// ---- _masked_scale: the kept entries scaled, the dropped ones zero.
Tensor _masked_scale_impl(const Tensor& self, const Tensor& mask, double scale) {
    return ops::mul(self, ops::where(mask.to(DType::Bool),
                                     ops::full({}, Scalar(scale), self.dtype(), self.device()),
                                     ops::full({}, Scalar(0.0), self.dtype(), self.device())));
}

// ---- _mkldnn_transpose / _to_sparse bridges route to the registered kernels
// Layout codes: 0 COO, 1 CSR, 2 CSC, 3 BSR, 4 BSC.
Tensor _to_sparse_impl(const Tensor& self, std::optional<int64_t> layout,
                       const std::optional<std::vector<int64_t>>& blocksize,
                       std::optional<int64_t> dense_dim) {
    const int64_t layout_to = layout.value_or(0);
    if ((layout_to == 3 || layout_to == 4) && !blocksize.has_value()) {
        TP_THROW(RuntimeError, "_to_sparse: blocksize is required for blocked sparse layouts");
    }
    if (layout_to < 3 && blocksize.has_value()) {
        TP_THROW(RuntimeError, "_to_sparse: blocksize is only supported for blocked sparse layouts");
    }
    switch (layout_to) {
        case 0: return ops::to_sparse(self, self.dim() - dense_dim.value_or(0));
        case 1:
            if (dense_dim.value_or(0) != 0) {
                TP_THROW(NotImplementedError, "_to_sparse: CSR conversion with dense dimensions");
            }
            return ops::to_sparse_csr(self);
        case 2: return ops::to_sparse_csc(self, dense_dim);
        case 3: return ops::to_sparse_bsr(self, *blocksize, dense_dim);
        case 4: return ops::to_sparse_bsc(self, *blocksize, dense_dim);
        default: break;
    }
    TP_THROW(RuntimeError, "_to_sparse: conversion to layout ", layout_to, " not supported");
}

Tensor _to_sparse_csr_impl(const Tensor& self,
                           std::optional<int64_t> dense_dim) {
    if (dense_dim.value_or(0) != 0) {
        TP_THROW(NotImplementedError,
                 "_to_sparse_csr: dense dimensions are not supported");
    }
    return ops::to_sparse_csr(self);
}

Tensor encoder_layer_norm(const Tensor& x, int64_t embed_dim, double eps,
                          const Tensor& weight, const Tensor& bias) {
    return std::get<0>(ops::native_layer_norm(
        x, {embed_dim}, std::optional<Tensor>(weight),
        std::optional<Tensor>(bias), eps));
}

Tensor encoder_ffn(const Tensor& input, const Tensor& w1, const Tensor& b1,
                   const Tensor& w2, const Tensor& b2, bool use_gelu) {
    Tensor flat = input.reshape({input.size(0) * input.size(1), input.size(2)});
    Tensor hidden = ops::addmm(b1, flat, ops::transpose(w1, 0, 1), Scalar(1),
                               Scalar(1));
    if (use_gelu) {
        hidden = ops::gelu(hidden);
    }
    hidden = hidden.reshape({input.size(0), input.size(1), -1});
    Tensor flat_out =
        hidden.reshape({hidden.size(0) * hidden.size(1), hidden.size(2)});
    Tensor out = ops::addmm(b2, flat_out, ops::transpose(w2, 0, 1), Scalar(1),
                            Scalar(1));
    return out.reshape({input.size(0), input.size(1), -1});
}

Tensor transformer_encoder_layer_fwd_impl(
    const Tensor& src, int64_t embed_dim, int64_t num_heads,
    const Tensor& qkv_weight, const Tensor& qkv_bias,
    const Tensor& proj_weight, const Tensor& proj_bias, bool use_gelu,
    bool norm_first, double eps, const Tensor& norm_weight_1,
    const Tensor& norm_bias_1, const Tensor& norm_weight_2,
    const Tensor& norm_bias_2, const Tensor& ffn_weight_1,
    const Tensor& ffn_bias_1, const Tensor& ffn_weight_2,
    const Tensor& ffn_bias_2, std::optional<Tensor> mask,
    std::optional<int64_t> mask_type) {
    if (src.numel() == 0) {
        return ops::clone(src);
    }
    Tensor x = src;
    if (norm_first) {
        x = encoder_layer_norm(x, embed_dim, eps, norm_weight_1, norm_bias_1);
    }
    x = std::get<0>(ops::_native_multi_head_attention(
        x, x, x, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight,
        proj_bias, mask, false, true, mask_type));
    x = ops::add(x, src);
    if (!norm_first) {
        x = encoder_layer_norm(x, embed_dim, eps, norm_weight_1, norm_bias_1);
    }

    const Tensor pre_ffn = x;
    if (norm_first) {
        x = encoder_layer_norm(x, embed_dim, eps, norm_weight_2, norm_bias_2);
    }
    x = encoder_ffn(x, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2,
                    use_gelu);
    x = ops::add(x, pre_ffn);
    if (!norm_first) {
        x = encoder_layer_norm(x, embed_dim, eps, norm_weight_2, norm_bias_2);
    }
    return x;
}
TENSORPLAY_LIBRARY_IMPL(Composite, MiscBridgeComposites) {
    m.impl("copysign_.Scalar", copysign__scalar);
    m.impl("copysign.Scalar_out", copysign_scalar_out);
    m.impl("clamp.Tensor", clamp_tensor);
    m.impl("clamp_.Tensor", clamp__tensor);
    m.impl("clip.Tensor", clip_tensor);
    m.impl("clip_.Tensor", clip__tensor);
    m.impl("clip.out", clip_scalar_out);
    m.impl("_conj", _conj_view);
    m.impl("_neg_view", _neg_view);
    m.impl("_conj_physical", _conj_physical_impl);
    m.impl("_conj_physical_", _conj_physical__);
    m.impl("copy", copy_impl);
    m.impl("contiguous", contiguous_format);
    m.impl("chalf", chalf_impl);
    m.impl("_shape_as_tensor", _shape_as_tensor_impl);
    m.impl("_dim_arange", _dim_arange_impl);
    m.impl("_masked_scale", _masked_scale_impl);
    m.impl("_to_sparse", _to_sparse_impl);
    m.impl("_to_sparse_csr", _to_sparse_csr_impl);
}

TENSORPLAY_LIBRARY_IMPL(CPU, TransformerEncoderCPU) {
    m.impl("_transformer_encoder_layer_fwd", transformer_encoder_layer_fwd_impl);
}

} // namespace composite
} // namespace tensorplay
