// CUDA registrations for operator spellings whose semantics are already
// served by kernels in this backend.  Two groups live here:
//
// 1. Alias spellings: internal names for ops tp implements under its
//    public name (the argument lists match, so the kernel is reused as-is).
// 2. Helper spellings: decomposition-level operators composed here from the
//    dispatched primitives, so the CUDA key holds a direct registration and
//    the dispatcher's composite fallthrough stays a fallback rather than the
//    only path.
//
// Spellings whose semantics tp does not provide (sparse-only helpers) get an
// explicit registration that reports the missing backend instead of falling
// through to a misleading dense result.
//
// Every kernel's signature must match the generated dispatcher stub for its
// spelling byte for byte: registrations are stored as type-erased function
// pointers, so a mismatch is silent undefined behaviour at call time.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "Generator.h"
#include "CUDARuntime.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;

// Kernels living in other translation units' anonymous namespaces are
// reached through their registered public operators.
template <typename Return, typename... Args>
Return dispatch_cuda(const char* op, Args... args) {
    return DispatchStub<Return, Args...>::call(
        std::string(op), DispatchKey::CUDA, std::forward<Args>(args)...);
}

// Used by the tril/triu index builders before its definition below.
Tensor repeat_interleave_tensor_cuda(const Tensor& repeats,
                                     std::optional<int64_t> output_size);

namespace {

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

void histc_expand_constant_range(DType dtype, double& lo, double& hi) {
    switch (dtype) {
        case DType::Float64:
            lo = std::min(
                std::nexttoward(lo, std::numeric_limits<double>::lowest()),
                lo - 1.0);
            hi = std::max(
                std::nexttoward(hi, std::numeric_limits<double>::max()),
                hi + 1.0);
            break;
        case DType::Float32:
            lo = std::min(
                static_cast<double>(std::nexttoward(
                    static_cast<float>(lo),
                    std::numeric_limits<float>::lowest())),
                lo - 1.0);
            hi = std::max(
                static_cast<double>(std::nexttoward(
                    static_cast<float>(hi),
                    std::numeric_limits<float>::max())),
                hi + 1.0);
            break;
        default:
            lo -= 1.0;
            hi += 1.0;
            break;
    }
}

// Channel statistics for batch normalization: mean/variance over every
// dimension except the channel axis, collapsed to one value per channel.
std::tuple<Tensor, Tensor> batch_norm_channel_stats(const Tensor& input) {
    const int64_t C = input.size(1);
    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < input.dim(); ++d) {
        if (d != 1) reduce_dims.push_back(d);
    }
    Tensor mean = ops::mean(input, reduce_dims, true).reshape({C});
    Tensor var = ops::var(input, reduce_dims, 0, true).reshape({C});
    return std::make_tuple(mean, var);
}

// Broadcast a per-channel (C,) parameter into input's layout: channel values
// line up with dim 1, remaining dims broadcast.
Tensor expand_channel_param(const Tensor& param, const Tensor& like) {
    std::vector<int64_t> sizes(static_cast<size_t>(like.dim()), 1);
    sizes[1] = like.size(1);
    return param.reshape(sizes);
}

// Forward pass shared by the native_batch_norm spellings.  The public
// batch_norm kernel updates the caller's running buffers in place during
// training; when no buffers are supplied it works on internal scratch and
// leaves the (optional) statistics untouched.
std::tuple<Tensor, Tensor, Tensor> batch_norm_forward_impl(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, bool training,
        double momentum, double eps) {
    Tensor out = ops::batch_norm(input, weight, bias, running_mean,
                                 running_var, training, momentum, eps);
    if (!training) {
        // Eval mode has no batch statistics to save.
        return std::make_tuple(out, Tensor(), Tensor());
    }
    Tensor mean;
    Tensor var;
    std::tie(mean, var) = batch_norm_channel_stats(input);
    Tensor invstd = ops::rsqrt(var + Scalar(eps));
    return std::make_tuple(out, mean, invstd);
}

}  // namespace

// ---------------------------------------------------------------------------
// histc: bin edges are equally spaced over [min, max]; values outside the
// range are dropped, the rightmost edge is inclusive.
// ---------------------------------------------------------------------------

Tensor interop_histc_cuda(const Tensor& self, int64_t bins, Scalar min, Scalar max) {
    if (bins <= 0) TP_THROW(RuntimeError, "histc(): bins must be positive");
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError, "histc(): expected a floating-point tensor, got ",
                 toString(self.dtype()));
    }
    double lo = min.toDouble();
    double hi = max.toDouble();
    if (lo == hi && self.numel() > 0) {
        auto extrema = ops::aminmax(self);
        lo = std::get<0>(extrema).item().toDouble();
        hi = std::get<1>(extrema).item().toDouble();
    }
    if (lo == hi) {
        histc_expand_constant_range(self.dtype(), lo, hi);
        histc_expand_constant_range(self.dtype(), lo, hi);
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        TP_THROW(RuntimeError, "histc: range of [", lo, ", ", hi,
                 "] is not finite");
    }
    if (!(lo < hi)) TP_THROW(RuntimeError, "histc: max must be larger than min");
    const Tensor flat = ops::reshape(self, {-1});
    const Tensor in_range = ops::logical_and(ops::ge(flat, Scalar(lo)),
                                             ops::le(flat, Scalar(hi)));
    const Tensor safe = Tensor::where(in_range, flat, Tensor::zeros_like(flat));
    Tensor idx = ops::div(
        ops::mul(ops::sub(safe, Scalar(lo)), Scalar(bins)),
        Scalar(hi - lo)).to(DType::Int64);
    idx = ops::clamp(idx, Scalar(int64_t(0)), Scalar(bins - 1));
    const Tensor counted = ops::masked_select(idx, in_range);
    return ops::bincount(counted, std::nullopt, bins).to(self.dtype());
}

Tensor& interop_histc_out_cuda(const Tensor& self, int64_t bins, Scalar min,
                               Scalar max, Tensor& out) {
    if (out.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "histc(): out tensor must have the same dtype as the input");
    }
    if (out.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "histc(): out tensor must be on the same device as the input");
    }
    Tensor r = interop_histc_cuda(self, bins, min, max);
    out.resize_(static_cast<std::vector<int64_t>>(r.shape()));
    out.copy_(r);
    return out;
}

// ---------------------------------------------------------------------------
// tril_indices / triu_indices: enumerate cells below/above the offset
// diagonal.  Per-row cell counts are clamped to the grid; each selected row
// owns a contiguous run of flat positions, so the column of position p is
// p - (row start) (+ the leading offset for triu).
// ---------------------------------------------------------------------------

namespace {

Tensor tri_diag_base(int64_t row, int64_t col, int64_t offset, bool lower,
                     const Device& dev) {
    // Row r contributes clamp(r + offset + 1, 0, col) cells below the
    // diagonal, clamp(col - max(r + offset, 0), 0, col) cells above it.
    Tensor lead = ops::add(ops::arange(Scalar(int64_t(0)), Scalar(row),
                                       Scalar(int64_t(1)), DType::Int64, dev),
                          Scalar(offset));
    Tensor counts = lower
        ? ops::clamp(ops::add(lead, Scalar(int64_t(1))),
                     Scalar(int64_t(0)), Scalar(col))
        : ops::clamp(Scalar(col) - ops::clamp(lead, Scalar(int64_t(0)),
                                              Scalar(col)),
                     Scalar(int64_t(0)), Scalar(col));
    return counts;
}

}  // namespace

Tensor interop_tril_indices_cuda(int64_t row, int64_t col, int64_t offset,
                                 DType dtype, std::optional<Device> device,
                                 bool pin_memory) {
    const Device dev = device.value_or(Device(DeviceType::CUDA));
    Tensor counts = tri_diag_base(row, col, offset, /*lower=*/true, dev);
    const int64_t count =
        static_cast<int64_t>(counts.sum().item().to<int64_t>());
    Tensor rows = repeat_interleave_tensor_cuda(counts, count);
    Tensor starts = ops::sub(ops::cumsum(counts, 0), counts);
    Tensor flat = ops::arange(Scalar(int64_t(0)), Scalar(count),
                              Scalar(int64_t(1)), DType::Int64, dev);
    Tensor cols = ops::sub(flat, ops::index_select(starts, 0, rows));
    Tensor result = ops::empty({2, count}, dtype, dev, pin_memory);
    result.select(0, 0).copy_(rows.to(dtype));
    result.select(0, 1).copy_(cols.to(dtype));
    return result;
}

Tensor interop_triu_indices_cuda(int64_t row, int64_t col, int64_t offset,
                                 DType dtype, std::optional<Device> device,
                                 bool pin_memory) {
    const Device dev = device.value_or(Device(DeviceType::CUDA));
    Tensor counts = tri_diag_base(row, col, offset, /*lower=*/false, dev);
    const int64_t count =
        static_cast<int64_t>(counts.sum().item().to<int64_t>());
    Tensor rows = repeat_interleave_tensor_cuda(counts, count);
    Tensor starts = ops::sub(ops::cumsum(counts, 0), counts);
    Tensor flat = ops::arange(Scalar(int64_t(0)), Scalar(count),
                              Scalar(int64_t(1)), DType::Int64, dev);
    // The first cell of row r sits at column max(r + offset, 0).
    Tensor first_col = ops::clamp(
        ops::add(ops::arange(Scalar(int64_t(0)), Scalar(row),
                             Scalar(int64_t(1)), DType::Int64, dev),
                 Scalar(offset)),
        Scalar(int64_t(0)), Scalar(col));
    Tensor cols = ops::add(ops::index_select(first_col, 0, rows),
                           ops::sub(flat, ops::index_select(starts, 0, rows)));
    Tensor result = ops::empty({2, count}, dtype, dev, pin_memory);
    result.select(0, 0).copy_(rows.to(dtype));
    result.select(0, 1).copy_(cols.to(dtype));
    return result;
}

// ---------------------------------------------------------------------------
// unique_consecutive: run-length encoding of adjacent equal values.  With a
// dimension given, the scan runs along that dimension; without one, the
// input is flattened first (so inverse indices index the flat input).
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor> interop_unique_consecutive_cuda(
        const Tensor& self, bool return_inverse, bool return_counts,
        std::optional<int64_t> dim) {
    Tensor flat = dim.has_value() ? self : self.reshape({-1});
    auto result = dispatch_cuda<std::tuple<Tensor, Tensor, Tensor>>(
        "unique_dim_consecutive", flat, dim.has_value() ? *dim : int64_t(0),
        return_inverse, return_counts);
    if (!dim.has_value() && return_inverse) {
        std::get<1>(result) = std::get<1>(result).reshape(
            static_cast<std::vector<int64_t>>(self.shape()));
    }
    return result;
}

// ---------------------------------------------------------------------------
// nonzero_static: first `size` nonzero coordinates, padded with fill_value
// rows beyond the actual count.
// ---------------------------------------------------------------------------

Tensor interop_nonzero_static_cuda(const Tensor& self,
                                   std::optional<int64_t> size,
                                   int64_t fill_value) {
    Tensor nz = ops::nonzero(self);
    const int64_t ndim = self.dim();
    const int64_t cap = size.value_or(nz.size(0));
    Tensor result = ops::full({cap, ndim}, Scalar(fill_value), DType::Int64,
                              self.device());
    const int64_t copy_n = std::min<int64_t>(cap, nz.size(0));
    if (copy_n > 0) {
        result.narrow(0, 0, copy_n).copy_(nz.narrow(0, 0, copy_n));
    }
    return result;
}

Tensor& interop_nonzero_static_out_cuda(const Tensor& self, int64_t size,
                                        int64_t fill_value, Tensor& out) {
    Tensor nz = ops::nonzero(self);
    const int64_t cap = size;
    Tensor result = ops::full({cap, self.dim()}, Scalar(fill_value),
                              DType::Int64, self.device());
    const int64_t copy_n = std::min<int64_t>(cap, nz.size(0));
    if (copy_n > 0) {
        result.narrow(0, 0, copy_n).copy_(nz.narrow(0, 0, copy_n));
    }
    out = std::move(result);
    return out;
}

// ---------------------------------------------------------------------------
// repeat_interleave.Tensor returns the flat source-index list
// [0 x r0, 1 x r1, ...] from cumulative repeat boundaries.
// ---------------------------------------------------------------------------

Tensor repeat_interleave_tensor_cuda(const Tensor& repeats,
                                     std::optional<int64_t> output_size) {
    TP_CHECK(repeats.dim() == 1,
             "repeat_interleave: repeats must be 1-dimensional");
    TP_CHECK(repeats.dtype() == DType::Int32 ||
                 repeats.dtype() == DType::Int64,
             "repeats must have Int32 or Int64 dtype");
    if (repeats.numel() == 0) {
        return Tensor::empty({0}, repeats.dtype(), repeats.device());
    }

    Tensor rep = repeats.contiguous();
    Tensor ends = ops::cumsum(rep, 0, DType::Int64);
    const int64_t required_size =
        ends.select(0, ends.size(0) - 1).item<int64_t>();
    const int64_t total = output_size.value_or(required_size);
    TP_CHECK(total == required_size,
             "allocated size does not match required size");
    TP_CHECK(rep.ge(Scalar(0)).all().item<bool>(),
             "repeats can not be negative");

    Tensor ar = ops::arange(Scalar(int64_t(0)), Scalar(total),
                            Scalar(int64_t(1)), DType::Int64, repeats.device());
    Tensor indices = ops::searchsorted(ends, ar, false, true);
    return repeats.dtype() == DType::Int32
        ? indices.to(DType::Int32)
        : indices;
}

Tensor interop_repeat_interleave_Tensor_cuda(
        const Tensor& repeats, std::optional<int64_t> output_size) {
    return repeat_interleave_tensor_cuda(repeats, output_size);
}

// ---------------------------------------------------------------------------
// Renormalize referenced rows whose selected norm exceeds max_norm.
// ---------------------------------------------------------------------------

Tensor& interop_embedding_renorm__cuda(Tensor& weight, const Tensor& indices,
                                       double max_norm, double norm_type) {
    TP_CHECK(weight.dim() == 2,
             "embedding_renorm_: weight must be 2-D");
    TP_CHECK(indices.dtype() == DType::Int64 || indices.dtype() == DType::Int32,
             "embedding_renorm_: indices must be Int64 or Int32");
    TP_CHECK(weight.device() == indices.device(),
             "embedding_renorm_: weight and indices must be on the same device");
    TP_CHECK(max_norm > 0.0,
             "embedding_renorm_: max_norm must be positive");
    TP_CHECK(norm_type > 0.0,
             "embedding_renorm_: norm_type must be positive");

    const int64_t d = 1;
    Tensor norms = ops::norm(weight, {d}, norm_type, true);
    Tensor idx_flat = indices.reshape({-1}).to(DType::Int64);
    Tensor row_norms = ops::index_select(norms, 0, idx_flat);
    Tensor scale = ops::reciprocal(ops::add(row_norms, Scalar(1e-7)))
                      .mul(Scalar(max_norm))
                      .clamp_max(Scalar(1.0));
    // Scale every selected row whose norm exceeds max_norm.
    Tensor w_sel = ops::index_select(weight, 0, idx_flat);
    Tensor scaled = ops::mul(w_sel, scale);
    Tensor updated = ops::where(ops::gt(row_norms, Scalar(max_norm)), scaled,
                                w_sel);
    weight.index_copy_(0, idx_flat, updated);
    return weight;
}

// ---------------------------------------------------------------------------
// sspaddmm: sparse-only in the reference; tp has no sparse CUDA backend, so
// the contract is an explicit rejection rather than a dense reinterpretation.
// ---------------------------------------------------------------------------

Tensor interop_sspaddmm_cuda(const Tensor& /*self*/, const Tensor& mat1,
                             const Tensor& /*mat2*/, Scalar /*beta*/,
                             Scalar /*alpha*/) {
    TP_THROW(NotImplementedError,
             "sspaddmm requires a sparse CUDA backend; mat1 is dense with dtype ",
             toString(mat1.dtype()));
}

Tensor& interop_sspaddmm_out_cuda(const Tensor& self, const Tensor& mat1,
                                  const Tensor& mat2, Scalar beta, Scalar alpha,
                                  Tensor& out) {
    out = interop_sspaddmm_cuda(self, mat1, mat2, beta, alpha);
    return out;
}

// ---------------------------------------------------------------------------
// _masked_scale: scale the masked-in entries by `scale`.
// ---------------------------------------------------------------------------

Tensor interop_masked_scale_cuda(const Tensor& self, const Tensor& mask,
                                 double scale) {
    return ops::where(mask, self * Scalar(scale), self);
}

// ---------------------------------------------------------------------------
// _masked_softmax: softmax over entries where mask is true; masked-out
// positions stay zero.  The reduction runs in float32 for reduced widths.
// ---------------------------------------------------------------------------

Tensor interop_masked_softmax_cuda(const Tensor& self, const Tensor& mask,
                                   std::optional<int64_t> dim,
                                   std::optional<int64_t> mask_type) {
    (void)mask_type;
    const int64_t d = dim.has_value() ? *dim : -1;
    Tensor neg_inf =
        ops::full_like(self, Scalar(-std::numeric_limits<double>::infinity()));
    Tensor masked = ops::where(mask, self, neg_inf);
    Tensor out = ops::softmax(masked, d, DType::Undefined);
    return ops::where(mask, out, ops::zeros_like(self));
}

Tensor interop_masked_softmax_backward_cuda(const Tensor& grad_output,
                                            const Tensor& output,
                                            const Tensor& mask,
                                            std::optional<int64_t> dim) {
    const int64_t d = dim.has_value() ? *dim : -1;
    Tensor g = ops::where(mask, grad_output, ops::zeros_like(grad_output));
    Tensor o = ops::where(mask, output, ops::zeros_like(output));
    Tensor dot = ops::sum(ops::mul(g, o), {d}, true);
    return ops::where(mask, ops::mul(o, ops::sub(g, dot)),
                      ops::zeros_like(grad_output));
}

// ---------------------------------------------------------------------------
// _fused_rms_norm: x / sqrt(mean(x^2) + eps) * weight.  The second output is
// the reciprocal standard deviation, saved for the backward pass.
// ---------------------------------------------------------------------------

std::vector<int64_t> norm_trailing_dims(const Tensor& input,
                                        const std::vector<int64_t>& normalized_shape) {
    std::vector<int64_t> dims;
    for (int64_t i = static_cast<int64_t>(input.dim()) -
                     static_cast<int64_t>(normalized_shape.size());
         i < input.dim(); ++i) {
        dims.push_back(i);
    }
    return dims;
}

std::tuple<Tensor, Tensor> interop_fused_rms_norm_cuda(
        const Tensor& input, const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight_opt,
        std::optional<double> eps_opt) {
    const double eps = eps_opt.value_or(1e-5);
    Tensor weight = weight_opt.value_or(Tensor());
    std::vector<int64_t> dims = norm_trailing_dims(input, normalized_shape);
    Tensor ms = ops::mean(ops::pow(input, Scalar(2.0)), dims, true);
    Tensor inv = ops::rsqrt(ops::add(ms, Scalar(eps)));
    Tensor out = ops::mul(input, inv);
    if (weight.defined()) out = ops::mul(out, weight);
    return std::make_tuple(out, inv);
}

std::tuple<Tensor, Tensor> interop_fused_rms_norm_backward_cuda(
        const Tensor& grad_out, const Tensor& input,
        const std::vector<int64_t>& normalized_shape, const Tensor& rstd,
        const std::optional<Tensor>& weight_opt,
        const std::vector<bool>& output_mask) {
    Tensor weight = weight_opt.value_or(Tensor());
    std::vector<int64_t> dims = norm_trailing_dims(input, normalized_shape);
    Tensor xhat = ops::mul(input, rstd);
    // d/dx [x * inv * w] with inv treated as constant to first order:
    // inv * (g_eff - xhat * mean(g_eff * xhat)) where g_eff folds in w.
    Tensor gw = weight.defined() ? ops::mul(grad_out, weight) : grad_out;
    Tensor dot = ops::mean(ops::mul(gw, xhat), dims, true);
    Tensor grad_input;
    if (output_mask.empty() || output_mask[0]) {
        grad_input = ops::mul(rstd, ops::sub(gw, ops::mul(xhat, dot)));
    } else {
        grad_input = Tensor();
    }
    Tensor grad_weight;
    if (output_mask.size() > 1 && output_mask[1] && weight.defined()) {
        // Sum of g * xhat per normalized slice; the reduced axes collapse to
        // the weight's own (possibly size-1) layout.
        Tensor full = ops::sum(ops::mul(grad_out, xhat), dims, true);
        grad_weight = full.reshape(
            static_cast<std::vector<int64_t>>(weight.shape()));
    } else {
        grad_weight = Tensor();
    }
    return std::make_tuple(grad_input, grad_weight);
}

// ---------------------------------------------------------------------------
// _chunk_cat: split the list into num_chunks even parts along dim, then cat.
// ---------------------------------------------------------------------------

Tensor interop_chunk_cat_cuda(const std::vector<Tensor>& tensors, int64_t dim,
                              int64_t num_chunks) {
    if (tensors.empty()) {
        TP_THROW(RuntimeError, "expected a non-empty list of Tensors");
    }
    if (num_chunks <= 0) {
        TP_THROW(RuntimeError, "num_chunks must be positive, got ", num_chunks);
    }
    std::vector<Tensor> parts;
    parts.reserve(static_cast<size_t>(num_chunks));
    const int64_t per = (static_cast<int64_t>(tensors.size()) + num_chunks - 1) /
                        num_chunks;
    for (int64_t c = 0; c < num_chunks; ++c) {
        const int64_t begin = c * per;
        const int64_t end = std::min<int64_t>(begin + per, tensors.size());
        if (begin >= end) break;
        parts.push_back(ops::cat(std::vector<Tensor>(tensors.begin() + begin,
                                                     tensors.begin() + end),
                                 dim));
    }
    return ops::cat(parts, dim);
}

Tensor& interop__chunk_cat_out_cuda(const std::vector<Tensor>& tensors,
                                    int64_t dim, int64_t num_chunks,
                                    Tensor& out) {
    out = interop_chunk_cat_cuda(tensors, dim, num_chunks);
    return out;
}

// ---------------------------------------------------------------------------
// _fused_dropout: dropout with its boolean mask output.  tp's dropout RNG
// always draws from the default generator stream, so the generator argument
// is accepted but does not reroute the draws.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> interop__fused_dropout_cuda(
        const Tensor& self, double p, std::optional<Generator> generator) {
    (void)generator;
    return dispatch_cuda<std::tuple<Tensor, Tensor>>("native_dropout", self, p);
}

// ---------------------------------------------------------------------------
// _local_scalar_dense: read a 0-dim value off the device into a Scalar.
// ---------------------------------------------------------------------------

Scalar interop__local_scalar_dense_cuda(const Tensor& self) {
    TP_CHECK(self.numel() == 1,
             "_local_scalar_dense only supports 1-element tensors, got ",
             self.numel());
    Tensor host = self.reshape({1}).contiguous().to(Device(DeviceType::CPU));
    switch (self.dtype()) {
        case DType::Float32: return Scalar(host.data_ptr<float>()[0]);
        case DType::Float64: return Scalar(host.data_ptr<double>()[0]);
        case DType::Int64: return Scalar(host.data_ptr<int64_t>()[0]);
        case DType::Int32: return Scalar(static_cast<int64_t>(host.data_ptr<int32_t>()[0]));
        case DType::Float16: return Scalar(static_cast<double>(host.data_ptr<Half>()[0]));
        case DType::BFloat16: return Scalar(static_cast<double>(host.data_ptr<BFloat16>()[0]));
        case DType::Bool: return Scalar(host.data_ptr<bool>()[0]);
        case DType::Int16: return Scalar(static_cast<int64_t>(host.data_ptr<int16_t>()[0]));
        case DType::Int8: return Scalar(static_cast<int64_t>(host.data_ptr<int8_t>()[0]));
        case DType::UInt8: return Scalar(static_cast<int64_t>(host.data_ptr<uint8_t>()[0]));
        default:
            TP_THROW(TypeError, "_local_scalar_dense: unsupported dtype ",
                     toString(self.dtype()));
    }
}

// ---------------------------------------------------------------------------
// Depthwise / slow convolution spellings.  tp's conv2d/conv3d kernels take
// groups natively; the depthwise spellings derive groups from the weight.
// ---------------------------------------------------------------------------

Tensor interop__conv_depthwise2d_cuda(const Tensor& self, const Tensor& weight,
                                      const std::vector<int64_t>& kernel_size,
                                      const std::optional<Tensor>& bias,
                                      const std::vector<int64_t>& stride,
                                      const std::vector<int64_t>& padding,
                                      const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return dispatch_cuda<Tensor>("conv2d", self, weight, bias, stride, padding,
                                 dilation, weight.size(0));
}

Tensor& interop__conv_depthwise2d_out_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, Tensor& out) {
    (void)kernel_size;
    out = dispatch_cuda<Tensor>("conv2d", self, weight, bias, stride, padding,
                                dilation, weight.size(0));
    return out;
}

Tensor interop_conv_depthwise3d_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return dispatch_cuda<Tensor>("conv3d", self, weight, bias, stride, padding,
                                 dilation, weight.size(0));
}

Tensor interop__slow_conv2d_forward_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding) {
    (void)kernel_size;
    const std::vector<int64_t> dilation{1, 1};
    return dispatch_cuda<Tensor>("conv2d", self, weight, bias, stride, padding,
                                 dilation, int64_t(1));
}

Tensor& interop__slow_conv2d_forward_output_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding, Tensor& output) {
    (void)kernel_size;
    const std::vector<int64_t> dilation{1, 1};
    output = dispatch_cuda<Tensor>("conv2d", self, weight, bias, stride,
                                   padding, dilation, int64_t(1));
    return output;
}

std::tuple<Tensor, Tensor, Tensor> interop__slow_conv2d_backward_grad_input_cuda(
        const Tensor& grad_output, const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding, Tensor& grad_input,
        Tensor& grad_weight, Tensor& grad_bias) {
    (void)kernel_size;
    const std::vector<int64_t> dilation{1, 1};
    grad_input = dispatch_cuda<Tensor>("conv2d_grad_input", grad_output, self,
                                      weight, stride, padding, dilation,
                                      int64_t(1));
    grad_weight = dispatch_cuda<Tensor>("conv2d_grad_weight", grad_output, self,
                                       weight, stride, padding, dilation,
                                       int64_t(1));
    grad_bias = dispatch_cuda<Tensor>("conv2d_grad_bias", grad_output, self,
                                      weight, stride, padding, dilation,
                                      int64_t(1));
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor>
interop__slow_conv2d_backward_output_mask_cuda(
        const Tensor& grad_output, const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<bool>& output_mask) {
    (void)kernel_size;
    const std::vector<int64_t> dilation{1, 1};
    const bool want_i = output_mask.size() > 0 && output_mask[0];
    const bool want_w = output_mask.size() > 1 && output_mask[1];
    const bool want_b = output_mask.size() > 2 && output_mask[2];
    Tensor gi = want_i ? dispatch_cuda<Tensor>("conv2d_grad_input", grad_output,
                                               self, weight, stride, padding,
                                               dilation, int64_t(1))
                       : Tensor();
    Tensor gw = want_w ? dispatch_cuda<Tensor>("conv2d_grad_weight", grad_output,
                                               self, weight, stride, padding,
                                               dilation, int64_t(1))
                       : Tensor();
    Tensor gb = want_b ? dispatch_cuda<Tensor>("conv2d_grad_bias", grad_output,
                                              self, weight, stride, padding,
                                              dilation, int64_t(1))
                       : Tensor();
    return std::make_tuple(gi, gw, gb);
}

Tensor interop_slow_conv_dilated2d_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return dispatch_cuda<Tensor>("conv2d", self, weight, bias, stride, padding,
                                 dilation, int64_t(1));
}

Tensor interop_slow_conv_dilated3d_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return dispatch_cuda<Tensor>("conv3d", self, weight, bias, stride, padding,
                                 dilation, int64_t(1));
}

Tensor& interop_slow_conv_transpose2d_out_cuda(
        const Tensor& input, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& output_padding,
        const std::vector<int64_t>& dilation, Tensor& out) {
    (void)kernel_size;
    out = dispatch_cuda<Tensor>("conv_transpose2d", input, weight, bias, stride,
                                padding, output_padding, int64_t(1), dilation);
    return out;
}

Tensor interop_slow_conv_transpose3d_cuda(
        const Tensor& input, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& output_padding,
        const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return dispatch_cuda<Tensor>("conv_transpose3d", input, weight, bias,
                                 stride, padding, output_padding, int64_t(1),
                                 dilation);
}

Tensor& interop_slow_conv_transpose3d_out_cuda(
        const Tensor& input, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& output_padding,
        const std::vector<int64_t>& dilation, Tensor& out) {
    (void)kernel_size;
    out = dispatch_cuda<Tensor>("conv_transpose3d", input, weight, bias,
                                stride, padding, output_padding, int64_t(1),
                                dilation);
    return out;
}

// ---------------------------------------------------------------------------
// native_batch_norm family: the public batch_norm kernel plus the saved
// batch statistics (mean, reciprocal standard deviation) for training mode.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor> interop_native_batch_norm_cuda(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, bool training,
        double momentum, double eps) {
    return batch_norm_forward_impl(input, weight, bias, running_mean,
                                   running_var, training, momentum, eps);
}

std::tuple<Tensor, Tensor, Tensor> interop_native_batch_norm_out_cuda(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, bool training,
        double momentum, double eps, Tensor& out, Tensor& save_mean,
        Tensor& save_invstd) {
    std::tie(out, save_mean, save_invstd) = batch_norm_forward_impl(
        input, weight, bias, running_mean, running_var, training, momentum,
        eps);
    return std::make_tuple(out, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> interop__native_batch_norm_legit_cuda(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, Tensor& running_mean,
        Tensor& running_var, bool training, double momentum, double eps) {
    return batch_norm_forward_impl(input, weight, bias, running_mean,
                                   running_var, training, momentum, eps);
}

std::tuple<Tensor, Tensor, Tensor> interop__native_batch_norm_legit_out_cuda(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, Tensor& running_mean,
        Tensor& running_var, bool training, double momentum, double eps,
        Tensor& out, Tensor& save_mean, Tensor& save_invstd) {
    std::tie(out, save_mean, save_invstd) = batch_norm_forward_impl(
        input, weight, bias, running_mean, running_var, training, momentum,
        eps);
    return std::make_tuple(out, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor>
interop__native_batch_norm_legit_no_stats_cuda(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, bool training, double momentum,
        double eps) {
    // No running statistics to maintain: the kernel runs on internal scratch.
    return batch_norm_forward_impl(input, weight, bias, std::nullopt,
                                   std::nullopt, training, momentum, eps);
}

std::tuple<Tensor, Tensor, Tensor>
interop__native_batch_norm_legit_no_stats_out_cuda(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, bool training, double momentum,
        double eps, Tensor& out, Tensor& save_mean, Tensor& save_invstd) {
    std::tie(out, save_mean, save_invstd) = batch_norm_forward_impl(
        input, weight, bias, std::nullopt, std::nullopt, training, momentum,
        eps);
    return std::make_tuple(out, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> interop__batch_norm_with_update_cuda(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, Tensor& running_mean,
        Tensor& running_var, double momentum, double eps) {
    Tensor out;
    Tensor save_mean;
    Tensor save_invstd;
    std::tie(out, save_mean, save_invstd) = batch_norm_forward_impl(
        input, weight, bias, running_mean, running_var, true, momentum, eps);
    // The fourth slot is an opaque reserve buffer nothing consumes here.
    Tensor reserve;
    return std::make_tuple(out, save_mean, save_invstd, reserve);
}

std::tuple<Tensor, Tensor, Tensor> interop__batch_norm_no_update_cuda(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, double momentum,
        double eps) {
    // Normalize with batch statistics but never touch the running buffers:
    // the kernel sees no statistics and works on internal scratch.
    Tensor out;
    Tensor save_mean;
    Tensor save_invstd;
    std::tie(out, save_mean, save_invstd) = batch_norm_forward_impl(
        input, weight, bias, std::nullopt, std::nullopt, true, momentum, eps);
    (void)running_mean;
    (void)running_var;
    return std::make_tuple(out, save_mean, save_invstd);
}

// batch_norm_stats: per-channel mean and reciprocal standard deviation.
std::tuple<Tensor, Tensor> interop_batch_norm_stats_cuda(const Tensor& input,
                                                         double eps) {
    Tensor mean;
    Tensor var;
    std::tie(mean, var) = batch_norm_channel_stats(input);
    return std::make_tuple(mean, ops::rsqrt(ops::add(var, Scalar(eps))));
}

// batch_norm_elemt: apply the affine transform with precomputed statistics.
Tensor interop_batch_norm_elemt_cuda(const Tensor& input,
                                     const std::optional<Tensor>& weight,
                                     const std::optional<Tensor>& bias,
                                     const Tensor& mean, const Tensor& invstd,
                                     double eps) {
    (void)eps;
    Tensor x = ops::mul(ops::sub(input, expand_channel_param(mean, input)),
                        expand_channel_param(invstd, input));
    if (weight.has_value() && weight->defined()) {
        x = ops::mul(x, expand_channel_param(*weight, input));
    }
    if (bias.has_value() && bias->defined()) {
        x = ops::add(x, expand_channel_param(*bias, input));
    }
    return x;
}

Tensor& interop_batch_norm_elemt_out_cuda(const Tensor& input,
                                          const std::optional<Tensor>& weight,
                                          const std::optional<Tensor>& bias,
                                          const Tensor& mean,
                                          const Tensor& invstd, double eps,
                                          Tensor& out) {
    out = interop_batch_norm_elemt_cuda(input, weight, bias, mean, invstd, eps);
    return out;
}

// batch_norm_backward_reduce: per-channel reduction sums consumed by the
// elementwise backward.  The four slots are (sum_dy, sum_dy_xmu, mean,
// count); tp's elementwise backward reads the count from the fourth slot.
std::tuple<Tensor, Tensor, Tensor, Tensor>
interop_batch_norm_backward_reduce_cuda(const Tensor& grad_out,
                                        const Tensor& input,
                                        const Tensor& mean,
                                        const Tensor& /*invstd*/,
                                        const std::optional<Tensor>& weight,
                                        bool input_g, bool weight_g,
                                        bool bias_g) {
    (void)weight_g;
    (void)bias_g;
    const int64_t C = input.size(1);
    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < input.dim(); ++d) {
        if (d != 1) reduce_dims.push_back(d);
    }
    Tensor sum_dy;
    Tensor sum_dy_xmu;
    if (input_g) {
        sum_dy = ops::sum(grad_out, reduce_dims, true).reshape({C});
        sum_dy_xmu = ops::sum(
                         ops::mul(grad_out,
                                  ops::sub(input, expand_channel_param(mean, input))),
                         reduce_dims, true)
                         .reshape({C});
    } else {
        sum_dy = Tensor();
        sum_dy_xmu = Tensor();
    }
    (void)weight;
    const int64_t per_channel = input.numel() / (C == 0 ? 1 : C);
    Tensor count = ops::full({C}, Scalar(static_cast<double>(per_channel)),
                             mean.dtype(), input.device());
    return std::make_tuple(sum_dy, sum_dy_xmu, mean.reshape({C}), count);
}

// batch_norm_backward_elemt: gradient wrt the input given the reductions.
Tensor interop_batch_norm_backward_elemt_cuda(
        const Tensor& grad_out, const Tensor& input, const Tensor& mean,
        const Tensor& invstd, const std::optional<Tensor>& weight,
        const Tensor& sum_dy, const Tensor& sum_dy_xmu, const Tensor& count) {
    Tensor mean_e = expand_channel_param(mean, input);
    Tensor invstd_e = expand_channel_param(invstd, input);
    Tensor sum_dy_e = expand_channel_param(sum_dy, input);
    Tensor sum_dy_xmu_e = expand_channel_param(sum_dy_xmu, input);
    Tensor count_e = expand_channel_param(count, input);
    // dL/dx = invstd * (g - mean(g) - xhat * mean(g * xhat)) with
    // xhat = (x - mean) * invstd and means over each channel's elements.
    Tensor gd = grad_out;
    if (weight.has_value() && weight->defined()) {
        gd = ops::mul(gd, expand_channel_param(*weight, input));
    }
    Tensor m1 = ops::div(sum_dy_e, count_e);
    Tensor m2 = ops::div(sum_dy_xmu_e, count_e);
    Tensor xhat = ops::mul(ops::sub(input, mean_e), invstd_e);
    return ops::mul(invstd_e, ops::sub(gd, ops::add(m1, ops::mul(xhat, m2))));
}

// batch_norm_gather_stats: fold per-batch statistics into the running
// buffers with the unbiased correction, returning the updated buffers.
std::tuple<Tensor, Tensor> interop_batch_norm_gather_stats_cuda(
        const Tensor& /*input*/, const Tensor& mean, const Tensor& invstd,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, double momentum, double eps,
        int64_t count) {
    Tensor save_mean = mean;
    Tensor var = ops::sub(ops::reciprocal(ops::pow(invstd, Scalar(2.0))),
                          Scalar(eps));
    if (running_mean.has_value() && running_mean->defined() &&
        running_var.has_value() && running_var->defined() && count > 1) {
        const double n = static_cast<double>(count);
        Tensor unbiased = ops::mul(var, Scalar(n / (n - 1.0)));
        save_mean = ops::add(ops::mul(*running_mean, Scalar(1.0 - momentum)),
                             ops::mul(mean, Scalar(momentum)));
        Tensor save_var = ops::add(ops::mul(*running_var, Scalar(1.0 - momentum)),
                                   ops::mul(unbiased, Scalar(momentum)));
        return std::make_tuple(save_mean, save_var);
    }
    return std::make_tuple(save_mean, var);
}

std::tuple<Tensor, Tensor> interop_batch_norm_gather_stats_with_counts_cuda(
        const Tensor& input, const Tensor& mean, const Tensor& invstd,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, double momentum, double eps,
        const Tensor& counts) {
    const int64_t count =
        static_cast<int64_t>(counts.sum().item().to<double>());
    return interop_batch_norm_gather_stats_cuda(input, mean, invstd,
                                                running_mean, running_var,
                                                momentum, eps, count);
}

// batch_norm_update_stats: new running statistics from batch moments.
std::tuple<Tensor, Tensor> interop_batch_norm_update_stats_cuda(
        const Tensor& input, const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, double momentum) {
    Tensor mean;
    Tensor var;
    std::tie(mean, var) = batch_norm_channel_stats(input);
    const int64_t n = input.numel() / input.size(1);
    Tensor unbiased = n > 1 ? ops::mul(var, Scalar(static_cast<double>(n) /
                                                   static_cast<double>(n - 1)))
                            : var;
    if (running_mean.has_value() && running_mean->defined()) {
        mean = ops::add(ops::mul(*running_mean, Scalar(1.0 - momentum)),
                        ops::mul(mean, Scalar(momentum)));
    }
    if (running_var.has_value() && running_var->defined()) {
        unbiased = ops::add(ops::mul(*running_var, Scalar(1.0 - momentum)),
                            ops::mul(unbiased, Scalar(momentum)));
    }
    return std::make_tuple(mean, unbiased);
}

// ---------------------------------------------------------------------------
// native_layer_norm: layer_norm output plus the per-row mean and reciprocal
// standard deviation saved for the backward pass.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor> interop_native_layer_norm_cuda(
        const Tensor& input, const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight, const std::optional<Tensor>& bias,
        double eps) {
    Tensor out = dispatch_cuda<Tensor>("layer_norm", input, normalized_shape,
                                       weight, bias, eps);
    std::vector<int64_t> dims = norm_trailing_dims(input, normalized_shape);
    // Rows are the leading (outer) dims collapsed into one axis.
    std::vector<int64_t> outer_sizes;
    for (int64_t i = 0;
         i < static_cast<int64_t>(input.dim()) -
             static_cast<int64_t>(normalized_shape.size());
         ++i) {
        outer_sizes.push_back(input.size(i));
    }
    if (outer_sizes.empty()) outer_sizes.push_back(1);
    Tensor mean = ops::mean(input, dims, true).reshape(outer_sizes);
    Tensor var = ops::var(input, dims, 0, true).reshape(outer_sizes);
    Tensor rstd = ops::rsqrt(ops::add(var, Scalar(eps)));
    return std::make_tuple(out, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> interop_native_layer_norm_backward_cuda(
        const Tensor& grad_out, const Tensor& input,
        const std::vector<int64_t>& normalized_shape, const Tensor& /*mean*/,
        const Tensor& /*rstd*/, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias,
        const std::vector<bool>& output_mask) {
    // tp's layer_norm_backward recomputes the row statistics internally; the
    // saved mean/rstd are accepted to honor the spelling's contract.
    auto grads = dispatch_cuda<std::tuple<Tensor, Tensor, Tensor>>(
        "layer_norm_backward", grad_out, input, normalized_shape, weight, bias,
        0.0);
    Tensor gi = output_mask.size() > 0 && output_mask[0] ? std::get<0>(grads)
                                                        : Tensor();
    Tensor gw = output_mask.size() > 1 && output_mask[1] ? std::get<1>(grads)
                                                        : Tensor();
    Tensor gb = output_mask.size() > 2 && output_mask[2] ? std::get<2>(grads)
                                                        : Tensor();
    return std::make_tuple(gi, gw, gb);
}

// ---------------------------------------------------------------------------
// native_group_norm: group_norm output plus per-(batch, group) statistics.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor> interop_native_group_norm_cuda(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, int64_t N, int64_t C, int64_t HxW,
        int64_t group, double eps) {
    Tensor out = dispatch_cuda<Tensor>("group_norm", input, group, weight,
                                       bias, eps);
    // Rows are (batch, group) pairs, each covering HxW * C/group elements.
    const int64_t group_size = HxW * (C / (group == 0 ? 1 : group));
    Tensor x = input.reshape({N * group, group_size});
    Tensor mean = ops::mean(x, {1}).reshape({N, group});
    Tensor var = ops::var(x, {1}, 0, false).reshape({N, group});
    Tensor rstd = ops::rsqrt(ops::add(var, Scalar(eps)));
    return std::make_tuple(out, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> interop_native_group_norm_backward_cuda(
        const Tensor& grad_out, const Tensor& input, const Tensor& /*mean*/,
        const Tensor& /*rstd*/, const std::optional<Tensor>& weight, int64_t N,
        int64_t C, int64_t HxW, int64_t group,
        const std::vector<bool>& output_mask) {
    (void)N;
    (void)HxW;
    // tp's group_norm_backward recomputes the group statistics internally;
    // the saved mean/rstd are accepted to honor the spelling's contract.
    auto grads = dispatch_cuda<std::tuple<Tensor, Tensor, Tensor>>(
        "group_norm_backward", grad_out, input, group, weight,
        std::optional<Tensor>(), 0.0);
    Tensor gi = output_mask.size() > 0 && output_mask[0] ? std::get<0>(grads)
                                                        : Tensor();
    Tensor gw = output_mask.size() > 1 && output_mask[1] ? std::get<1>(grads)
                                                        : Tensor();
    Tensor gb = output_mask.size() > 2 && output_mask[2] ? std::get<2>(grads)
                                                        : Tensor();
    return std::make_tuple(gi, gw, gb);
}

// ---------------------------------------------------------------------------
// _fft_r2c / _fft_c2r / _fft_c2c: single-dimension transform spellings of
// the public fft kernels, which always transform the last dimension, so the
// requested dimension is permuted there first and restored afterwards.
// ---------------------------------------------------------------------------

namespace {

Tensor fft_move_dim_last(const Tensor& self, int64_t dim) {
    const int64_t nd = self.dim();
    dim = (dim % nd + nd) % nd;
    std::vector<int64_t> perm;
    perm.reserve(static_cast<size_t>(nd));
    for (int64_t d = 0; d < nd; ++d) {
        if (d != dim) perm.push_back(d);
    }
    perm.push_back(dim);
    return self.permute(perm).contiguous();
}

Tensor fft_move_dim_back(const Tensor& out, int64_t orig_dim, int64_t nd) {
    orig_dim = (orig_dim % nd + nd) % nd;
    // Undo "move to last": the transformed axis sits at the end and returns
    // to its original position; every axis after it shifts up one slot.
    std::vector<int64_t> perm;
    perm.reserve(static_cast<size_t>(nd));
    for (int64_t a = 0; a < nd; ++a) {
        if (a < orig_dim) {
            perm.push_back(a);
        } else if (a == orig_dim) {
            perm.push_back(nd - 1);
        } else {
            perm.push_back(a - 1);
        }
    }
    return out.permute(perm).contiguous();
}

const char* fft_norm_name(int64_t normalization) {
    // 0 = backward (no scaling), 1 = forward (1/n), 2 = ortho (1/sqrt(n)).
    return normalization == 1 ? "forward" : normalization == 2 ? "ortho"
                                                               : "backward";
}

}  // namespace

Tensor interop__fft_r2c_cuda(const Tensor& self,
                            const std::vector<int64_t>& dim,
                            int64_t normalization, bool onesided) {
    TP_CHECK(dim.size() == 1,
             "_fft_r2c transforms exactly one dimension, got ", dim.size());
    const int64_t d = dim[0];
    Tensor moved = fft_move_dim_last(self, d);
    const char* norm = fft_norm_name(normalization);
    Tensor out;
    if (onesided) {
        out = dispatch_cuda<Tensor>("fft_rfft", moved, int64_t(-1),
                                    moved.dim() - 1, std::string(norm));
    } else {
        // A real input's full spectrum is the plain c2c transform.
        out = dispatch_cuda<Tensor>("fft_fft", moved, int64_t(-1),
                                    moved.dim() - 1, std::string(norm));
    }
    return fft_move_dim_back(out, d, self.dim());
}

Tensor& interop__fft_r2c_out_cuda(const Tensor& self,
                                 const std::vector<int64_t>& dim,
                                 int64_t normalization, bool onesided,
                                 Tensor& out) {
    out = interop__fft_r2c_cuda(self, dim, normalization, onesided);
    return out;
}

Tensor interop__fft_c2r_cuda(const Tensor& self,
                            const std::vector<int64_t>& dim,
                            int64_t normalization, int64_t last_dim_size) {
    TP_CHECK(dim.size() == 1,
             "_fft_c2r transforms exactly one dimension, got ", dim.size());
    const int64_t d = dim[0];
    Tensor moved = fft_move_dim_last(self, d);
    const char* norm = fft_norm_name(normalization);
    Tensor out = dispatch_cuda<Tensor>("fft_irfft", moved, last_dim_size,
                                       moved.dim() - 1, std::string(norm));
    return fft_move_dim_back(out, d, self.dim());
}

Tensor& interop__fft_c2r_out_cuda(const Tensor& self,
                                 const std::vector<int64_t>& dim,
                                 int64_t normalization, int64_t last_dim_size,
                                 Tensor& out) {
    out = interop__fft_c2r_cuda(self, dim, normalization, last_dim_size);
    return out;
}

Tensor interop__fft_c2c_cuda(const Tensor& self,
                            const std::vector<int64_t>& dim,
                            int64_t normalization, bool forward) {
    TP_CHECK(dim.size() == 1,
             "_fft_c2c transforms exactly one dimension, got ", dim.size());
    const int64_t d = dim[0];
    Tensor moved = fft_move_dim_last(self, d);
    const char* norm = fft_norm_name(normalization);
    Tensor out;
    if (forward) {
        out = dispatch_cuda<Tensor>("fft_fft", moved, int64_t(-1),
                                    moved.dim() - 1, std::string(norm));
    } else {
        out = dispatch_cuda<Tensor>("fft_ifft", moved, int64_t(-1),
                                    moved.dim() - 1, std::string(norm));
    }
    return fft_move_dim_back(out, d, self.dim());
}

Tensor& interop__fft_c2c_out_cuda(const Tensor& self,
                                 const std::vector<int64_t>& dim,
                                 int64_t normalization, bool forward,
                                 Tensor& out) {
    out = interop__fft_c2c_cuda(self, dim, normalization, forward);
    return out;
}

// ---------------------------------------------------------------------------
// _cholesky_solve_helper: the dispatch-level spelling of cholesky_solve.
// ---------------------------------------------------------------------------

Tensor interop__cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A,
                                           bool upper) {
    return dispatch_cuda<Tensor>("cholesky_solve", self, A, upper);
}

namespace {

constexpr int kSparseIndexThreads = 256;
constexpr int kSparseIndexMaxBatchDims = 64;

struct SparseBatchShape {
    int64_t ndim;
    int64_t dims[kSparseIndexMaxBatchDims];
};

inline void sparse_index_cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("CUDA Error: ") + cudaGetErrorString(status));
    }
}

void clear_sparse_index_output(Tensor& result) {
    if (result.numel() == 0) return;
    sparse_index_cuda_check(cudaMemsetAsync(
        result.data_ptr(), 0,
        static_cast<size_t>(result.numel()) * result.itemsize(),
        getCurrentCUDAStream().stream()));
}

template <typename input_t, typename output_t>
__global__ void coo_to_csr_kernel(output_t* data_out, const input_t* data_in,
                                  int64_t size, int64_t numel) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t work = tid; work <= numel; work += stride) {
        if (work == 0) {
            const int64_t first = static_cast<int64_t>(data_in[0]);
            for (int64_t i = 0; i <= first; ++i) {
                data_out[i] = static_cast<output_t>(0);
            }
        } else if (work < numel) {
            const int64_t begin = static_cast<int64_t>(data_in[work - 1]);
            const int64_t end = static_cast<int64_t>(data_in[work]);
            for (int64_t i = begin; i < end; ++i) {
                data_out[i + 1] = static_cast<output_t>(work);
            }
        } else {
            const int64_t last = static_cast<int64_t>(data_in[numel - 1]);
            for (int64_t i = last + 1; i < size + 1; ++i) {
                data_out[i] = static_cast<output_t>(numel);
            }
        }
    }
}

template <typename input_t, typename output_t>
void fill_coo_to_csr_cuda(Tensor& result, const Tensor& input,
                          int64_t size) {
    const Tensor input_c = input.contiguous();
    const int64_t numel = input_c.numel();
    if (numel == 0) {
        clear_sparse_index_output(result);
        return;
    }
    const int64_t blocks =
        (numel + 1 + kSparseIndexThreads - 1) / kSparseIndexThreads;
    coo_to_csr_kernel<input_t, output_t>
        <<<static_cast<unsigned int>(blocks), kSparseIndexThreads, 0,
           getCurrentCUDAStream().stream()>>>(
            result.data_ptr<output_t>(), input_c.data_ptr<input_t>(), size,
            numel);
    sparse_index_cuda_check(cudaGetLastError());
}

template <typename output_t>
void dispatch_coo_to_csr_input_cuda(Tensor& result, const Tensor& input,
                                    int64_t size) {
#define TP_CUDA_COO_TO_CSR_CASE(ctype, name)                                  \
    case DType::name:                                                          \
        fill_coo_to_csr_cuda<ctype, output_t>(result, input, size);            \
        return;
    switch (input.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_CUDA_COO_TO_CSR_CASE)
        default:
            TP_THROW(TypeError,
                     "_convert_indices_from_coo_to_csr: input must be integral");
    }
#undef TP_CUDA_COO_TO_CSR_CASE
}

void check_coo_to_csr_cuda(const Tensor& input, int64_t size) {
    TP_CHECK(input.dim() <= 1,
             "_convert_indices_from_coo_to_csr: input must be a vector, got ",
             input.dim(), " dimensions");
    TP_CHECK(size >= 0,
             "_convert_indices_from_coo_to_csr: size must be non-negative, got ",
             size);
    TP_CHECK(isIntegralType(input.dtype(), false),
             "_convert_indices_from_coo_to_csr: input must be integral");
}

template <typename input_t, typename output_t>
__global__ void csr_to_coo_rows_kernel(output_t* row_data,
                                       const input_t* crow_data,
                                       int64_t nrows, int64_t nnz,
                                       int64_t nbatches) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t row_count = nrows * nbatches;
    for (int64_t linear_row = tid; linear_row < row_count;
         linear_row += stride) {
        const int64_t batch = linear_row / nrows;
        const int64_t row = linear_row % nrows;
        const int64_t base = batch * (nrows + 1);
        const int64_t begin = static_cast<int64_t>(crow_data[base + row]);
        const int64_t end = static_cast<int64_t>(crow_data[base + row + 1]);
        for (int64_t index = begin; index < end; ++index) {
            row_data[batch * nnz + index] = static_cast<output_t>(row);
        }
    }
}

template <typename col_t, typename output_t>
__global__ void csr_to_coo_columns_kernel(
    output_t* result_data, output_t* row0, output_t* row1,
    const col_t* col_data, int64_t total_nnz, int64_t nnz,
    SparseBatchShape batch_shape, bool transpose) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = tid; index < total_nnz; index += stride) {
        const int64_t batch = index / nnz;
        int64_t remainder = batch;
        for (int64_t dim = batch_shape.ndim - 1; dim >= 0; --dim) {
            const int64_t extent = batch_shape.dims[dim];
            result_data[dim * total_nnz + index] =
                static_cast<output_t>(remainder % extent);
            remainder /= extent;
        }
        if (transpose) {
            row0[index] = static_cast<output_t>(col_data[index]);
        } else {
            row1[index] = static_cast<output_t>(col_data[index]);
        }
    }
}

template <typename crow_t, typename col_t, typename output_t>
void fill_csr_to_coo_cuda(Tensor& result, const Tensor& crow_indices,
                          const Tensor& col_indices, bool transpose,
                          const std::vector<int64_t>& batch_shape) {
    const Tensor crow_c = crow_indices.contiguous();
    const Tensor col_c = col_indices.contiguous();
    const int64_t nrows = crow_c.size(-1) - 1;
    const int64_t nnz = col_c.size(-1);
    const int64_t total_nnz = col_c.numel();
    const int64_t batch_ndim = static_cast<int64_t>(batch_shape.size());
    const int64_t batch_count =
        batch_ndim == 0 ? 1 : (nnz == 0 ? 0 : total_nnz / nnz);
    if (nrows == 0 || nnz == 0 || total_nnz == 0) {
        clear_sparse_index_output(result);
        return;
    }

    SparseBatchShape shape{};
    shape.ndim = batch_ndim;
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        shape.dims[dim] = batch_shape[static_cast<size_t>(dim)];
    }
    output_t* result_data = result.data_ptr<output_t>();
    output_t* row0 = result.select(0, transpose ? batch_ndim + 1 : batch_ndim)
                          .data_ptr<output_t>();
    output_t* row1 = result.select(0, transpose ? batch_ndim : batch_ndim + 1)
                          .data_ptr<output_t>();
    const int64_t row_count = batch_count * nrows;
    const int row_blocks =
        static_cast<int>((row_count + kSparseIndexThreads - 1) /
                         kSparseIndexThreads);
    csr_to_coo_rows_kernel<crow_t, output_t>
        <<<static_cast<unsigned int>(row_blocks), kSparseIndexThreads, 0,
           getCurrentCUDAStream().stream()>>>(
            transpose ? row1 : row0, crow_c.data_ptr<crow_t>(), nrows, nnz,
            batch_count);
    sparse_index_cuda_check(cudaGetLastError());

    const int64_t column_blocks =
        (total_nnz + kSparseIndexThreads - 1) / kSparseIndexThreads;
    csr_to_coo_columns_kernel<col_t, output_t>
        <<<static_cast<unsigned int>(column_blocks), kSparseIndexThreads, 0,
           getCurrentCUDAStream().stream()>>>(
            result_data, row0, row1, col_c.data_ptr<col_t>(), total_nnz, nnz,
            shape, transpose);
    sparse_index_cuda_check(cudaGetLastError());
}

template <typename crow_t, typename output_t>
void dispatch_csr_to_coo_columns_cuda(
    Tensor& result, const Tensor& crow_indices, const Tensor& col_indices,
    bool transpose, const std::vector<int64_t>& batch_shape) {
#define TP_CUDA_CSR_TO_COO_CASE(ctype, name)                                  \
    case DType::name:                                                          \
        fill_csr_to_coo_cuda<crow_t, ctype, output_t>(                         \
            result, crow_indices, col_indices, transpose, batch_shape);        \
        return;
    switch (col_indices.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_CUDA_CSR_TO_COO_CASE)
        default:
            TP_THROW(TypeError,
                     "_convert_indices_from_csr_to_coo: columns must be integral");
    }
#undef TP_CUDA_CSR_TO_COO_CASE
}

template <typename output_t>
void dispatch_csr_to_coo_rows_cuda(
    Tensor& result, const Tensor& crow_indices, const Tensor& col_indices,
    bool transpose, const std::vector<int64_t>& batch_shape) {
#define TP_CUDA_CSR_TO_COO_ROW_CASE(ctype, name)                              \
    case DType::name:                                                          \
        dispatch_csr_to_coo_columns_cuda<ctype, output_t>(                     \
            result, crow_indices, col_indices, transpose, batch_shape);        \
        return;
    switch (crow_indices.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_CUDA_CSR_TO_COO_ROW_CASE)
        default:
            TP_THROW(TypeError,
                     "_convert_indices_from_csr_to_coo: row pointers must be integral");
    }
#undef TP_CUDA_CSR_TO_COO_ROW_CASE
}

std::vector<int64_t> check_csr_to_coo_cuda(const Tensor& crow_indices,
                                           const Tensor& col_indices) {
    TP_CHECK(crow_indices.dim() >= 1 && col_indices.dim() >= 1,
             "_convert_indices_from_csr_to_coo: inputs must have at least one dimension");
    TP_CHECK(crow_indices.dim() == col_indices.dim(),
             "_convert_indices_from_csr_to_coo: inputs must have the same dimensionality");
    TP_CHECK(crow_indices.size(-1) >= 1,
             "_convert_indices_from_csr_to_coo: row pointer dimension must be non-empty");
    TP_CHECK(crow_indices.dim() - 1 <= kSparseIndexMaxBatchDims,
             "_convert_indices_from_csr_to_coo: too many batch dimensions");
    for (int64_t dim = 0; dim < crow_indices.dim() - 1; ++dim) {
        TP_CHECK(crow_indices.size(dim) == col_indices.size(dim),
                 "_convert_indices_from_csr_to_coo: batch dimensions must match");
    }
    TP_CHECK(isIntegralType(crow_indices.dtype(), false),
             "_convert_indices_from_csr_to_coo: row pointers must be integral");
    TP_CHECK(isIntegralType(col_indices.dtype(), false),
             "_convert_indices_from_csr_to_coo: columns must be integral");

    std::vector<int64_t> batch_shape;
    batch_shape.reserve(static_cast<size_t>(crow_indices.dim() - 1));
    int64_t batch_count = 1;
    for (int64_t dim = 0; dim < crow_indices.dim() - 1; ++dim) {
        const int64_t extent = crow_indices.size(dim);
        batch_shape.push_back(extent);
        batch_count *= extent;
    }
    const int64_t nrows = crow_indices.size(-1) - 1;
    const int64_t nnz = col_indices.size(-1);
    TP_CHECK(col_indices.numel() == batch_count * nnz,
             "_convert_indices_from_csr_to_coo: invalid batch layout");
    TP_CHECK(crow_indices.numel() == batch_count * (nrows + 1),
             "_convert_indices_from_csr_to_coo: invalid row pointer layout");
    return batch_shape;
}

} // namespace

Tensor convert_indices_from_coo_to_csr_cuda(const Tensor& input,
                                            int64_t size, bool out_int32) {
    check_coo_to_csr_cuda(input, size);
    Tensor result = ops::empty(
        {size + 1}, out_int32 ? DType::Int32 : DType::Int64, input.device());
    if (out_int32) {
        dispatch_coo_to_csr_input_cuda<int32_t>(result, input, size);
    } else {
        dispatch_coo_to_csr_input_cuda<int64_t>(result, input, size);
    }
    return result;
}

Tensor& _convert_indices_from_coo_to_csr_structured_cuda(
    const Tensor& input, int64_t size, bool out_int32, Tensor& out) {
    check_coo_to_csr_cuda(input, size);
    const DType dtype = out_int32 ? DType::Int32 : DType::Int64;
    TP_CHECK(out.defined() && out.dtype() == dtype,
             "_convert_indices_from_coo_to_csr: output dtype is incorrect");
    TP_CHECK(out.device() == input.device(),
             "_convert_indices_from_coo_to_csr: output device must match input");
    const std::vector<int64_t> shape = {size + 1};
    out.resize_(shape);
    const bool copy_back = !out.is_contiguous();
    Tensor target = copy_back ? ops::empty(shape, dtype, input.device()) : out;
    if (out_int32) {
        dispatch_coo_to_csr_input_cuda<int32_t>(target, input, size);
    } else {
        dispatch_coo_to_csr_input_cuda<int64_t>(target, input, size);
    }
    if (copy_back) out.copy_(target);
    return out;
}

Tensor convert_indices_from_csr_to_coo_cuda(
    const Tensor& crow_indices, const Tensor& col_indices, bool out_int32,
    bool transpose) {
    const std::vector<int64_t> batch_shape =
        check_csr_to_coo_cuda(crow_indices, col_indices);
    Tensor result = ops::empty(
        {col_indices.dim() + 1, col_indices.numel()},
        out_int32 ? DType::Int32 : DType::Int64, crow_indices.device());
    if (out_int32) {
        dispatch_csr_to_coo_rows_cuda<int32_t>(
            result, crow_indices, col_indices, transpose, batch_shape);
    } else {
        dispatch_csr_to_coo_rows_cuda<int64_t>(
            result, crow_indices, col_indices, transpose, batch_shape);
    }
    return result;
}

Tensor& _convert_indices_from_csr_to_coo_structured_cuda(
    const Tensor& crow_indices, const Tensor& col_indices, bool out_int32,
    bool transpose, Tensor& out) {
    const std::vector<int64_t> batch_shape =
        check_csr_to_coo_cuda(crow_indices, col_indices);
    const DType dtype = out_int32 ? DType::Int32 : DType::Int64;
    TP_CHECK(out.defined() && out.dtype() == dtype,
             "_convert_indices_from_csr_to_coo: output dtype is incorrect");
    TP_CHECK(out.device() == crow_indices.device(),
             "_convert_indices_from_csr_to_coo: output device must match input");
    const std::vector<int64_t> shape =
        {col_indices.dim() + 1, col_indices.numel()};
    out.resize_(shape);
    const bool copy_back = !out.is_contiguous();
    Tensor target = copy_back
        ? ops::empty(shape, dtype, crow_indices.device())
        : out;
    if (out_int32) {
        dispatch_csr_to_coo_rows_cuda<int32_t>(
            target, crow_indices, col_indices, transpose, batch_shape);
    } else {
        dispatch_csr_to_coo_rows_cuda<int64_t>(
            target, crow_indices, col_indices, transpose, batch_shape);
    }
    if (copy_back) out.copy_(target);
    return out;
}

// ---------------------------------------------------------------------------
// _thnn_fused_gru_cell: gate arithmetic for the fused GRU cell.  Gates run
// along the last dimension in [reset, update, candidate] order.
//
// Forward: r = sigmoid(gi_r + gh_r + b_r), z = sigmoid(gi_z + gh_z + b_z),
//          n = tanh(gi_n + r * gh_n + b_n), hy = n + z * (hx - n).
// The workspace saves [r, z, n, hx, gh_n + b_n] for the backward pass.
// ---------------------------------------------------------------------------

namespace {

Tensor gate_slice(const Tensor& gates, int64_t idx, int64_t total_gates) {
    const int64_t span = gates.size(-1) / total_gates;
    return gates.narrow(-1, idx * span, span);
}

}  // namespace

std::tuple<Tensor, Tensor> interop__thnn_fused_gru_cell_cuda(
        const Tensor& input_gates, const Tensor& hidden_gates,
        const Tensor& hx, const std::optional<Tensor>& input_bias,
        const std::optional<Tensor>& hidden_bias) {
    Tensor gi = input_gates;
    Tensor gh = hidden_gates;
    if (input_bias.has_value() && input_bias->defined()) {
        gi = ops::add(gi, *input_bias);
    }
    if (hidden_bias.has_value() && hidden_bias->defined()) {
        gh = ops::add(gh, *hidden_bias);
    }
    Tensor r = ops::sigmoid(ops::add(gate_slice(gi, 0, 3), gate_slice(gh, 0, 3)));
    Tensor z = ops::sigmoid(ops::add(gate_slice(gi, 1, 3), gate_slice(gh, 1, 3)));
    Tensor n = ops::tanh(ops::add(gate_slice(gi, 2, 3),
                                  ops::mul(r, gate_slice(gh, 2, 3))));
    Tensor hy = ops::add(n, ops::mul(z, ops::sub(hx, n)));
    // Workspace layout: [r, z, n, hx, gh_n + b_n].
    Tensor workspace = ops::cat(
        {r, z, n, hx, gate_slice(gh, 2, 3)}, -1);
    return std::make_tuple(hy, workspace);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
interop__thnn_fused_gru_cell_backward_cuda(const Tensor& grad_hy,
                                           const Tensor& workspace,
                                           bool has_bias) {
    // Workspace slices (per hidden unit): r, z, n, hx, hgn.
    Tensor r = gate_slice(workspace, 0, 5);
    Tensor z = gate_slice(workspace, 1, 5);
    Tensor n = gate_slice(workspace, 2, 5);
    Tensor hx = gate_slice(workspace, 3, 5);
    Tensor hgn = gate_slice(workspace, 4, 5);
    Tensor go = grad_hy;
    // hy = n + z * (hx - n):
    //   dz = go * (hx - n) * z * (1 - z)
    //   dn = go * (1 - z) * (1 - n^2)
    //   dhx = go * z
    //   dr = dn * hgn * r * (1 - r)
    // gate grads: input [dr, dz, dn]; hidden [dr, dz, r * dn].
    Tensor sig_z = ops::mul(z, ops::sub(ops::ones_like(z), z));
    Tensor sig_r = ops::mul(r, ops::sub(ops::ones_like(r), r));
    Tensor gz = ops::mul(ops::mul(go, ops::sub(hx, n)), sig_z);
    Tensor gn = ops::mul(ops::mul(go, ops::sub(ops::ones_like(z), z)),
                         ops::sub(ops::ones_like(ops::mul(n, n)), ops::mul(n, n)));
    Tensor gr = ops::mul(ops::mul(gn, hgn), sig_r);
    Tensor grad_hx = ops::mul(go, z);
    Tensor grad_input_gates = ops::cat({gr, gz, gn}, -1);
    Tensor grad_hidden_gates = ops::cat({gr, gz, ops::mul(r, gn)}, -1);
    Tensor grad_input_bias;
    Tensor grad_hidden_bias;
    if (has_bias) {
        // Bias gradients sum over every batch axis.
        std::vector<int64_t> reduce_dims;
        for (int64_t d = 0; d < grad_input_gates.dim() - 1; ++d) {
            reduce_dims.push_back(d);
        }
        grad_input_bias = ops::sum(grad_input_gates, reduce_dims, false);
        grad_hidden_bias = ops::sum(grad_hidden_gates, reduce_dims, false);
    }
    return std::make_tuple(grad_input_gates, grad_hidden_gates, grad_hx,
                           grad_input_bias, grad_hidden_bias);
}

// ---------------------------------------------------------------------------
// _thnn_fused_lstm_cell: gates run along the last dimension in
// [input, forget, cell, output] order.
//
// cy = f * cx + i * c, hy = o * tanh(cy).  The workspace saves [i, f, c, o].
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor> interop__thnn_fused_lstm_cell_cuda(
        const Tensor& input_gates, const Tensor& hidden_gates,
        const Tensor& cx, const std::optional<Tensor>& input_bias,
        const std::optional<Tensor>& hidden_bias) {
    Tensor gi = input_gates;
    Tensor gh = hidden_gates;
    if (input_bias.has_value() && input_bias->defined()) {
        gi = ops::add(gi, *input_bias);
    }
    if (hidden_bias.has_value() && hidden_bias->defined()) {
        gh = ops::add(gh, *hidden_bias);
    }
    Tensor i = ops::sigmoid(ops::add(gate_slice(gi, 0, 4), gate_slice(gh, 0, 4)));
    Tensor f = ops::sigmoid(ops::add(gate_slice(gi, 1, 4), gate_slice(gh, 1, 4)));
    Tensor c = ops::tanh(ops::add(gate_slice(gi, 2, 4), gate_slice(gh, 2, 4)));
    Tensor o = ops::sigmoid(ops::add(gate_slice(gi, 3, 4), gate_slice(gh, 3, 4)));
    Tensor cy = ops::add(ops::mul(f, cx), ops::mul(i, c));
    Tensor hy = ops::mul(o, ops::tanh(cy));
    Tensor workspace = ops::cat({i, f, c, o}, -1);
    return std::make_tuple(hy, cy, workspace);
}

std::tuple<Tensor, Tensor, Tensor> interop__thnn_fused_lstm_cell_backward_impl_cuda(
        const std::optional<Tensor>& grad_hy,
        const std::optional<Tensor>& grad_cy, const Tensor& cx,
        const Tensor& cy, const Tensor& workspace, bool has_bias) {
    const bool has_ghy = grad_hy.has_value() && grad_hy->defined();
    const bool has_gcy = grad_cy.has_value() && grad_cy->defined();
    if (!has_ghy && !has_gcy) {
        return std::tuple<Tensor, Tensor, Tensor>();
    }
    Tensor i = gate_slice(workspace, 0, 4);
    Tensor f = gate_slice(workspace, 1, 4);
    Tensor c = gate_slice(workspace, 2, 4);
    Tensor o = gate_slice(workspace, 3, 4);
    Tensor tanh_cy = ops::tanh(cy);
    Tensor go = has_ghy ? *grad_hy : Tensor();
    Tensor goc = has_gcy ? *grad_cy : Tensor();
    // hy = o * tanh(cy); cy = f * cx + i * c.
    Tensor gog = has_ghy ? ops::mul(go, tanh_cy) : Tensor();
    // gcx accumulates the total cell gradient before the forget gate.
    Tensor gcx;
    if (has_ghy) {
        Tensor tanh_cy_sq = ops::mul(tanh_cy, tanh_cy);
        gcx = ops::add(ops::mul(go, ops::mul(o,
                                  ops::sub(ops::ones_like(tanh_cy_sq),
                                           tanh_cy_sq))),
                       has_gcy ? goc : ops::zeros_like(cy));
    } else {
        gcx = goc;
    }
    Tensor gig = ops::mul(gcx, c);
    Tensor gfg = ops::mul(gcx, cx);
    Tensor gcg = ops::mul(gcx, i);
    Tensor grad_cx = ops::mul(gcx, f);
    gig = ops::mul(gig, ops::mul(ops::sub(ops::ones_like(i), i), i));
    gfg = ops::mul(gfg, ops::mul(ops::sub(ops::ones_like(f), f), f));
    gcg = ops::mul(gcg, ops::sub(ops::ones_like(ops::mul(c, c)), ops::mul(c, c)));
    if (has_ghy) {
        gog = ops::mul(gog, ops::mul(ops::sub(ops::ones_like(o), o), o));
    } else {
        gog = ops::zeros_like(o);
    }
    Tensor grad_gates = ops::cat({gig, gfg, gcg, gog}, -1);
    Tensor grad_bias;
    if (has_bias) {
        std::vector<int64_t> reduce_dims;
        for (int64_t d = 0; d < grad_gates.dim() - 1; ++d) {
            reduce_dims.push_back(d);
        }
        grad_bias = ops::sum(grad_gates, reduce_dims, false);
    }
    return std::make_tuple(grad_gates, grad_cx, grad_bias);
}

// ---------------------------------------------------------------------------
// _adaptive_avg_pool* / _cummax/min helper alias spellings
// ---------------------------------------------------------------------------

Tensor interop__adaptive_avg_pool2d_cuda(
        const Tensor& self, const std::vector<int64_t>& output_size) {
    return dispatch_cuda<Tensor>("adaptive_avg_pool2d", self, output_size);
}

Tensor interop__adaptive_avg_pool2d_backward_cuda(const Tensor& grad_output,
                                                  const Tensor& input) {
    return dispatch_cuda<Tensor>("adaptive_avg_pool2d_backward", grad_output,
                                 input);
}

Tensor interop__adaptive_avg_pool3d_cuda(
        const Tensor& self, const std::vector<int64_t>& output_size) {
    return dispatch_cuda<Tensor>("adaptive_avg_pool3d", self, output_size);
}

Tensor interop__adaptive_avg_pool3d_backward_cuda(const Tensor& grad_output,
                                                  const Tensor& input) {
    return dispatch_cuda<Tensor>("adaptive_avg_pool3d_backward", grad_output,
                                 input);
}

void interop__cummax_helper_cuda(const Tensor& self, Tensor& values,
                                 Tensor& indices, int64_t dim) {
    std::tie(values, indices) =
        dispatch_cuda<std::tuple<Tensor, Tensor>>("cummax", self, dim);
}

void interop__cummin_helper_cuda(const Tensor& self, Tensor& values,
                                 Tensor& indices, int64_t dim) {
    std::tie(values, indices) =
        dispatch_cuda<std::tuple<Tensor, Tensor>>("cummin", self, dim);
}

// ---------------------------------------------------------------------------
// Sampling spellings (standard gamma, binomial, dirichlet sample/grad) and
// their kernels live in RandomKernels.cu, where the philox RNG helpers are
// defined.
// ---------------------------------------------------------------------------

TENSORPLAY_LIBRARY_IMPL(CUDA, InteropAliasKernels) {
    // histogram / diagonal index builders
    m.impl("histc", interop_histc_cuda);
    m.impl("histc.out", interop_histc_out_cuda);
    m.impl("tril_indices", interop_tril_indices_cuda);
    m.impl("triu_indices", interop_triu_indices_cuda);

    // uniqueness / static nonzero
    m.impl("unique_consecutive", interop_unique_consecutive_cuda);
    m.impl("nonzero_static", interop_nonzero_static_cuda);
    m.impl("nonzero_static.out", interop_nonzero_static_out_cuda);

    // misc internal spellings
    m.impl("repeat_interleave.Tensor", interop_repeat_interleave_Tensor_cuda);
    m.impl("embedding_renorm_", interop_embedding_renorm__cuda);
    m.impl("sspaddmm", interop_sspaddmm_cuda);
    m.impl("sspaddmm.out", interop_sspaddmm_out_cuda);
    m.impl("_masked_scale", interop_masked_scale_cuda);
    m.impl("_masked_softmax", interop_masked_softmax_cuda);
    m.impl("_masked_softmax_backward", interop_masked_softmax_backward_cuda);
    m.impl("_fused_rms_norm", interop_fused_rms_norm_cuda);
    m.impl("_fused_rms_norm_backward", interop_fused_rms_norm_backward_cuda);
    m.impl("_chunk_cat", interop_chunk_cat_cuda);
    m.impl("_chunk_cat.out", interop__chunk_cat_out_cuda);
    m.impl("_fused_dropout", interop__fused_dropout_cuda);
    m.impl("_local_scalar_dense", interop__local_scalar_dense_cuda);

    // depthwise / slow convolutions
    m.impl("_conv_depthwise2d", interop__conv_depthwise2d_cuda);
    m.impl("_conv_depthwise2d.out", interop__conv_depthwise2d_out_cuda);
    m.impl("conv_depthwise3d", interop_conv_depthwise3d_cuda);
    m.impl("_slow_conv2d_forward", interop__slow_conv2d_forward_cuda);
    m.impl("_slow_conv2d_forward.output", interop__slow_conv2d_forward_output_cuda);
    m.impl("_slow_conv2d_backward.grad_input", interop__slow_conv2d_backward_grad_input_cuda);
    m.impl("_slow_conv2d_backward.output_mask", interop__slow_conv2d_backward_output_mask_cuda);
    m.impl("slow_conv_dilated2d", interop_slow_conv_dilated2d_cuda);
    m.impl("slow_conv_dilated3d", interop_slow_conv_dilated3d_cuda);
    m.impl("slow_conv_transpose2d.out", interop_slow_conv_transpose2d_out_cuda);
    m.impl("slow_conv_transpose3d", interop_slow_conv_transpose3d_cuda);
    m.impl("slow_conv_transpose3d.out", interop_slow_conv_transpose3d_out_cuda);

    // batch normalization family
    m.impl("native_batch_norm", interop_native_batch_norm_cuda);
    m.impl("native_batch_norm.out", interop_native_batch_norm_out_cuda);
    m.impl("_native_batch_norm_legit", interop__native_batch_norm_legit_cuda);
    m.impl("_native_batch_norm_legit.out", interop__native_batch_norm_legit_out_cuda);
    m.impl("_native_batch_norm_legit.no_stats", interop__native_batch_norm_legit_no_stats_cuda);
    m.impl("_native_batch_norm_legit.no_stats_out", interop__native_batch_norm_legit_no_stats_out_cuda);
    m.impl("_batch_norm_with_update", interop__batch_norm_with_update_cuda);
    m.impl("_batch_norm_no_update", interop__batch_norm_no_update_cuda);
    m.impl("batch_norm_stats", interop_batch_norm_stats_cuda);
    m.impl("batch_norm_elemt", interop_batch_norm_elemt_cuda);
    m.impl("batch_norm_elemt.out", interop_batch_norm_elemt_out_cuda);
    m.impl("batch_norm_backward_reduce", interop_batch_norm_backward_reduce_cuda);
    m.impl("batch_norm_backward_elemt", interop_batch_norm_backward_elemt_cuda);
    m.impl("batch_norm_gather_stats", interop_batch_norm_gather_stats_cuda);
    m.impl("batch_norm_gather_stats_with_counts", interop_batch_norm_gather_stats_with_counts_cuda);
    m.impl("batch_norm_update_stats", interop_batch_norm_update_stats_cuda);

    // layer / group normalization
    m.impl("native_layer_norm", interop_native_layer_norm_cuda);
    m.impl("native_layer_norm_backward", interop_native_layer_norm_backward_cuda);
    m.impl("native_group_norm", interop_native_group_norm_cuda);
    m.impl("native_group_norm_backward", interop_native_group_norm_backward_cuda);

    // fft spellings
    m.impl("_fft_r2c", interop__fft_r2c_cuda);
    m.impl("_fft_r2c.out", interop__fft_r2c_out_cuda);
    m.impl("_fft_c2r", interop__fft_c2r_cuda);
    m.impl("_fft_c2r.out", interop__fft_c2r_out_cuda);
    m.impl("_fft_c2c", interop__fft_c2c_cuda);
    m.impl("_fft_c2c.out", interop__fft_c2c_out_cuda);

    // linear algebra helper spellings
    m.impl("_cholesky_solve_helper", interop__cholesky_solve_helper_cuda);
    m.impl("_convert_indices_from_coo_to_csr", convert_indices_from_coo_to_csr_cuda);
    m.impl("_convert_indices_from_coo_to_csr.out",
           _convert_indices_from_coo_to_csr_structured_cuda);
    m.impl("_convert_indices_from_csr_to_coo", convert_indices_from_csr_to_coo_cuda);
    m.impl("_convert_indices_from_csr_to_coo.out",
           _convert_indices_from_csr_to_coo_structured_cuda);

    // fused rnn cells
    m.impl("_thnn_fused_gru_cell", interop__thnn_fused_gru_cell_cuda);
    m.impl("_thnn_fused_gru_cell_backward", interop__thnn_fused_gru_cell_backward_cuda);
    m.impl("_thnn_fused_lstm_cell", interop__thnn_fused_lstm_cell_cuda);
    m.impl("_thnn_fused_lstm_cell_backward_impl", interop__thnn_fused_lstm_cell_backward_impl_cuda);

    // adaptive pooling alias spellings
    m.impl("_adaptive_avg_pool2d", interop__adaptive_avg_pool2d_cuda);
    m.impl("_adaptive_avg_pool2d_backward", interop__adaptive_avg_pool2d_backward_cuda);
    m.impl("_adaptive_avg_pool3d", interop__adaptive_avg_pool3d_cuda);
    m.impl("_adaptive_avg_pool3d_backward", interop__adaptive_avg_pool3d_backward_cuda);

    // cummax/cummin out-style helpers
    m.impl("_cummax_helper", interop__cummax_helper_cuda);
    m.impl("_cummin_helper", interop__cummin_helper_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
