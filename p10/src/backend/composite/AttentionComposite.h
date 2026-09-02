// Shared composite bodies for the private attention/CTC dispatcher ops.
// The bodies compose recordable dispatcher primitives, so each inner op
// dispatches on the tensor's own backend; the same source registers under
// both the CPU and CUDA keys.
// Nothing here is exported through the public p10 API.
#pragma once

#include "Tensor.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

// Dispatcher-level primitives (defined in TPXOpsGenerated.cpp; declared
// locally because tpx headers are not visible below the p10 layer -- same
// pattern as Einsum.cpp).
namespace tensorplay {
// Dispatcher-level entry points (tensorplay::tpx::ops) are declared by the
// generated TPXOpsGenerated.h included above; no local re-declarations here,
// so inline-merged wrappers stay bindable at every call site.

namespace composite {

namespace ops = tpx::ops;

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// Softmax over the last dim that yields an all-zero row instead of NaN when
// every entry of the row is -inf (fully masked query positions).
inline Tensor safe_softmax_lastdim(const Tensor& scores) {
  using ops::amax, ops::eq, ops::where, ops::sub, ops::exp, ops::sum, ops::div;
  Tensor row_max = amax(scores, {-1}, true);
  // -inf rows would poison exp(x - max); shift them to a finite pivot.
  Tensor finite_max =
      where(eq(row_max, Scalar(kNegInf)), Scalar(0), row_max);
  Tensor e = exp(sub(scores, finite_max));
  Tensor denom = sum(e, {-1}, true);
  Tensor probs = div(e, denom);
  return where(eq(denom, Scalar(0)), Scalar(0), probs);
}

// Additive float mask from a bool mask (True entries attend, False -> -inf).
inline Tensor bool_mask_to_additive(const Tensor& mask, DType dtype) {
  Tensor mask_b = mask.dtype() == DType::Bool ? mask : mask.to(DType::Bool);
  return ops::where(mask_b, Scalar(0),
                    ops::full({}, Scalar(kNegInf), dtype, mask_b.device()));
}

// Causal additive mask, top-left aligned: query row t sees keys <= t.
inline Tensor causal_additive_mask(int64_t l, int64_t skv, DType dtype,
                                   const Device& device) {
  using ops::arange, ops::narrow, ops::view, ops::ge, ops::full,
      ops::logical_not, ops::masked_fill;
  Tensor idx = arange(Scalar(0), Scalar(std::max(l, skv)), Scalar(1),
                      DType::Int64, device);
  Tensor keep = ge(view(narrow(idx, 0, 0, l), {l, 1}),
                   view(narrow(idx, 0, 0, skv), {1, skv}));
  Tensor zeros = full({l, skv}, Scalar(0), dtype, device);
  return masked_fill(zeros, logical_not(keep), Scalar(kNegInf));
}

// Scale factor for the math backend: the query side carries sqrt(scale) and
// the key side carries sqrt(scale) so the score product carries `scale`.
inline double math_scale_factor(const std::optional<double>& scale,
                                int64_t head_dim) {
  return scale.has_value()
             ? *scale
             : 1.0 / std::sqrt(static_cast<double>(head_dim));
}

// Expand key/value head counts for group-query attention:
// (..., Hkv, S, D) -> (..., Hq, S, D) with each kv head serving a contiguous
// block of query heads.
inline std::pair<Tensor, Tensor> expand_gqa(const Tensor& query,
                                            const Tensor& key,
                                            const Tensor& value,
                                            bool enable_gqa) {
  using ops::view, ops::expand, ops::reshape;
  if (!enable_gqa) return {key, value};
  if (query.dim() < 3 || key.dim() < 3) {
    TP_THROW(ValueError, "sdpa math: enable_gqa requires 4D inputs");
  }
  const int64_t hq = query.size(-3);
  const int64_t hk = key.size(-3);
  if (hq == hk) return {key, value};
  if (hq % hk != 0) {
    TP_THROW(ValueError,
             "sdpa math: enable_gqa requires the query head count to be "
             "divisible by the key/value head count");
  }
  const int64_t g = hq / hk;
  auto expand_heads = [&](const Tensor& t) {
    std::vector<int64_t> shape(static_cast<std::vector<int64_t>>(t.shape()));
    const int64_t rank = t.dim();
    // (..., Hk, S, D) -> (..., Hk, 1, S, D) -> (..., Hk, g, S, D)
    shape.insert(shape.end() - 2, 1);
    shape[rank - 2] = g;
    Tensor t5 = view(t, shape);
    Tensor t5e = expand(t5, shape);
    // (..., Hk, g, S, D) -> (..., Hq, S, D) with Hq = Hk * g
    std::vector<int64_t> out_shape(shape.begin(), shape.end() - 3);
    out_shape.push_back(hq);
    out_shape.push_back(t.size(-2));
    out_shape.push_back(t.size(-1));
    return reshape(t5e, out_shape);
  };
  return {expand_heads(key), expand_heads(value)};
}

// `_scaled_dot_product_attention_math`: naive attention composed from
// recordable primitives.  Reduced dtypes accumulate in float32.
inline std::tuple<Tensor, Tensor> sdpa_math_composite(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_mask, double dropout_p, bool is_causal,
    const std::optional<Tensor>& dropout_mask, std::optional<double> scale,
    bool enable_gqa) {
  using ops::to, ops::matmul, ops::mul, ops::transpose, ops::add, ops::softmax,
      ops::where, ops::native_dropout;
  const DType origin_dtype = query.dtype();
  if (origin_dtype != DType::Float32 && origin_dtype != DType::Float64 &&
      origin_dtype != DType::Float16 && origin_dtype != DType::BFloat16) {
    TP_THROW(NotImplementedError,
             "sdpa math: expected float32/float64/float16/bfloat16");
  }
  const int64_t head_dim = query.size(-1);
  if (head_dim == 0) {
    TP_THROW(ValueError, "sdpa math: head dimension must be non-zero");
  }
  const bool reduce = origin_dtype == DType::Float16 ||
                      origin_dtype == DType::BFloat16;
  Tensor q = reduce ? to(query, DType::Float32) : query;
  Tensor k = reduce ? to(key, DType::Float32) : key;
  Tensor v = reduce ? to(value, DType::Float32) : value;

  std::tie(k, v) = expand_gqa(q, k, v, enable_gqa);
  if (k.shape() != v.shape()) {
    TP_THROW(ValueError, "sdpa math: key and value shapes must match");
  }
  if (k.size(-1) != q.size(-1)) {
    TP_THROW(ValueError, "sdpa math: query and key head dims must match");
  }
  const int64_t q_rank = q.dim(), k_rank = k.dim();
  if (q_rank >= 3 && k_rank == q_rank) {
    const std::vector<int64_t> qs = q.shape(), ks = k.shape();
    if (!std::equal(qs.begin(), qs.end() - 2, ks.begin())) {
      TP_THROW(ValueError,
               "sdpa math: query and key/value leading dims must match");
    }
  }

  const double s = math_scale_factor(scale, head_dim);
  const double sqrt_s = std::sqrt(std::abs(s));
  Tensor q_scaled = mul(q, Scalar(s < 0 ? -sqrt_s : sqrt_s));
  Tensor scores = matmul(q_scaled, mul(transpose(k, -2, -1), Scalar(sqrt_s)));

  const bool masked = is_causal || attn_mask.has_value();
  if (is_causal) {
    if (attn_mask.has_value()) {
      TP_THROW(ValueError,
               "sdpa math: explicit attn_mask must not be set when is_causal");
    }
    scores = add(scores, causal_additive_mask(q.size(-2), k.size(-2),
                                              q.dtype(), q.device()));
  }
  if (attn_mask.has_value()) {
    const Tensor& m = *attn_mask;
    if (m.dtype() == DType::Bool) {
      scores = add(scores, bool_mask_to_additive(m, q.dtype()));
    } else {
      scores = add(scores, to(m, q.dtype()));
    }
  }

  Tensor probs = masked ? safe_softmax_lastdim(scores)
                        : softmax(scores, -1, DType::Undefined);

  if (dropout_p > 0.0) {
    if (dropout_mask.has_value()) {
      // Validation helper: reuse a caller-supplied drop mask (True = dropped).
      probs = where(*dropout_mask, Scalar(0), probs);
      probs = mul(probs, Scalar(1.0 / (1.0 - dropout_p)));
    } else {
      probs = std::get<0>(ops::native_dropout(probs, dropout_p));
    }
  }

  Tensor out = matmul(probs, v);
  if (reduce) {
    return {to(out, origin_dtype), to(probs, origin_dtype)};
  }
  return {out, probs};
}

// `_native_multi_head_attention`: packed input projection, per-head batched
// matmuls, masked softmax, output projection.  Composite of recordable
// primitives so autograd sees the same graph as the composed reference.
inline std::tuple<Tensor, Tensor> native_mha_composite(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t embed_dim, int64_t num_head, const Tensor& qkv_weight,
    const Tensor& qkv_bias, const Tensor& proj_weight, const Tensor& proj_bias,
    const std::optional<Tensor>& mask, bool need_weights,
    bool average_attn_weights, std::optional<int64_t> mask_type) {
  using ops::narrow, ops::linear, ops::view, ops::permute, ops::contiguous,
      ops::mul, ops::bmm, ops::transpose, ops::softmax, ops::add, ops::mean,
      ops::to, ops::linear;
  const int64_t D = embed_dim;
  if (query.dim() != 3 || key.dim() != 3 || value.dim() != 3) {
    TP_THROW(ValueError,
             "native multi-head attention: expected 3-D query/key/value");
  }
  if (query.size(2) != D) {
    TP_THROW(ValueError,
             "native multi-head attention: embed_dim does not match query's "
             "last dim");
  }
  if (query.shape() != key.shape() || key.shape() != value.shape()) {
    TP_THROW(ValueError,
             "native multi-head attention: query/key/value shapes must match");
  }
  if (qkv_weight.dim() != 2 || qkv_weight.size(0) != 3 * D ||
      qkv_weight.size(1) != D) {
    TP_THROW(ValueError,
             "native multi-head attention: qkv_weight must be {3*embed_dim, "
             "embed_dim}");
  }
  if (qkv_bias.dim() != 1 || qkv_bias.size(0) != 3 * D) {
    TP_THROW(ValueError,
             "native multi-head attention: qkv_bias must be 1-D of "
             "3*embed_dim");
  }
  if (D % num_head != 0) {
    TP_THROW(ValueError,
             "native multi-head attention: embed_dim must divide evenly by "
             "num_heads");
  }
  const int64_t B = query.size(0), T = query.size(1);
  const int64_t dh = D / num_head;
  const DType origin_dtype = query.dtype();
  if (origin_dtype != DType::Float32 && origin_dtype != DType::Float64 &&
      origin_dtype != DType::Float16 && origin_dtype != DType::BFloat16) {
    TP_THROW(NotImplementedError,
             "native multi-head attention: expected "
             "float32/float64/float16/bfloat16");
  }
  const bool reduce = origin_dtype == DType::Float16 ||
                      origin_dtype == DType::BFloat16;
  Tensor q_in = reduce ? to(query, DType::Float32) : query;
  Tensor k_in = reduce ? to(key, DType::Float32) : key;
  Tensor v_in = reduce ? to(value, DType::Float32) : value;
  Tensor w = reduce ? to(qkv_weight, DType::Float32) : qkv_weight;
  Tensor b = reduce ? to(qkv_bias, DType::Float32) : qkv_bias;

  // Packed input projection split into q/k/v thirds.
  Tensor w_q = narrow(w, 0, 0, D);
  Tensor w_k = narrow(w, 0, D, D);
  Tensor w_v = narrow(w, 0, 2 * D, D);
  Tensor b_q = narrow(b, 0, 0, D);
  Tensor b_k = narrow(b, 0, D, D);
  Tensor b_v = narrow(b, 0, 2 * D, D);
  Tensor q = linear(q_in, w_q, b_q);
  Tensor kk = linear(k_in, w_k, b_k);
  Tensor vv = linear(v_in, w_v, b_v);

  // (B, T, D) -> (B, H, T, dh); queries rescale by 1/sqrt(dh).
  auto to_heads = [&](const Tensor& t) {
    return contiguous(permute(view(t, {B, T, num_head, dh}), {0, 2, 1, 3}));
  };
  q = to_heads(q);
  kk = to_heads(kk);
  vv = to_heads(vv);
  q = mul(q, Scalar(1.0 / std::sqrt(static_cast<double>(dh))));

  // Scores per (B, H): flatten heads into the batch dim for bmm.
  Tensor q2 = view(q, {B * num_head, T, dh});
  Tensor k2 = view(kk, {B * num_head, T, dh});
  Tensor v2 = view(vv, {B * num_head, T, dh});
  Tensor scores = bmm(q2, transpose(k2, 1, 2));

  Tensor probs;
  if (mask.has_value() && mask->defined()) {
    const Tensor& m = *mask;
    // Bool -> additive float (True attends, False -> -inf) in the compute
    // dtype, then reshape per the declared mask layout.
    Tensor additive = bool_mask_to_additive(
        m.dtype() == DType::Bool ? m : m.to(DType::Bool), q_in.dtype());
    int64_t mt = mask_type.has_value() ? *mask_type : -1;
    if (mt == 0 && additive.dim() == 2) {
      // (L, S) attention mask broadcast over batch and heads.
      additive = view(additive, {1, 1, T, T});
    } else if (mt == 1 && additive.dim() == 2) {
      // (B, S) key-padding mask broadcast over heads and query positions.
      additive = view(additive, {B, 1, 1, T});
    } else if (additive.dim() == 3) {
      // (B*H, L, S) generic mask folded back to 4-D.
      additive = view(additive, {B, num_head, T, T});
    } else {
      TP_THROW(ValueError,
               "native multi-head attention: unsupported mask layout");
    }
    probs = safe_softmax_lastdim(add(scores, additive));
  } else {
    probs = softmax(scores, -1, DType::Undefined);
  }

  Tensor ctx = bmm(probs, v2);   // (B*H, T, dh)
  Tensor ctx4 = view(ctx, {B, num_head, T, dh});
  Tensor merged =
      view(contiguous(permute(ctx4, {0, 2, 1, 3})), {B, T, D});

  Tensor pw = reduce ? to(proj_weight, DType::Float32) : proj_weight;
  std::optional<Tensor> pb;
  if (proj_bias.defined()) {
    pb = reduce ? to(proj_bias, DType::Float32) : proj_bias;
  }
  Tensor out = linear(merged, pw, pb);
  if (reduce) out = to(out, origin_dtype);

  Tensor weights;
  if (need_weights) {
    weights = view(probs, {B, num_head, T, T});
    if (reduce) weights = to(weights, origin_dtype);
    if (average_attn_weights) {
      weights = mean(weights, {1});
    }
  }
  return {out, weights};
}

// Backend selection shared with the nn.attention routing flags:
// FLASH_ATTENTION(1) covers the plain fused case, MATH(0) the rest, ERROR(-1)
// when nothing can run.
inline int64_t fused_sdp_choice_common(const Tensor& query,
                                       const std::optional<Tensor>& attn_mask,
                                       double dropout_p, bool enable_gqa) {
  const bool plain = !attn_mask.has_value() && dropout_p == 0.0 &&
                     query.dim() == 4 && !enable_gqa && query.size(3) != 0;
  if (plain) return 1;  // FLASH_ATTENTION
  const DType dt = query.dtype();
  const bool math_ok = dt == DType::Float32 || dt == DType::Float64 ||
                       dt == DType::Float16 || dt == DType::BFloat16;
  return math_ok ? 0 : -1;  // MATH : ERROR
}

// Shared body behind the two public `ctc_loss` overloads: one `_ctc_loss`
// call (the derivative formula attaches to it, so gradients flow), then the
// requested reduction.  Impossible alignments stay +inf; `zero_infinity`
// zeroes those entries before reduction.  Mean divides by the clamped target
// lengths so empty targets contribute zero, not inf.
inline Tensor ctc_loss_compose(const Tensor& log_probs, const Tensor& targets,
                               const Tensor& input_lengths,
                               const Tensor& target_lengths, int64_t blank,
                               int64_t reduction, bool zero_infinity) {
  using ops::_ctc_loss, ops::where, ops::eq, ops::zeros_like, ops::clamp_min,
      ops::div, ops::mean, ops::sum, ops::squeeze, ops::unsqueeze;
  const bool is_batched = log_probs.dim() == 3;
  Tensor lp = is_batched ? log_probs : unsqueeze(log_probs, 1);
  Tensor res = std::get<0>(_ctc_loss(lp, targets, input_lengths,
                                     target_lengths, blank, zero_infinity));
  if (zero_infinity) {
    res = where(eq(res, Scalar(std::numeric_limits<double>::infinity())),
                zeros_like(res), res);
  }
  if (reduction == 1) {  // Mean
    Tensor tl = target_lengths.to(res.dtype());
    Tensor tl_clamped = clamp_min(tl, Scalar(1));
    return mean(div(res, tl_clamped));
  }
  if (reduction == 2) {  // Sum
    return sum(res);
  }
  // None: unbatched callers see a single-sequence scalar.
  return is_batched ? res : squeeze(res, 0);
}

inline std::vector<int64_t> lengths_to_vector(const Tensor& lengths) {
  auto l = lengths.contiguous();
  if (l.dtype() == DType::Int64) {
    const int64_t* p = l.data_ptr<int64_t>();
    return std::vector<int64_t>(p, p + l.numel());
  }
  if (l.dtype() == DType::Int32) {
    const int32_t* p = l.data_ptr<int32_t>();
    return std::vector<int64_t>(p, p + l.numel());
  }
  TP_THROW(TypeError, "ctc_loss: lengths must be int32 or int64");
}

} // namespace composite
} // namespace tensorplay
