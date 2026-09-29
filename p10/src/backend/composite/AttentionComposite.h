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

// Window additive mask.  Both bounds are measured from the diagonal running
// from the top left corner to the bottom right one, so query row t sees the
// keys from t + skv - l - left up to but not including t + skv - l + right + 1;
// a negative bound leaves that side unbounded, which is why causal attention is
// the special case (left unbounded, right zero) rather than a separate mask.
inline Tensor window_additive_mask(int64_t l, int64_t skv, int64_t left,
                                   int64_t right, DType dtype,
                                   const Device& device) {
  using ops::add, ops::arange, ops::full, ops::ge, ops::logical_and,
      ops::logical_not, ops::lt, ops::masked_fill, ops::narrow, ops::sub,
      ops::view;
  if (left < 0 && right < 0) {
    // Nothing is excluded, so there is no mask to add.
    return Tensor();
  }
  Tensor idx = arange(Scalar(0), Scalar(std::max(l, skv)), Scalar(1),
                      DType::Int64, device);
  Tensor rows = add(view(narrow(idx, 0, 0, l), {l, 1}), Scalar(skv - l));
  Tensor cols = view(narrow(idx, 0, 0, skv), {1, skv});
  std::optional<Tensor> keep;
  if (left >= 0) keep.emplace(ge(cols, sub(rows, Scalar(left))));
  if (right >= 0) {
    Tensor nearer = lt(cols, add(rows, Scalar(right + 1)));
    keep = keep.has_value() ? logical_and(*keep, nearer) : nearer;
  }
  Tensor zeros = full({l, skv}, Scalar(0), dtype, device);
  return masked_fill(zeros, logical_not(*keep), Scalar(kNegInf));
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
  using ops::view, ops::expand, ops::reshape, ops::unsqueeze;
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
    // (..., Hk, S, D) -> (..., Hk, g, S, D) -> (..., Hq, S, D)
    const auto extents = static_cast<std::vector<int64_t>>(t.shape());
    const int64_t head_axis = static_cast<int64_t>(extents.size()) - 3;
    // A group axis of one is added behind the head axis, and then expanded to
    // g: adding an axis is what unsqueeze is for, and repeating along the new
    // axis is what expand is for.  Reading the head and the group as one axis
    // afterwards is a reshape, since both are there and no axis is added.
    // Which side the group axis sits on decides the mapping, because the
    // reshape reads the two as one axis: behind the head axis the merged axis
    // orders as (Hk, g), so query head h reads key head h / g and one key head
    // serves a contiguous block of g query heads.  Ahead of the head axis the
    // merged axis orders as (g, Hk) and the same query head would instead read
    // key head h % Hk, which is a different function of h.
    Tensor repeated = expand(unsqueeze(t, head_axis + 1), [&] {
      std::vector<int64_t> shape = extents;
      shape.insert(shape.begin() + head_axis + 1, g);
      return shape;
    }());
    std::vector<int64_t> merged = extents;
    merged[static_cast<size_t>(head_axis)] = hq;
    return reshape(repeated, merged);
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
      probs = std::get<0>(ops::native_dropout(probs, dropout_p, true));
    }
  }

  Tensor out = matmul(probs, v);
  if (reduce) {
    return {to(out, origin_dtype), to(probs, origin_dtype)};
  }
  return {out, probs};
}

// The same composite, also reporting the softmax normalizing constant.  The
// scores are already formed to weight the values, and the constant is a
// reduction over that same tensor rather than a second product, so asking for
// it costs one pass and cannot disagree with the probabilities.
//
// A row with no visible key has no softmax: dividing by its zero total would
// put a NaN in the output, which `safe_softmax_lastdim` already avoids.  The
// constant for such a row is reported as positive infinity, because a backward
// pass that reuses it to reweight the scores then gets zeros out of
// exp(score - inf) -- the right derivative for a row that contributes nothing,
// where the opposite sign would hand back a NaN.
inline std::tuple<Tensor, Tensor> sdpa_math_composite_with_lse(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_mask, double dropout_p, bool is_causal,
    const std::optional<Tensor>& dropout_mask, std::optional<double> scale,
    bool enable_gqa) {
  using ops::add, ops::amax, ops::div, ops::eq, ops::exp, ops::full, ops::log,
      ops::matmul, ops::mul, ops::ones_like, ops::sub, ops::sum, ops::to,
      ops::transpose, ops::where, ops::zeros_like;
  const DType origin_dtype = query.dtype();
  if (origin_dtype != DType::Float32 && origin_dtype != DType::Float64 &&
      origin_dtype != DType::Float16 && origin_dtype != DType::BFloat16) {
    TP_THROW(NotImplementedError,
             "sdpa math: expected float32/float64/float16/bfloat16");
  }
  const bool reduce = origin_dtype == DType::Float16 ||
                      origin_dtype == DType::BFloat16;
  const DType working = reduce ? DType::Float32 : origin_dtype;
  Tensor q = reduce ? to(query, working) : query;
  Tensor k = reduce ? to(key, working) : key;
  Tensor v = reduce ? to(value, working) : value;
  std::tie(k, v) = expand_gqa(q, k, v, enable_gqa);
  const int64_t head_dim = q.size(-1);
  const double s = math_scale_factor(scale, head_dim);
  const double sqrt_s = std::sqrt(std::abs(s));
  Tensor scores = matmul(mul(q, Scalar(sqrt_s)), mul(transpose(k, -2, -1), Scalar(sqrt_s)));
  if (is_causal) {
    scores = add(scores, causal_additive_mask(q.size(-2), k.size(-2), q.dtype(), q.device()));
  }
  if (attn_mask.has_value()) {
    const Tensor& m = *attn_mask;
    scores = add(scores, m.dtype() == DType::Bool ? bool_mask_to_additive(m, q.dtype())
                                                   : to(m, q.dtype()));
  }
  // A row whose largest score is minus infinity has nothing to exponentiate
  // against; its reference point is zero and the masked entries still
  // exponentiate to zero, which is what leaves the row's total at zero.
  Tensor row_max = amax(scores, {-1}, /*keepdim=*/true);
  row_max = where(eq(row_max, Scalar(-std::numeric_limits<double>::infinity())),
                  ops::zeros_like(row_max), row_max);
  Tensor shifted = sub(scores, row_max);
  Tensor probs = exp(shifted);
  Tensor total = sum(probs, {-1}, /*keepdim=*/true);
  Tensor empty = eq(total, Scalar(0));
  Tensor normalizer = where(empty, ops::ones_like(total), total);
  // The reductions keep a trailing axis so they broadcast against the scores;
  // the constant does not have one, or every caller would have to know which.
  Tensor lse = where(
      empty.squeeze(-1),
      ops::full({}, Scalar(std::numeric_limits<double>::infinity()),
                DType::Float32, query.device()),
      to(add(row_max, log(total)).squeeze(-1), DType::Float32));
  Tensor out = matmul(div(probs, normalizer), v);
  return {reduce ? to(out, origin_dtype) : out, lse};
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

// The widest head the fused device tiles are shaped for.  A wider head has no
// tile to land in, so it stays on the composed reference.
inline constexpr int64_t kFusedMaxHeadDim = 128;

// Which device-side schedule answers a call, when one does.  Three are named
// because they cover different shapes and precisions, and the caller has to
// pick between them:
//   kSquare    -- one head count, one token count, the head width's own
//                 normaliser: the tensor-core, warp-per-row and GEMM-backed
//                 schedules between them, in every precision each has a kernel
//                 for.  Also the entry the matching fused backward hangs off.
//   kTiled     -- anything carrying an explicit normaliser, grouped heads, or
//                 two lengths.  All three are launch parameters of the tiled
//                 schedule rather than shape rewrites, and it comes in the two
//                 reduced precisions its tiles are cut for.
//   kCrossGemm -- two lengths and nothing else that the tiled schedule would
//                 have to model.  Materializes the score matrix, but it is the
//                 only schedule that answers a wide precision on a shape the
//                 square entry point cannot state.
enum class FusedSdpaSchedule { kNone, kSquare, kTiled, kCrossGemm };

// Which schedule can answer this call, as a shape predicate.  Every schedule
// declines the two things none of them model -- an additive mask and a drop --
// and wants 4-D inputs, one head width shared by query, key and value inside the
// tiled range, matching dtypes, one batch, and a key and value of one length.
inline FusedSdpaSchedule fused_sdpa_schedule(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_mask, double dropout_p,
    const std::optional<double>& scale, bool enable_gqa) {
  if (attn_mask.has_value() || dropout_p != 0.0) return FusedSdpaSchedule::kNone;
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    return FusedSdpaSchedule::kNone;
  }
  const DType dt = query.dtype();
  if (key.dtype() != dt || value.dtype() != dt) return FusedSdpaSchedule::kNone;
  const int64_t head_dim = query.size(3);
  if (head_dim == 0 || head_dim > kFusedMaxHeadDim) {
    return FusedSdpaSchedule::kNone;
  }
  if (key.size(3) != head_dim || value.size(3) != head_dim) {
    return FusedSdpaSchedule::kNone;
  }
  if (key.size(0) != query.size(0) || value.size(0) != query.size(0)) {
    return FusedSdpaSchedule::kNone;
  }
  if (query.size(2) == 0 || key.size(2) == 0) {
    return FusedSdpaSchedule::kNone;
  }
  if (key.size(2) != value.size(2)) return FusedSdpaSchedule::kNone;
  const int64_t hq = query.size(1), hk = key.size(1);
  if (hq == 0 || hk == 0) return FusedSdpaSchedule::kNone;
  const bool grouped = hq != hk;
  // Grouped heads that do not divide evenly are not a shape; leaving them out
  // of every schedule leaves the composed path to say so in its own terms.
  if (enable_gqa ? (hq % hk != 0) : grouped) return FusedSdpaSchedule::kNone;

  const bool square = !grouped && query.size(2) == key.size(2);
  const bool reduced = dt == DType::Float16 || dt == DType::BFloat16;
  if (square && !scale.has_value()) {
    const bool any_precision = dt == DType::Float32 || reduced;
    return any_precision ? FusedSdpaSchedule::kSquare : FusedSdpaSchedule::kNone;
  }
  // The tiled schedule takes the normaliser, the group ratio and the two
  // lengths as launch parameters, so it covers every other shape that has
  // tiles -- which is the two reduced precisions.
  if (reduced) return FusedSdpaSchedule::kTiled;
  // A wide precision on a shape the tiled schedule has no tiles for.  The
  // GEMM-backed schedule materializes the scores and answers it, but it names
  // no normaliser, so a call that carries one stays on the composed path.
  const bool wide = dt == DType::Float32 && !scale.has_value();
  return wide ? FusedSdpaSchedule::kCrossGemm : FusedSdpaSchedule::kNone;
}

// Backend selection shared with the nn.attention routing flags:
// FLASH_ATTENTION(1) covers the fused case, MATH(0) the rest, ERROR(-1)
// when nothing can run.
inline int64_t fused_sdp_choice_common(const Tensor& query, const Tensor& key,
                                       const Tensor& value,
                                       const std::optional<Tensor>& attn_mask,
                                       double dropout_p,
                                       std::optional<double> scale,
                                       bool enable_gqa) {
  if (fused_sdpa_schedule(query, key, value, attn_mask, dropout_p, scale,
                          enable_gqa) != FusedSdpaSchedule::kNone) {
    return 1;  // FLASH_ATTENTION
  }
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

// The gradient of the attention above, in terms of the same operations the
// attention itself is in.  Writing it this way rather than as a second set of
// hand-rolled kernels is what keeps a masked, dropped, scaled or grouped call
// differentiable by the same argument as a plain one: the scores are taken
// again here, so whatever was added to them, and whatever was folded into the
// query, is accounted for by the same arithmetic that produced them.
//
// With S = softmax(scores) and O = S @ V, the gradients are
//   dS = grad_out @ V^T,  dV = S^T @ grad_out,
//   d(scores) = S * (dS - rowsum(dS * S)),  scaled back onto the query by the
//   same factor the scores were scaled by.
inline std::tuple<Tensor, Tensor, Tensor> sdpa_math_backward_composite(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const std::optional<Tensor>& attn_mask,
    double dropout_p, bool is_causal, const std::optional<double>& scale,
    bool enable_gqa) {
  using ops::matmul, ops::mul, ops::transpose, ops::sum, ops::where, ops::to,
      ops::add, ops::reshape, ops::sub;

  // The key and value are given as many heads as the query has, by the same
  // expansion the forward does, so that both directions agree on which head
  // each query head reads.
  auto [k, v] = expand_gqa(query, key, value, enable_gqa);
  const int64_t hk = key.dim() >= 3 ? key.size(-3) : 1;
  const int64_t hq = query.dim() >= 3 ? query.size(-3) : 1;
  const int64_t g = (enable_gqa && hq != hk) ? hq / hk : 1;

  double s = math_scale_factor(scale, query.size(-1));
  const bool half_to_float = query.dtype() == DType::Float16 ||
                             query.dtype() == DType::BFloat16;
  const DType compute = half_to_float ? DType::Float32 : query.dtype();
  Tensor q = query.to(compute), kk = k.to(compute), vv = v.to(compute);
  Tensor go = grad_output.to(compute);

  Tensor scores = matmul(mul(q, Scalar(s)), transpose(kk, -2, -1));
  if (is_causal) {
    // The same top-left alignment the forward masks with: query row t sees
    // keys <= t, whatever the two lengths are.  A gradient has to be
    // differentiated through the mask the forward actually applied, so this
    // one is built by that same helper rather than spelled out again here.
    scores = add(scores, causal_additive_mask(scores.size(-2), scores.size(-1),
                                              scores.dtype(), scores.device()));
  }
  if (attn_mask.has_value() && attn_mask->defined()) {
    const Tensor& m = *attn_mask;
    scores = (m.dtype() == DType::Bool) ? where(m, scores, Scalar(kNegInf))
                                         : add(scores, m.to(compute));
  }
  // The same all--inf handling the forward's softmax has, so a fully masked
  // query row is a row of zeros here too rather than a number over nothing.
  Tensor probs = safe_softmax_lastdim(scores);

  Tensor d_probs = matmul(go, transpose(vv, -2, -1));
  Tensor d_value = matmul(transpose(probs, -2, -1), go);
  Tensor d_scores = mul(probs, sub(d_probs,
                                   sum(mul(d_probs, probs), {-1}, /*keepdim=*/true)));
  Tensor d_query = mul(matmul(d_scores, kk), Scalar(s));
  Tensor d_key = matmul(transpose(d_scores, -2, -1), mul(q, Scalar(s)));

  if (g > 1) {
    // The key and value each served g query heads, so the gradients arriving
    // at them are summed over the group axis they were expanded along.
    d_key = sum(d_key.reshape({d_key.size(0), hk, g, d_key.size(-2), d_key.size(-1)}), {2});
    d_value = sum(d_value.reshape({d_value.size(0), hk, g, d_value.size(-2), d_value.size(-1)}), {2});
  }
  return {d_query.to(query.dtype()), d_key.to(key.dtype()),
          d_value.to(value.dtype())};
}

} // namespace composite
} // namespace tensorplay

