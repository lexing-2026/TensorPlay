#pragma once
// Second derivatives through the scaled-dot-product attention backwards.
//
// Both backward spellings express one function at dropout_p == 0.  With
// S = s Q K^T (plus the causal/mask terms) and P = softmax(S):
//   dP = gO V^T,  dV = P^T gO,  dS = P (dP - rowsum(dP P)),
//   dQ = s dS K,  dK = s dS^T Q.
// The log-sum-exp spelling folds the scale into dS and reads the row
// statistic delta = rowsum(gO O) -- the same quantity rowsum(dP P), taken
// from the saved output -- and rebuilds P as exp(S - lse).
//
// The nodes below differentiate that function once more with respect to the
// backward's own inputs (the incoming gradient, the operands, the mask and,
// for the log-sum-exp spelling, the saved output and constant).  Every step
// composes recordable dispatcher primitives, so a pass with create_graph
// records and a third pass can differentiate again.  A backward that dropped
// probabilities has no second derivative: the dropped entries' Jacobian is
// the drop mask itself and a pass differentiating through one raises.

#include "Node.h"
#include "Autograd.h"
#include "SavedVariable.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tensorplay {
namespace tpx {

namespace sdpa_bwd_detail {

using tensorplay::tpx::ops::sum;

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// The additive causal mask the forward scores carry: query row t sees keys
// up to and including t, top-left aligned whatever the two lengths are.
inline Tensor causal_additive_mask(int64_t l, int64_t skv, DType dtype,
                                   const Device& device) {
    using ops::arange, ops::full, ops::ge, ops::logical_not, ops::masked_fill,
        ops::narrow, ops::view;
    Tensor idx = arange(Scalar(0), Scalar(std::max(l, skv)), Scalar(1),
                        DType::Int64, device);
    Tensor keep = ge(view(narrow(idx, 0, 0, l), {l, 1}),
                     view(narrow(idx, 0, 0, skv), {1, skv}));
    Tensor zeros = full({l, skv}, Scalar(0), dtype, device);
    return masked_fill(zeros, logical_not(keep), Scalar(kNegInf));
}

// Softmax over the last dim that yields an all-zero row instead of NaN when
// every entry of the row is -inf (fully masked query positions).
inline Tensor safe_softmax_lastdim(const Tensor& scores) {
    using ops::amax, ops::div, ops::eq, ops::exp, ops::sub, ops::sum, ops::where;
    Tensor row_max = amax(scores, {-1}, true);
    // -inf rows would poison exp(x - max); shift them to a finite pivot.
    Tensor finite_max = where(eq(row_max, Scalar(kNegInf)), Scalar(0), row_max);
    Tensor e = exp(sub(scores, finite_max));
    Tensor denom = sum(e, {-1}, true);
    Tensor probs = div(e, denom);
    return where(eq(denom, Scalar(0)), Scalar(0), probs);
}

// (B, Hkv, T, D) -> (B, Hq, T, D): each kv head serves a contiguous block of
// g query heads, the expansion the forward applies for grouped calls.
inline Tensor expand_kv_heads(const Tensor& t, int64_t hq) {
    using ops::expand, ops::reshape, ops::unsqueeze;
    const auto extents = static_cast<std::vector<int64_t>>(t.shape());
    const int64_t rank = static_cast<int64_t>(extents.size());
    const int64_t hk = extents[static_cast<size_t>(rank - 3)];
    const int64_t g = hq / hk;
    std::vector<int64_t> group_shape = extents;
    group_shape.insert(group_shape.begin() + (rank - 2), g);
    Tensor repeated = expand(unsqueeze(t, rank - 2), group_shape);
    std::vector<int64_t> merged = extents;
    merged[static_cast<size_t>(rank - 3)] = hq;
    return reshape(repeated, merged);
}

// The adjoint of that expansion followed by the group sum the backward closes
// with: the op sums each kv head's gradient over the g query heads it served,
// so an adjoint of the sum is the same gradient on every group member, and a
// gradient of the expanded algebra collapses back along the group axis.
inline Tensor sum_group_heads(const Tensor& t, int64_t hk, int64_t g) {
    using ops::reshape, ops::sum;
    if (g == 1) return t;
    const auto extents = static_cast<std::vector<int64_t>>(t.shape());
    std::vector<int64_t> group_shape = extents;
    group_shape[static_cast<size_t>(group_shape.size() - 3)] = hk;
    group_shape.insert(group_shape.begin() + (group_shape.size() - 2), g);
    return sum(reshape(t, group_shape), {2}, false);
}

// d <A, softmax(x)> / dx over the last dim, with fully masked rows (zero
// probability) left at zero.
inline Tensor softmax_backward_transposed(const Tensor& adj, const Tensor& probs) {
    using ops::mul, ops::sub, ops::sum;
    return mul(probs, sub(adj, sum(mul(adj, probs), {-1}, true)));
}

// The adjoint set of the math spelling: one gradient per tensor input of
// scaled_dot_product_attention_backward, in schema order.  grad_mask stays
// undefined when the mask is absent or takes no gradient.
struct MathAdjoints {
    Tensor grad_output;
    Tensor grad_query;
    Tensor grad_key;
    Tensor grad_value;
    Tensor grad_mask;
};

// The double backward of the composed math backward.  Recomputes the scores
// and probabilities from the operands (the backward itself did), then walks
// the algebra above backwards with the incoming adjoints.
inline MathAdjoints sdpa_math_double_backward(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const std::optional<Tensor>& attn_mask, bool is_causal,
    std::optional<double> scale, bool enable_gqa, const Tensor& adj_query,
    const Tensor& adj_key, const Tensor& adj_value) {
    using ops::add, ops::matmul, ops::mul, ops::softmax, ops::sub,
        ops::sum_to_size, ops::transpose, ops::where, ops::zeros_like;
    const DType origin = query.dtype();
    const bool half = origin == DType::Float16 || origin == DType::BFloat16;
    const DType compute = half ? DType::Float32 : origin;
    // The cast records, so a third pass sees through the widened compute.
    auto widen = [&](const Tensor& t) {
        return t.dtype() == compute ? t : to(t, compute);
    };
    Tensor q = widen(query);
    Tensor k = widen(key);
    Tensor v = widen(value);
    Tensor go = widen(grad_output);
    Tensor ggq = adj_query.defined() ? widen(adj_query) : zeros_like(q);
    Tensor ggk = adj_key.defined() ? widen(adj_key) : zeros_like(k);
    Tensor ggv = adj_value.defined() ? widen(adj_value) : zeros_like(v);

    const int64_t hq = query.size(1), hk = key.size(1);
    const int64_t g = (enable_gqa && hq != hk) ? hq / hk : 1;
    if (g > 1) {
        k = expand_kv_heads(k, hq);
        v = expand_kv_heads(v, hq);
        ggk = expand_kv_heads(ggk, hq);
        ggv = expand_kv_heads(ggv, hq);
    }

    const double s = scale.has_value()
                         ? *scale
                         : 1.0 / std::sqrt(static_cast<double>(query.size(-1)));
    Tensor scores = matmul(mul(q, Scalar(s)), transpose(k, -2, -1));
    const bool masked = is_causal || (attn_mask.has_value() && attn_mask->defined());
    if (is_causal) {
        scores = add(scores, causal_additive_mask(scores.size(-2), scores.size(-1),
                                                  scores.dtype(), scores.device()));
    }
    if (attn_mask.has_value() && attn_mask->defined()) {
        const Tensor& m = *attn_mask;
        scores = m.dtype() == DType::Bool ? where(m, scores, Scalar(kNegInf))
                                          : add(scores, widen(m));
    }
    Tensor probs = masked ? safe_softmax_lastdim(scores)
                          : softmax(scores, -1, DType::Undefined);

    Tensor d_probs = matmul(go, transpose(v, -2, -1));
    Tensor row_total = sum(mul(d_probs, probs), {-1}, true);
    Tensor score_grad = mul(probs, sub(d_probs, row_total));
    Tensor score_adj = mul(add(matmul(ggq, transpose(k, -2, -1)),
                               matmul(q, transpose(ggk, -2, -1))),
                           Scalar(s));
    Tensor share = sum(mul(score_adj, probs), {-1}, true);
    Tensor d_probs_adj = mul(probs, sub(score_adj, share));
    Tensor probs_adj = add(matmul(go, transpose(ggv, -2, -1)),
                           sub(mul(score_adj, sub(d_probs, row_total)),
                               mul(share, d_probs)));
    Tensor scores_adj = softmax_backward_transposed(probs_adj, probs);

    MathAdjoints out;
    out.grad_output = add(matmul(probs, ggv), matmul(d_probs_adj, v));
    out.grad_query = mul(add(matmul(score_grad, ggk), matmul(scores_adj, k)),
                         Scalar(s));
    out.grad_key = mul(add(matmul(transpose(score_grad, -2, -1), ggq),
                           matmul(transpose(scores_adj, -2, -1), q)),
                       Scalar(s));
    out.grad_value = matmul(transpose(d_probs_adj, -2, -1), go);
    if (g > 1) {
        out.grad_key = sum_group_heads(out.grad_key, hk, g);
        out.grad_value = sum_group_heads(out.grad_value, hk, g);
    }
    if (attn_mask.has_value() && attn_mask->defined() &&
        attn_mask->dtype() != DType::Bool) {
        // The additive mask entered the scores verbatim; a bool mask only
        // selects positions and takes no gradient of its own.
        out.grad_mask = to(sum_to_size(scores_adj, attn_mask->sizes()),
                           attn_mask->dtype());
    }
    out.grad_output = to(std::move(out.grad_output), grad_output.dtype());
    out.grad_query = to(std::move(out.grad_query), origin);
    out.grad_key = to(std::move(out.grad_key), key.dtype());
    out.grad_value = to(std::move(out.grad_value), value.dtype());
    return out;
}

// The log-sum-exp spelling only reads the constant and the saved output when
// they line up with the self-attention shape its kernel serves; anything else
// ran the composed math, which reads neither.
inline bool lse_serves(const Tensor& grad_output, const Tensor& query,
                       const Tensor& key, const Tensor& value,
                       const Tensor& output, const Tensor& logsumexp) {
    const DType dt = query.dtype();
    if (dt != DType::Float32 && dt != DType::Float16 && dt != DType::BFloat16) {
        return false;
    }
    if (key.dtype() != dt || value.dtype() != dt || output.dtype() != dt ||
        grad_output.dtype() != dt) {
        return false;
    }
    if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4 ||
        output.dim() != 4 || grad_output.dim() != 4) {
        return false;
    }
    const auto qs = static_cast<std::vector<int64_t>>(query.shape());
    if (static_cast<std::vector<int64_t>>(key.shape()) != qs ||
        static_cast<std::vector<int64_t>>(value.shape()) != qs ||
        static_cast<std::vector<int64_t>>(output.shape()) != qs ||
        static_cast<std::vector<int64_t>>(grad_output.shape()) != qs) {
        return false;
    }
    if (qs[3] <= 0) return false;
    if (logsumexp.dtype() != DType::Float32 || !logsumexp.is_contiguous()) {
        return false;
    }
    return logsumexp.numel() == qs[0] * qs[1] * qs[2];
}

// The adjoint set of the log-sum-exp spelling: the backward's tensor inputs
// in schema order, including the saved output and the constant.
struct LseAdjoints {
    Tensor grad_output;
    Tensor grad_query;
    Tensor grad_key;
    Tensor grad_value;
    Tensor grad_out;
    Tensor grad_lse;
};

// The double backward of the log-sum-exp backward.  The probabilities come
// back as exp(s Q K^T - lse) with causal entries forced to zero, the row
// statistic is read off the saved output the way the kernel read it, and the
// constant's own adjoint collects how every probability leans on it.
inline LseAdjoints sdpa_lse_double_backward(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& output, const Tensor& logsumexp,
    bool is_causal, const Tensor& adj_query, const Tensor& adj_key,
    const Tensor& adj_value) {
    using ops::add, ops::arange, ops::exp, ops::ge, ops::matmul, ops::mul,
        ops::reshape, ops::sub, ops::sum, ops::transpose, ops::view,
        ops::where, ops::zeros_like;
    // The serving kernel widens the reduced precisions and keeps every
    // accumulation in float32; the constant is float32 throughout.
    const DType origin = query.dtype();
    auto widen = [&](const Tensor& x) {
        return x.dtype() == DType::Float32 ? x : to(x, DType::Float32);
    };
    Tensor q = widen(query);
    Tensor k = widen(key);
    Tensor v = widen(value);
    Tensor go = widen(grad_output);
    Tensor o = widen(output);
    Tensor ggq = adj_query.defined() ? widen(adj_query) : zeros_like(q);
    Tensor ggk = adj_key.defined() ? widen(adj_key) : zeros_like(k);
    Tensor ggv = adj_value.defined() ? widen(adj_value) : zeros_like(v);

    const int64_t b = query.size(0), h = query.size(1), t = query.size(2),
                  d = query.size(3);
    const double s = 1.0 / std::sqrt(static_cast<double>(d));
    Tensor lse = reshape(logsumexp, {b, h, t, 1});
    Tensor scores = matmul(mul(q, Scalar(s)), transpose(k, -2, -1));
    Tensor probs = exp(sub(scores, lse));
    if (is_causal) {
        // Keys past the query row have no probability; the kernel zeroes
        // them rather than exponentiating a score it never formed.
        Tensor idx = arange(Scalar(0), Scalar(t), Scalar(1), DType::Int64,
                            query.device());
        Tensor keep = ge(view(idx, {1, 1, t, 1}), view(idx, {1, 1, 1, t}));
        probs = where(keep, probs, Scalar(0));
    }
    Tensor d_probs = matmul(go, transpose(v, -2, -1));
    Tensor delta = sum(mul(go, o), {-1}, true);
    Tensor score_grad = mul(probs, sub(d_probs, delta));
    Tensor score_adj = mul(add(matmul(ggq, transpose(k, -2, -1)),
                               matmul(q, transpose(ggk, -2, -1))),
                           Scalar(s));
    Tensor share = sum(mul(score_adj, probs), {-1}, true);
    // The statistic is a constant of the probability algebra here, so the
    // probability gradient misses the row-total term the softmax spelling
    // carries and the output gradient picks it up instead.
    Tensor d_probs_adj = mul(probs, score_adj);
    Tensor probs_adj = add(matmul(go, transpose(ggv, -2, -1)),
                           mul(score_adj, sub(d_probs, delta)));
    Tensor scores_adj = mul(probs_adj, probs);

    LseAdjoints out;
    out.grad_output = sub(add(matmul(probs, ggv), matmul(d_probs_adj, v)),
                          mul(o, share));
    out.grad_query = mul(add(matmul(score_grad, ggk), matmul(scores_adj, k)),
                         Scalar(s));
    out.grad_key = mul(add(matmul(transpose(score_grad, -2, -1), ggq),
                           matmul(transpose(scores_adj, -2, -1), q)),
                       Scalar(s));
    out.grad_value = matmul(transpose(d_probs_adj, -2, -1), go);
    out.grad_out = mul(go, mul(share, Scalar(-1)));
    out.grad_lse = mul(sum(mul(probs_adj, probs), {-1}, false), Scalar(-1));
    out.grad_output = to(std::move(out.grad_output), grad_output.dtype());
    out.grad_query = to(std::move(out.grad_query), origin);
    out.grad_key = to(std::move(out.grad_key), key.dtype());
    out.grad_value = to(std::move(out.grad_value), value.dtype());
    out.grad_out = to(std::move(out.grad_out), output.dtype());
    return out;
}

}  // namespace sdpa_bwd_detail

// The derivative of scaled_dot_product_attention_backward: gradients for the
// incoming gradient, the query, the key, the value and the mask, in schema
// order. dropout_p == 0 only; a call that dropped probabilities has no
// second derivative and a pass differentiating through one raises.
struct ScaledDotProductAttentionBackwardBackward : public Node {
    SavedVariable grad_output_;
    SavedVariable query_;
    SavedVariable key_;
    SavedVariable value_;
    std::optional<Tensor> attn_mask_;
    double dropout_p_;
    bool is_causal_;
    std::optional<double> scale_;
    bool enable_gqa_;

    ScaledDotProductAttentionBackwardBackward(
        Tensor grad_output, Tensor query, Tensor key, Tensor value,
        std::optional<Tensor> attn_mask, double dropout_p, bool is_causal,
        std::optional<double> scale, bool enable_gqa)
        : grad_output_(std::move(grad_output)), query_(std::move(query)),
          key_(std::move(key)), value_(std::move(value)),
          attn_mask_(std::move(attn_mask)), dropout_p_(dropout_p),
          is_causal_(is_causal), scale_(scale), enable_gqa_(enable_gqa) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        const Tensor adj_query = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor adj_key = inputs.size() > 1 ? inputs[1] : Tensor();
        const Tensor adj_value = inputs.size() > 2 ? inputs[2] : Tensor();
        variable_list grads(5);
        if (!adj_query.defined() && !adj_key.defined() && !adj_value.defined()) {
            return grads;
        }
        if (dropout_p_ != 0.0) {
            TP_THROW(NotImplementedError,
                     "scaled_dot_product_attention_backward: the second "
                     "derivative is not defined when dropout_p != 0");
        }
        const Tensor go = grad_output_.unpack();
        const Tensor q = query_.unpack();
        const Tensor k = key_.unpack();
        const Tensor v = value_.unpack();
        if (!go.defined() || !q.defined() || !k.defined() || !v.defined()) {
            return grads;
        }
        const sdpa_bwd_detail::MathAdjoints r =
            sdpa_bwd_detail::sdpa_math_double_backward(
                go, q, k, v, attn_mask_, is_causal_, scale_, enable_gqa_,
                adj_query, adj_key, adj_value);
        if (should_compute_output(0)) grads[0] = r.grad_output;
        if (should_compute_output(1)) grads[1] = r.grad_query;
        if (should_compute_output(2)) grads[2] = r.grad_key;
        if (should_compute_output(3)) grads[3] = r.grad_value;
        if (should_compute_output(4)) grads[4] = r.grad_mask;
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        grad_output_.reset_data();
        query_.reset_data();
        key_.reset_data();
        value_.reset_data();
        attn_mask_.reset();
    }
};

// The derivative of _scaled_dot_product_flash_attention_for_cpu_backward:
// gradients for the incoming gradient, the query, the key, the value, the
// saved output, the constant and the mask, in schema order.  The composed
// math recomputes the probabilities from the operands, so the saved output
// and the constant -- both functions of those operands -- take none.
// dropout_p == 0 only, as for the other spellings.
struct ScaledDotProductFlashAttentionForCpuBackwardBackward : public Node {
    SavedVariable grad_out_;
    SavedVariable query_;
    SavedVariable key_;
    SavedVariable value_;
    double dropout_p_;
    bool is_causal_;
    std::optional<Tensor> attn_mask_;
    std::optional<double> scale_;

    ScaledDotProductFlashAttentionForCpuBackwardBackward(
        Tensor grad_out, Tensor query, Tensor key, Tensor value, double dropout_p,
        bool is_causal, std::optional<Tensor> attn_mask, std::optional<double> scale)
        : grad_out_(std::move(grad_out)), query_(std::move(query)),
          key_(std::move(key)), value_(std::move(value)), dropout_p_(dropout_p),
          is_causal_(is_causal), attn_mask_(std::move(attn_mask)), scale_(scale) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        const Tensor adj_query = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor adj_key = inputs.size() > 1 ? inputs[1] : Tensor();
        const Tensor adj_value = inputs.size() > 2 ? inputs[2] : Tensor();
        variable_list grads(7);
        if (!adj_query.defined() && !adj_key.defined() && !adj_value.defined()) {
            return grads;
        }
        if (dropout_p_ != 0.0) {
            TP_THROW(NotImplementedError,
                     "_scaled_dot_product_flash_attention_for_cpu_backward: the "
                     "second derivative is not defined when dropout_p != 0");
        }
        const Tensor go = grad_out_.unpack();
        const Tensor q = query_.unpack();
        const Tensor k = key_.unpack();
        const Tensor v = value_.unpack();
        if (!go.defined() || !q.defined() || !k.defined() || !v.defined()) {
            return grads;
        }
        const sdpa_bwd_detail::MathAdjoints r =
            sdpa_bwd_detail::sdpa_math_double_backward(
                go, q, k, v, attn_mask_, is_causal_, scale_,
                /*enable_gqa=*/q.size(1) != k.size(1), adj_query, adj_key, adj_value);
        if (should_compute_output(0)) grads[0] = r.grad_output;
        if (should_compute_output(1)) grads[1] = r.grad_query;
        if (should_compute_output(2)) grads[2] = r.grad_key;
        if (should_compute_output(3)) grads[3] = r.grad_value;
        if (should_compute_output(6)) grads[6] = r.grad_mask;
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        grad_out_.reset_data();
        query_.reset_data();
        key_.reset_data();
        value_.reset_data();
        attn_mask_.reset();
    }
};

// The derivative of _scaled_dot_product_attention_backward_with_lse:
// gradients for the incoming gradient, the query, the key, the value, the
// saved output and the constant, in schema order.  The probabilities the
// serving kernel used come back from the constant; shapes it did not serve
// ran the composed math, which reads neither the saved output nor the
// constant, so those two take none.
struct ScaledDotProductAttentionBackwardWithLseBackward : public Node {
    SavedVariable grad_output_;
    SavedVariable query_;
    SavedVariable key_;
    SavedVariable value_;
    SavedVariable output_;
    SavedVariable logsumexp_;
    bool is_causal_;
    int64_t impl_;

    ScaledDotProductAttentionBackwardWithLseBackward(
        Tensor grad_output, Tensor query, Tensor key, Tensor value,
        Tensor output, Tensor logsumexp, bool is_causal, int64_t impl)
        : grad_output_(std::move(grad_output)), query_(std::move(query)),
          key_(std::move(key)), value_(std::move(value)),
          output_(std::move(output)), logsumexp_(std::move(logsumexp)),
          is_causal_(is_causal), impl_(impl) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        const Tensor adj_query = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor adj_key = inputs.size() > 1 ? inputs[1] : Tensor();
        const Tensor adj_value = inputs.size() > 2 ? inputs[2] : Tensor();
        variable_list grads(6);
        if (!adj_query.defined() && !adj_key.defined() && !adj_value.defined()) {
            return grads;
        }
        const Tensor go = grad_output_.unpack();
        const Tensor q = query_.unpack();
        const Tensor k = key_.unpack();
        const Tensor v = value_.unpack();
        const Tensor o = output_.unpack();
        const Tensor lse = logsumexp_.unpack();
        if (!go.defined() || !q.defined() || !k.defined() || !v.defined() ||
            !o.defined() || !lse.defined()) {
            return grads;
        }
        if (impl_ == 0 &&
            sdpa_bwd_detail::lse_serves(go, q, k, v, o, lse)) {
            const sdpa_bwd_detail::LseAdjoints r =
                sdpa_bwd_detail::sdpa_lse_double_backward(
                    go, q, k, v, o, lse, is_causal_, adj_query, adj_key,
                    adj_value);
            if (should_compute_output(0)) grads[0] = r.grad_output;
            if (should_compute_output(1)) grads[1] = r.grad_query;
            if (should_compute_output(2)) grads[2] = r.grad_key;
            if (should_compute_output(3)) grads[3] = r.grad_value;
            if (should_compute_output(4)) grads[4] = r.grad_out;
            if (should_compute_output(5)) grads[5] = r.grad_lse;
            return grads;
        }
        // The composed math the op fell back to reads neither the saved
        // output nor the constant.
        const sdpa_bwd_detail::MathAdjoints r =
            sdpa_bwd_detail::sdpa_math_double_backward(
                go, q, k, v, std::nullopt, is_causal_, std::nullopt, false,
                adj_query, adj_key, adj_value);
        if (should_compute_output(0)) grads[0] = r.grad_output;
        if (should_compute_output(1)) grads[1] = r.grad_query;
        if (should_compute_output(2)) grads[2] = r.grad_key;
        if (should_compute_output(3)) grads[3] = r.grad_value;
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        grad_output_.reset_data();
        query_.reset_data();
        key_.reset_data();
        value_.reset_data();
        output_.reset_data();
        logsumexp_.reset_data();
    }
};

}  // namespace tpx
}  // namespace tensorplay
