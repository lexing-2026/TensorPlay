// Scaled dot-product attention, composed schedule, MACA lane.
//
// The fused schedules live in AttentionKernels.cu, whose kernels ride on the
// tensor-core surface this toolchain does not ship, so that file is left out
// of the build and the dispatcher keeps no CUDA entry for the attention
// operators.  This translation unit fills those slots with the composed
// schedule: the same matmul / mask / softmax walk the CPU backend runs, built
// from recordable primitives, so attention and its backward stay available
// and differentiable.
//
// The entry points answer the fused signatures the callers already use, so no
// caller has to know which schedule ran.  The log-sum-exp variant reports the
// same normalizer the composed walk computes; the backward recomputes the
// scores from the query, key and value it is given rather than reading the
// forward's workspace.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "../composite/AttentionComposite.h"

#include <optional>
#include <tuple>

namespace tensorplay {
namespace cuda {

namespace {

Tensor sdpa_composed(const Tensor& query, const Tensor& key, const Tensor& value,
                     const std::optional<Tensor>& attn_mask, double dropout_p,
                     bool is_causal, std::optional<double> scale,
                     bool enable_gqa) {
  return std::get<0>(composite::sdpa_math_composite(
      query, key, value, attn_mask, dropout_p, is_causal, std::nullopt, scale,
      enable_gqa));
}

std::tuple<Tensor, Tensor> sdpa_composed_with_lse(
    const Tensor& query, const Tensor& key, const Tensor& value, bool is_causal,
    int64_t impl) {
  (void)impl;
  return composite::sdpa_math_composite_with_lse(
      query, key, value, std::nullopt, 0.0, is_causal, std::nullopt,
      std::nullopt, false);
}

std::tuple<Tensor, Tensor, Tensor> sdpa_composed_backward(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const std::optional<Tensor>& attn_mask,
    double dropout_p, bool is_causal, std::optional<double> scale,
    bool enable_gqa) {
  return composite::sdpa_math_backward_composite(
      grad_output, query, key, value, attn_mask, dropout_p, is_causal, scale,
      enable_gqa);
}

std::tuple<Tensor, Tensor, Tensor> sdpa_composed_backward_with_lse(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& output, const Tensor& logsumexp,
    bool is_causal, int64_t impl) {
  (void)output;
  (void)logsumexp;
  (void)impl;
  return composite::sdpa_math_backward_composite(
      grad_output, query, key, value, std::nullopt, 0.0, is_causal,
      std::nullopt, false);
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SdpaComposed) {
  m.impl("scaled_dot_product_attention", sdpa_composed);
  m.impl("scaled_dot_product_attention_backward", sdpa_composed_backward);
  m.impl("_scaled_dot_product_attention_with_lse", sdpa_composed_with_lse);
  m.impl("_scaled_dot_product_attention_backward_with_lse",
         sdpa_composed_backward_with_lse);
}

}  // namespace cuda
}  // namespace tensorplay
