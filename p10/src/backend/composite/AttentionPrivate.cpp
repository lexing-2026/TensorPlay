// Composite registrations of the attention contracts in AttentionPrivate.h.
// They serve every backend; a backend with a fused kernel of its own registers
// it under its key and answers first.

#include "AttentionPrivate.h"
#include "Dispatcher.h"

namespace tensorplay {
namespace composite {
namespace attention {
namespace {

std::tuple<Tensor, Tensor, Tensor, Tensor, SymInt, SymInt, Tensor, Tensor, Tensor>
sdpa_flash_quantized(const Tensor&, const Tensor&, const Tensor&,
                     const std::optional<Tensor>&, const std::optional<Tensor>&,
                     const std::optional<Tensor>&, double, bool, bool,
                     std::optional<double>) {
  // Scores formed from eight-bit inputs keep the rounding that precision
  // implies; widening the inputs first would land somewhere else, so there is
  // nothing to compose here.
  TP_THROW(NotImplementedError,
           "low-precision flash attention needs a low-precision attention "
           "kernel; register one with "
           "tensorplay.nn.attention.activate_flash_attention_impl(\"FA3\")");
}

std::tuple<Tensor, Tensor, Tensor, Tensor, SymInt, SymInt, Tensor, Tensor, Tensor>
sdpa_fused_overrideable(const Tensor&, const Tensor&, const Tensor&,
                        const std::optional<Tensor>&, double, bool, bool,
                        std::optional<double>) {
  TP_THROW(NotImplementedError,
           "_scaled_dot_product_fused_attention_overrideable has no "
           "implementation of its own: it is the entry point a backend "
           "registers its fused attention kernel under");
}

std::tuple<Tensor, Tensor, Tensor, Tensor> sdpa_fused_overrideable_backward(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const std::vector<bool>&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, int64_t, int64_t, double, bool, const Tensor&, const Tensor&,
    std::optional<double>) {
  TP_THROW(NotImplementedError,
           "_scaled_dot_product_fused_attention_overrideable_backward has no "
           "implementation of its own: it is the entry point a backend "
           "registers its fused attention backward under");
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(Composite, AttentionPrivate) {
  m.impl("_flash_attention_forward", flash_forward);
  m.impl("_flash_attention_forward_no_dropout_inplace", flash_forward_into);
  m.impl("_flash_attention_backward", flash_backward);
  m.impl("_efficient_attention_forward", efficient_forward);
  m.impl("_efficient_attention_backward", efficient_backward);
  m.impl("_cudnn_attention_forward", cudnn_forward);
  m.impl("_cudnn_attention_backward", cudnn_backward);
  m.impl("_scaled_dot_product_flash_attention", sdpa_flash);
  m.impl("_scaled_dot_product_flash_attention.quantized", sdpa_flash_quantized);
  m.impl("_scaled_dot_product_flash_attention_backward", sdpa_flash_backward);
  m.impl("_scaled_dot_product_efficient_attention", sdpa_efficient);
  m.impl("_scaled_dot_product_efficient_attention_backward",
         sdpa_efficient_backward);
  m.impl("_scaled_dot_product_cudnn_attention", sdpa_cudnn);
  m.impl("_scaled_dot_product_cudnn_attention_backward", sdpa_cudnn_backward);
  m.impl("_scaled_dot_product_fused_attention_overrideable",
         sdpa_fused_overrideable);
  m.impl("_scaled_dot_product_fused_attention_overrideable_backward",
         sdpa_fused_overrideable_backward);
}

}  // namespace attention
}  // namespace composite
}  // namespace tensorplay
