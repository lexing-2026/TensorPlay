#pragma once
// oneDNN sequence-level LSTM backward (CPU, fp32).
//
// Declared here (p10) so the replay-based RNN backward node
// (tpx/include/RNNBackward.h) can route the lstm kind through oneDNN's fused
// lstm_backward primitive instead of the decomposed per-step sweep.  Defined in
// backend/cpu/Tier5OpsKernels.cpp next to onednn_lstm_forward.
//
// lstm_backward primitive per layer+direction consuming the forward workspace.
// Because TensorPlay's autograd node only saves (input, hx, params) -- the
// generated wrapper does not thread the forward workspace -- this re-runs the
// oneDNN forward (forward_training) per layer+direction to regenerate the
// workspace and forward outputs, then runs the fused backward.  Same replay
// structure as the native path, but both the replay and the sweep run as
// oneDNN primitives.
//
// Returns (grad_input, {grad_hx, grad_cx}, grad_params) to match
// rnn_backward_impl, or std::nullopt to fall back to the decomposed replay
// when oneDNN is unavailable/disabled or the shape/dtype is unsupported.

#include "Tensor.h"

#include <optional>
#include <tuple>
#include <vector>
#include <cstdint>

namespace tensorplay {
namespace cpu {

std::optional<std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>>
onednn_lstm_backward(const Tensor& grad_y, const Tensor& grad_hy,
                     const Tensor& grad_cy, const Tensor& input,
                     const std::vector<Tensor>& hx,
                     const std::vector<Tensor>& params, bool has_biases,
                     int64_t num_layers, bool bidirectional, bool batch_first);

} // namespace cpu
} // namespace tensorplay
