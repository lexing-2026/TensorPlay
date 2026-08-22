#pragma once

// Fused RNN cell primitives ported from
// third_party/pytorch/aten/src/ATen/native/cuda/RNN.cu.
// Layout contract: gate tensors are contiguous row-major (N, G) with
// G = 4*H (LSTM) or 3*H (GRU); states are contiguous (N, H); biases are
// contiguous (G,) or undefined. Half/BFloat16 compute in float32
// (at::acc_type parity); Float64 computes in double.

#include "Tensor.h"
#include <tuple>

namespace tensorplay {
namespace cuda {
namespace rnn {

std::tuple<Tensor, Tensor, Tensor> fused_lstm_cell(
    const Tensor& input_gates, const Tensor& hidden_gates,
    const Tensor& cx, const Tensor& input_bias, const Tensor& hidden_bias);

std::tuple<Tensor, Tensor> fused_gru_cell(
    const Tensor& input_gates, const Tensor& hidden_gates,
    const Tensor& hx, const Tensor& input_bias, const Tensor& hidden_bias);

// grad_hy / grad_cy may be undefined tensors (absent gradients).
std::tuple<Tensor, Tensor, Tensor> fused_lstm_cell_backward_impl(
    const Tensor& grad_hy, const Tensor& grad_cy,
    const Tensor& cx, const Tensor& cy, const Tensor& workspace);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> fused_gru_cell_backward(
    const Tensor& grad_hy, const Tensor& workspace);

} // namespace rnn
} // namespace cuda
} // namespace tensorplay
