#pragma once

#include "Tensor.h"

#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cpu {

// Weight/bias pre-packing for the quantized 2D convolution, matching the
// access patterns of the quantized convolution compute shaders.
//
// The weight arrives as an Int8 tensor [O, C, KH, KW] with per-output-channel
// scales/zero_points; it is dequantized into the float domain on the host and
// rearranged so that the shader reads run linearly:
//  - depthwise: {4, N4*C, KH*KW} (each NxN filter flattened into one row,
//    groups of four filters stacked vertically);
//  - regular / transposed: {4, N4*KH, C_aligned*KW} (input-channel groups of
//    four folded into the width axis, output-channel groups of four stacked
//    vertically); the transposed form first flips both spatial axes and
//    swaps the channel roles.
// The bias is padded to a multiple of four and reshaped to {4, 1, L4}.
//
// The unpack routines invert the rearrangement so that the CPU/CUDA run
// paths can feed the packed payload back through the float convolution.
std::tuple<Tensor, Tensor> quantized_conv2d_prepack_cpu(
    const Tensor& weight,
    const Tensor& weight_scales,
    const Tensor& weight_zero_points,
    std::optional<Tensor> bias,
    bool transposed);

// Rebuilds the float-domain weight [O, C, KH, KW] and bias [O] from the
// packed payload produced by quantized_conv2d_prepack_cpu.
std::tuple<Tensor, Tensor> quantized_conv2d_unpack_cpu(
    const Tensor& weight_packed,
    const Tensor& bias_packed,
    const std::vector<int64_t>& weight_sizes,
    bool transposed,
    bool depthwise);

} // namespace cpu
} // namespace tensorplay
