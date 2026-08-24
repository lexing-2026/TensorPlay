#pragma once

#include "Tensor.h"

#include <optional>

namespace tensorplay {
namespace cpu {

// Affine quantization on Int8: q = clamp(round(x / scale) + zp, qmin, qmax).
// Floating inputs are promoted to Float32/Float64 paths internally; outputs
// are Int8 with the input's shape.  dequantize_* take the Int8 tensor back
// to Float32: x = (q - zp) * scale.
Tensor quantize_per_tensor_cpu(const Tensor& self, double scale,
                                int64_t zero_point, int64_t quant_min,
                                int64_t quant_max);
Tensor dequantize_per_tensor_cpu(const Tensor& self, double scale,
                                  int64_t zero_point);
Tensor quantize_per_channel_cpu(const Tensor& self, const Tensor& scales,
                                 const Tensor& zero_points, int64_t axis);
Tensor dequantize_per_channel_cpu(const Tensor& self, const Tensor& scales,
                                   const Tensor& zero_points, int64_t axis);
// Fused Int8 GEMM: out[m,n] = x_scale * w_scale[n] * Σ_k (x_q[m,k]-x_zp) *
// (w_q[n,k]-w_zp[n]) + bias[n] -> Float32 [M,N].
Tensor quantized_linear_cpu(const Tensor& input, const Tensor& weight,
                            double input_scale, int64_t input_zero_point,
                            const Tensor& weight_scales,
                            const Tensor& weight_zero_points,
                            std::optional<Tensor> bias);

} // namespace cpu

#ifdef USE_CUDA
namespace cuda {

Tensor quantize_per_tensor_cuda(const Tensor& self, double scale,
                                 int64_t zero_point, int64_t quant_min,
                                 int64_t quant_max);
Tensor dequantize_per_tensor_cuda(const Tensor& self, double scale,
                                  int64_t zero_point);
Tensor quantize_per_channel_cuda(const Tensor& self, const Tensor& scales,
                                  const Tensor& zero_points, int64_t axis);
Tensor dequantize_per_channel_cuda(const Tensor& self, const Tensor& scales,
                                    const Tensor& zero_points, int64_t axis);
Tensor quantized_linear_cuda(const Tensor& input, const Tensor& weight,
                             double input_scale, int64_t input_zero_point,
                             const Tensor& weight_scales,
                             const Tensor& weight_zero_points,
                             std::optional<Tensor> bias);

} // namespace cuda
#endif
} // namespace tensorplay
