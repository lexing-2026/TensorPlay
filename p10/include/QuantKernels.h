#pragma once

#include "Tensor.h"

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

} // namespace cuda
#endif
} // namespace tensorplay
