#include "QuantKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"
#include "SizesAndStrides.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

// Defined in ConvKernels.cu.
Tensor conv2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride,
                   const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups);

// Defined in PoolingKernels.cu; the quantized window maximum shares the
// float kernel's window logic order-preservingly on Int8 storage.
Tensor max_pool2d_cuda(const Tensor& input,
                       const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation, bool ceil_mode);

Tensor dequantize_per_tensor_qint8_cuda(const Tensor& self, double scale,
                                         int64_t zero_point);

__global__ void quantized_requantize_kernel(
    int64_t numel,
    const float* __restrict__ in,
    int8_t* __restrict__ out,
    float inv_out_scale,
    float out_zp) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float q = rintf(in[i] * inv_out_scale) + out_zp;
    out[i] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, q)));
}

Tensor quantized_max_pool2d_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    // The window maximum is order-preserving in the quantized domain, so the
    // pooling runs on an Int8 view of the code storage and the output is
    // re-wrapped with the input quantizer untouched.
    if (self.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "quantized_max_pool2d(): expected a QInt8 tensor");
    }
    Tensor codes = quantized::strip_quantizer(self);
    Tensor out_codes =
        max_pool2d_cuda(codes, kernel_size, stride, padding, dilation,
                        ceil_mode);
    return quantized::make_qtensor(out_codes, self.impl()->quantizer(),
                                   DType::QInt8);
}

Tensor quantized_conv2d_cuda(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias,
    double input_scale, int64_t input_zero_point, double weight_scale,
    int64_t weight_zero_point, double out_scale, int64_t out_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dtype() != DType::QInt8 || weight.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "quantized_conv2d(): activations and weights must be QInt8");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_conv2d(): out_scale must be positive");
    }
    // Dequantize both operands, run the float convolution, then requantize
    // into the output qparams.
    Tensor x = dequantize_per_tensor_qint8_cuda(
        input, input_scale, input_zero_point);
    Tensor w = dequantize_per_tensor_qint8_cuda(
        weight, weight_scale, weight_zero_point);
    Tensor acc = conv2d_cuda(
        x, w,
        bias.has_value() ? bias->to(DType::Float32).contiguous() : Tensor(),
        stride, padding, dilation, groups);

    Tensor out = Tensor::empty(
        static_cast<std::vector<int64_t>>(acc.shape()), DType::QInt8,
        input.device());
    const int64_t numel = acc.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantized_requantize_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, acc.data_ptr<float>(), out.data_ptr<int8_t>(),
        static_cast<float>(1.0 / out_scale),
        static_cast<float>(out_zero_point));
    checkCuda(cudaGetLastError(), "CUDA quantized_conv2d requantize kernel");
    out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
        out_scale, out_zero_point, DType::QInt8));
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantKernels) {
    m.impl("quantized_max_pool2d", quantized_max_pool2d_cuda);
    m.impl("quantized_conv2d", quantized_conv2d_cuda);
}

} // namespace cuda
} // namespace tensorplay
