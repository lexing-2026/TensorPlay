#include "QuantNative.h"
// Native quantized pooling kernels for the CUDA backend.
//
// Window maximum is order-preserving on the affine Int8 grid, so pooling runs
// directly on the code storage and the output inherits the input quantizer.
// The 1d window is lifted into a 2d window and dispatched to the existing 2d
// pooling kernel; the 3d window uses a dedicated kernel.

#include "QuantCudaKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

__global__ void quantized_max_pool2d_kernel(
    const int8_t* __restrict__ in, int8_t* __restrict__ out,
    int64_t total, int64_t C, int64_t H, int64_t W, int64_t OH, int64_t OW,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW, int64_t pH, int64_t pW,
    int64_t dH, int64_t dW) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t ow = idx % OW;
    const int64_t oh = (idx / OW) % OH;
    const int64_t c = (idx / (OW * OH)) % C;
    const int64_t n = idx / (OW * OH * C);
    int8_t best = -128;
    for (int64_t kh = 0; kh < kH; ++kh) {
        const int64_t ih = oh * sH - pH + kh * dH;
        if (ih < 0 || ih >= H) continue;
        for (int64_t kw = 0; kw < kW; ++kw) {
            const int64_t iw = ow * sW - pW + kw * dW;
            if (iw < 0 || iw >= W) continue;
            const int8_t v = in[((n * C + c) * H + ih) * W + iw];
            if (v > best) best = v;
        }
    }
    out[idx] = best;
}

__global__ void quantized_max_pool3d_kernel(
    const int8_t* __restrict__ in, int8_t* __restrict__ out,
    int64_t total, int64_t C, int64_t D, int64_t H, int64_t W, int64_t OD,
    int64_t OH, int64_t OW, int64_t kD, int64_t kH, int64_t kW, int64_t sD,
    int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW, int64_t dD,
    int64_t dH, int64_t dW) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t ow = idx % OW;
    const int64_t oh = (idx / OW) % OH;
    const int64_t od = (idx / (OW * OH)) % OD;
    const int64_t c = (idx / (OW * OH * OD)) % C;
    const int64_t n = idx / (OW * OH * OD * C);
    int8_t best = -128;
    for (int64_t kd = 0; kd < kD; ++kd) {
        const int64_t id = od * sD - pD + kd * dD;
        if (id < 0 || id >= D) continue;
        for (int64_t kh = 0; kh < kH; ++kh) {
            const int64_t ih = oh * sH - pH + kh * dH;
            if (ih < 0 || ih >= H) continue;
            for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t iw = ow * sW - pW + kw * dW;
                if (iw < 0 || iw >= W) continue;
                const int8_t v =
                    in[((n * C + c) * D + id) * H * W + ih * W + iw];
                if (v > best) best = v;
            }
        }
    }
    out[idx] = best;
}

int64_t pool_out_dim(int64_t in, int64_t kernel, int64_t stride, int64_t pad,
                     int64_t dilation, bool ceil_mode) {
    const int64_t effective = dilation * (kernel - 1) + 1;
    int64_t out = (in + 2 * pad - effective) / stride + 1;
    if (ceil_mode && (in + 2 * pad - effective) % stride != 0) {
        out += 1;
    }
    return out;
}

} // namespace

Tensor quantized_max_pool2d_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    const auto kernel=kernel_size.size()==1 ? std::vector<int64_t>(2,kernel_size[0]) : kernel_size;
    return quant::pool_cuda(self, kernel, stride, padding, dilation,
                            ceil_mode, false, std::nullopt, 2);
}

Tensor quantized_max_pool1d_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    return quant::pool_cuda(self, kernel_size, stride, padding, dilation,
                            ceil_mode, false, std::nullopt, 2);
}

Tensor quantized_max_pool3d_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    return quant::pool_cuda(self, kernel_size, stride, padding, dilation,
                            ceil_mode, false, std::nullopt, 2);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantPoolKernels) {
    m.impl("quantized_max_pool2d", quantized_max_pool2d_cuda);
}

} // namespace cuda
} // namespace tensorplay
