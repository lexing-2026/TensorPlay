#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

namespace tensorplay {
namespace cuda {

namespace {

// Per-row quantization for 2-bit / 1-bit signed codes.
//
//   scale = max(|row|) / scale_divisor
//   codes = clamp(round(x / scale), quant_min, quant_max)
//
// The divisor is passed separately from `quant_max` so that a 2-bit grid
// {-2, -1, 0, 1} can use `scale = max_abs / 2` while still clamping the
// upper edge to 1, and a 1-bit grid {-1, +1} can use `scale = max_abs`.
template <typename scalar_t>
__global__ void lowbit_quantize_kernel(const scalar_t* __restrict__ x,
                                       int8_t* __restrict__ codes,
                                       float* __restrict__ scales, int K,
                                       float eps, float scale_divisor,
                                       int quant_min, int quant_max,
                                       bool sign_mode) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const scalar_t* row_ptr = x + static_cast<int64_t>(row) * K;

    float absmax = 0.0f;
    for (int i = tid; i < K; i += nthreads) {
        absmax = fmaxf(absmax, fabsf(static_cast<float>(row_ptr[i])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, offset));
    }
    __shared__ float s_max[32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) s_max[warp] = absmax;
    __syncthreads();
    if (warp == 0) {
        absmax = (lane < (nthreads >> 5)) ? s_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, offset));
        }
        if (lane == 0) s_max[0] = absmax;
    }
    __syncthreads();

    const float scale = fmaxf(s_max[0], eps) / scale_divisor;
    if (tid == 0) scales[row] = scale;

    int8_t* codes_row = codes + static_cast<int64_t>(row) * K;
    for (int i = tid; i < K; i += nthreads) {
        int c;
        if (sign_mode) {
            // Strict sign quantization: +1 when x >= 0, otherwise -1.
            c = (static_cast<float>(row_ptr[i]) >= 0.0f) ? quant_max
                                                         : quant_min;
        } else {
            c = static_cast<int>(
                rintf(static_cast<float>(row_ptr[i]) / scale));
            c = c < quant_min ? quant_min : (c > quant_max ? quant_max : c);
        }
        codes_row[i] = static_cast<int8_t>(c);
    }
}

void launch_lowbit_quantize(const Tensor& x, Tensor& codes, Tensor& scales,
                            float eps, float scale_divisor, int quant_min,
                            int quant_max, bool sign_mode, cudaStream_t stream) {
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    constexpr int kThreads = 256;
    const dim3 grid(static_cast<unsigned>(M));
    switch (x.dtype()) {
        case DType::Float16:
            lowbit_quantize_kernel<tensorplay::Half>
                <<<grid, kThreads, 0, stream>>>(
                    static_cast<const tensorplay::Half*>(x.data_ptr()),
                    codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                    static_cast<int>(K), eps, scale_divisor, quant_min,
                    quant_max, sign_mode);
            break;
        case DType::BFloat16:
            lowbit_quantize_kernel<tensorplay::BFloat16>
                <<<grid, kThreads, 0, stream>>>(
                    static_cast<const tensorplay::BFloat16*>(x.data_ptr()),
                    codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                    static_cast<int>(K), eps, scale_divisor, quant_min,
                    quant_max, sign_mode);
            break;
        case DType::Float32:
            lowbit_quantize_kernel<float>
                <<<grid, kThreads, 0, stream>>>(
                    static_cast<const float*>(x.data_ptr()),
                    codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                    static_cast<int>(K), eps, scale_divisor, quant_min,
                    quant_max, sign_mode);
            break;
        default:
            TP_THROW(TypeError, "lowbit quantize: unsupported input dtype");
    }
}

std::tuple<Tensor, Tensor> quantize_lowbit_rowwise(const Tensor& self,
                                                   double eps,
                                                   float scale_divisor,
                                                   int quant_min,
                                                   int quant_max,
                                                   bool sign_mode,
                                                   const char* op_name) {
    if (self.dim() != 2) {
        TP_THROW(ValueError, std::string(op_name) + "(): expected a 2-D tensor");
    }
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const Tensor x = self.contiguous();
    Tensor codes = Tensor::empty({M, K}, DType::Int8, self.device());
    Tensor scales = Tensor::empty({M}, DType::Float32, self.device());
    if (M == 0 || K == 0) {
        return {codes, scales};
    }
    launch_lowbit_quantize(x, codes, scales, static_cast<float>(eps),
                           scale_divisor, quant_min, quant_max, sign_mode,
                           getCurrentCUDAStream().stream());
    checkCuda(cudaGetLastError(), op_name);
    return {codes, scales};
}

}  // namespace

// Row-wise int2 quantization: signed codes in {-2, -1, 0, 1} and
// scale = max(|row|) / 2, so the dequantized row spans [-2*scale, 1*scale].
std::tuple<Tensor, Tensor> _quantize_int2_rowwise_cuda(const Tensor& self,
                                                       double eps) {
    return quantize_lowbit_rowwise(self, eps, 2.0f, -2, 1, false,
                                   "_quantize_int2_rowwise");
}

// Row-wise 1-bit quantization: strict sign codes in {-1, +1} and
// scale = max(|row|), so the dequantized row is sign(row) * max(|row|).
std::tuple<Tensor, Tensor> _quantize_binary_rowwise_cuda(const Tensor& self,
                                                         double eps) {
    return quantize_lowbit_rowwise(self, eps, 1.0f, -1, 1, true,
                                   "_quantize_binary_rowwise");
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LowBitGemmKernels) {
    m.impl("_quantize_int2_rowwise", _quantize_int2_rowwise_cuda);
    m.impl("_quantize_binary_rowwise", _quantize_binary_rowwise_cuda);
}

}  // namespace cuda
}  // namespace tensorplay