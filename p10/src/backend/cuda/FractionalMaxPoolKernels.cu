// fractional_max_pool2d / fractional_max_pool3d CUDA kernels.
//
// Port of aten/src/ATen/native/cuda/FractionalMaxPool2d.cu /
// FractionalMaxPool3d.cu: one thread per output point, window starts from
// get_interval(sample, ...) driven by the caller-provided random_samples
// tensor (no internal RNG). Backward scatters grad via gpuAtomicAdd.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"
#include "Atomic.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kThreads = 256;

// ATen cuda/FractionalMaxPool2d.cu get_interval.
template <typename compute_t>
__device__ inline int64_t get_interval(compute_t sample, int64_t index,
                                       int64_t inputSize, int64_t outputSize,
                                       int64_t poolSize) {
    if (index == outputSize - 1) {
        return inputSize - poolSize;
    }
    compute_t alpha = static_cast<compute_t>(inputSize - poolSize) /
                      static_cast<compute_t>(outputSize - 1);
    return static_cast<int64_t>(
        static_cast<int>((index + sample) * alpha) - static_cast<int>(sample * alpha));
}

template <typename storage_t, typename compute_t>
__global__ void fractional_max_pool2d_fwd_kernel(
        const storage_t* __restrict__ input,
        storage_t* __restrict__ output,
        int64_t* __restrict__ indices,
        const storage_t* __restrict__ samples,
        const int64_t numPlanes, const int64_t inputH, const int64_t inputW,
        const int64_t outputH, const int64_t outputW,
        const int64_t poolSizeH, const int64_t poolSizeW) {
    const int64_t ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t plane = blockIdx.y;
    const int64_t batch = blockIdx.z;
    const int64_t out_vol = outputH * outputW;
    if (ourOutputPoint >= out_vol) return;

    const int64_t outputWIdx = ourOutputPoint % outputW;
    const int64_t outputHIdx = ourOutputPoint / outputW;

    const storage_t* samplesForPlane = samples + (batch * numPlanes + plane) * 2;
    const int64_t poolW = get_interval<compute_t>(
        static_cast<compute_t>(samplesForPlane[0]), outputWIdx, inputW, outputW, poolSizeW);
    const int64_t poolH = get_interval<compute_t>(
        static_cast<compute_t>(samplesForPlane[1]), outputHIdx, inputH, outputH, poolSizeH);

    const storage_t* inputForPlane =
        input + (batch * numPlanes + plane) * inputH * inputW;
    compute_t maxVal = std::numeric_limits<compute_t>::lowest();
    int64_t maxIndex = poolH * inputW + poolW;
    for (int64_t h = poolH; h < poolH + poolSizeH; ++h) {
        for (int64_t w = poolW; w < poolW + poolSizeW; ++w) {
            const compute_t val = static_cast<compute_t>(inputForPlane[h * inputW + w]);
            // ATen: favor the first max; NaNs propagate forward.
            if (val > maxVal || isnan(val)) {
                maxVal = val;
                maxIndex = h * inputW + w;
            }
        }
    }
    const int64_t off = (batch * numPlanes + plane) * out_vol + ourOutputPoint;
    output[off] = static_cast<storage_t>(maxVal);
    indices[off] = maxIndex;
}

template <typename storage_t>
__global__ void fractional_max_pool2d_bwd_kernel(
        storage_t* __restrict__ grad_input,
        const storage_t* __restrict__ grad_output,
        const int64_t* __restrict__ indices,
        const int64_t numPlanes, const int64_t inputH, const int64_t inputW,
        const int64_t outputH, const int64_t outputW) {
    const int64_t ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t plane = blockIdx.y;
    const int64_t batch = blockIdx.z;
    const int64_t out_vol = outputH * outputW;
    if (ourOutputPoint >= out_vol) return;

    const int64_t off = (batch * numPlanes + plane) * out_vol + ourOutputPoint;
    const int64_t index = indices[off];
    if (index < 0 || index >= inputH * inputW) return;
    gpuAtomicAdd(grad_input + (batch * numPlanes + plane) * inputH * inputW + index,
                 grad_output[off]);
}

template <typename storage_t, typename compute_t>
__global__ void fractional_max_pool3d_fwd_kernel(
        const storage_t* __restrict__ input,
        storage_t* __restrict__ output,
        int64_t* __restrict__ indices,
        const storage_t* __restrict__ samples,
        const int64_t numPlanes, const int64_t inputT, const int64_t inputH,
        const int64_t inputW, const int64_t outputT, const int64_t outputH,
        const int64_t outputW, const int64_t poolSizeT, const int64_t poolSizeH,
        const int64_t poolSizeW) {
    const int64_t ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t plane = blockIdx.y;
    const int64_t batch = blockIdx.z;
    const int64_t out_vol = outputT * outputH * outputW;
    if (ourOutputPoint >= out_vol) return;

    const int64_t outputWIdx = ourOutputPoint % outputW;
    const int64_t outputHIdx = (ourOutputPoint / outputW) % outputH;
    const int64_t outputTIdx = ourOutputPoint / (outputH * outputW);

    const storage_t* samplesForPlane = samples + (batch * numPlanes + plane) * 3;
    // ATen FractionalMaxPool3d: sample order is (T, H, W).
    const int64_t poolT = get_interval<compute_t>(
        static_cast<compute_t>(samplesForPlane[0]), outputTIdx, inputT, outputT, poolSizeT);
    const int64_t poolH = get_interval<compute_t>(
        static_cast<compute_t>(samplesForPlane[1]), outputHIdx, inputH, outputH, poolSizeH);
    const int64_t poolW = get_interval<compute_t>(
        static_cast<compute_t>(samplesForPlane[2]), outputWIdx, inputW, outputW, poolSizeW);

    const storage_t* inputForPlane =
        input + (batch * numPlanes + plane) * inputT * inputH * inputW;
    compute_t maxVal = std::numeric_limits<compute_t>::lowest();
    int64_t maxIndex = (poolT * inputH + poolH) * inputW + poolW;
    for (int64_t t = poolT; t < poolT + poolSizeT; ++t) {
        for (int64_t h = poolH; h < poolH + poolSizeH; ++h) {
            for (int64_t w = poolW; w < poolW + poolSizeW; ++w) {
                const int64_t idx = (t * inputH + h) * inputW + w;
                const compute_t val = static_cast<compute_t>(inputForPlane[idx]);
                if (val > maxVal || isnan(val)) {
                    maxVal = val;
                    maxIndex = idx;
                }
            }
        }
    }
    const int64_t off = (batch * numPlanes + plane) * out_vol + ourOutputPoint;
    output[off] = static_cast<storage_t>(maxVal);
    indices[off] = maxIndex;
}

template <typename storage_t>
__global__ void fractional_max_pool3d_bwd_kernel(
        storage_t* __restrict__ grad_input,
        const storage_t* __restrict__ grad_output,
        const int64_t* __restrict__ indices,
        const int64_t numPlanes, const int64_t inputT, const int64_t inputH,
        const int64_t inputW, const int64_t outputT, const int64_t outputH,
        const int64_t outputW) {
    const int64_t ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t plane = blockIdx.y;
    const int64_t batch = blockIdx.z;
    const int64_t out_vol = outputT * outputH * outputW;
    if (ourOutputPoint >= out_vol) return;

    const int64_t off = (batch * numPlanes + plane) * out_vol + ourOutputPoint;
    const int64_t index = indices[off];
    const int64_t in_vol = inputT * inputH * inputW;
    if (index < 0 || index >= in_vol) return;
    gpuAtomicAdd(grad_input + (batch * numPlanes + plane) * in_vol + index,
                 grad_output[off]);
}

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

void fractional_pool_check(const Tensor& input, const Tensor& random_samples,
                           int64_t ndim_spatial, const char* name) {
    if (input.dtype() != random_samples.dtype())
        TP_THROW(RuntimeError, std::string(name) + ": expect random_samples to have the same dtype as input");
    if (random_samples.dim() != 3)
        TP_THROW(RuntimeError, std::string(name) + ": expect random_samples to have 3 dimensions");
    const int64_t input_batch = input.dim() == ndim_spatial + 2 ? input.size(0) : 1;
    const int64_t input_channel = input.dim() == ndim_spatial + 2 ? input.size(1) : input.size(0);
    if (random_samples.size(0) < input_batch)
        TP_THROW(RuntimeError, std::string(name) + ": random_samples.size(0) must be >= input batch size");
    if (random_samples.size(1) != input_channel)
        TP_THROW(RuntimeError, std::string(name) + ": random_samples.size(1) must equal input channels");
    if (random_samples.size(2) != ndim_spatial)
        TP_THROW(RuntimeError, std::string(name) + ": random_samples.size(2) must equal the number of spatial dims");
}

}  // namespace

template <typename storage_t, typename compute_t>
static std::tuple<Tensor, Tensor> fractional_max_pool2d_cuda_impl(
        const Tensor& input, const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& random_samples) {
    const bool batched = input.dim() == 4;
    const int64_t numBatch = batched ? input.size(0) : 1;
    const int64_t numPlanes = input.size(batched ? 1 : 0);
    const int64_t inputH = input.size(batched ? 2 : 1);
    const int64_t inputW = input.size(batched ? 3 : 2);
    const int64_t outputH = output_size[0];
    const int64_t outputW = output_size[1];

    std::vector<int64_t> out_shape = batched
        ? std::vector<int64_t>{numBatch, numPlanes, outputH, outputW}
        : std::vector<int64_t>{numPlanes, outputH, outputW};
    Tensor output = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    const int64_t out_vol = outputH * outputW;
    if (out_vol == 0 || numPlanes == 0) return {output, indices};

    const Tensor ic = input.contiguous();
    const Tensor sc = random_samples.contiguous();
    dim3 block(kThreads);
    dim3 grid(ceil_div(out_vol, kThreads), numPlanes, numBatch);
    fractional_max_pool2d_fwd_kernel<storage_t, compute_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
        ic.data_ptr<storage_t>(), output.data_ptr<storage_t>(),
        indices.data_ptr<int64_t>(), sc.data_ptr<storage_t>(),
        numPlanes, inputH, inputW, outputH, outputW,
        kernel_size[0], kernel_size[1]);
    return {output, indices};
}

template <typename storage_t>
static Tensor fractional_max_pool2d_backward_cuda_impl(
        const Tensor& grad_output, const Tensor& input,
        const std::vector<int64_t>& output_size, const Tensor& indices) {
    const bool batched = input.dim() == 4;
    const int64_t numBatch = batched ? input.size(0) : 1;
    const int64_t numPlanes = input.size(batched ? 1 : 0);
    const int64_t inputH = input.size(batched ? 2 : 1);
    const int64_t inputW = input.size(batched ? 3 : 2);
    const int64_t outputH = output_size[0];
    const int64_t outputW = output_size[1];

    std::vector<int64_t> in_shape = batched
        ? std::vector<int64_t>{numBatch, numPlanes, inputH, inputW}
        : std::vector<int64_t>{numPlanes, inputH, inputW};
    Tensor grad_input = Tensor::zeros(in_shape, input.dtype(), input.device());
    const int64_t out_vol = outputH * outputW;
    if (out_vol == 0 || numPlanes == 0) return grad_input;

    const Tensor goc = grad_output.contiguous();
    const Tensor ic = indices.contiguous();
    dim3 block(kThreads);
    dim3 grid(ceil_div(out_vol, kThreads), numPlanes, numBatch);
    fractional_max_pool2d_bwd_kernel<storage_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
        grad_input.data_ptr<storage_t>(), goc.data_ptr<storage_t>(),
        ic.data_ptr<int64_t>(), numPlanes, inputH, inputW, outputH, outputW);
    return grad_input;
}

template <typename storage_t, typename compute_t>
static std::tuple<Tensor, Tensor> fractional_max_pool3d_cuda_impl(
        const Tensor& input, const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& random_samples) {
    const bool batched = input.dim() == 5;
    const int64_t numBatch = batched ? input.size(0) : 1;
    const int64_t numPlanes = input.size(batched ? 1 : 0);
    const int64_t inputT = input.size(batched ? 2 : 1);
    const int64_t inputH = input.size(batched ? 3 : 2);
    const int64_t inputW = input.size(batched ? 4 : 3);
    const int64_t outputT = output_size[0];
    const int64_t outputH = output_size[1];
    const int64_t outputW = output_size[2];

    std::vector<int64_t> out_shape = batched
        ? std::vector<int64_t>{numBatch, numPlanes, outputT, outputH, outputW}
        : std::vector<int64_t>{numPlanes, outputT, outputH, outputW};
    Tensor output = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    const int64_t out_vol = outputT * outputH * outputW;
    if (out_vol == 0 || numPlanes == 0) return {output, indices};

    const Tensor ic = input.contiguous();
    const Tensor sc = random_samples.contiguous();
    dim3 block(kThreads);
    dim3 grid(ceil_div(out_vol, kThreads), numPlanes, numBatch);
    fractional_max_pool3d_fwd_kernel<storage_t, compute_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
        ic.data_ptr<storage_t>(), output.data_ptr<storage_t>(),
        indices.data_ptr<int64_t>(), sc.data_ptr<storage_t>(),
        numPlanes, inputT, inputH, inputW, outputT, outputH, outputW,
        kernel_size[0], kernel_size[1], kernel_size[2]);
    return {output, indices};
}

template <typename storage_t>
static Tensor fractional_max_pool3d_backward_cuda_impl(
        const Tensor& grad_output, const Tensor& input,
        const std::vector<int64_t>& output_size, const Tensor& indices) {
    const bool batched = input.dim() == 5;
    const int64_t numBatch = batched ? input.size(0) : 1;
    const int64_t numPlanes = input.size(batched ? 1 : 0);
    const int64_t inputT = input.size(batched ? 2 : 1);
    const int64_t inputH = input.size(batched ? 3 : 2);
    const int64_t inputW = input.size(batched ? 4 : 3);
    const int64_t outputT = output_size[0];
    const int64_t outputH = output_size[1];
    const int64_t outputW = output_size[2];

    std::vector<int64_t> in_shape = batched
        ? std::vector<int64_t>{numBatch, numPlanes, inputT, inputH, inputW}
        : std::vector<int64_t>{numPlanes, inputT, inputH, inputW};
    Tensor grad_input = Tensor::zeros(in_shape, input.dtype(), input.device());
    const int64_t out_vol = outputT * outputH * outputW;
    if (out_vol == 0 || numPlanes == 0) return grad_input;

    const Tensor goc = grad_output.contiguous();
    const Tensor ic = indices.contiguous();
    dim3 block(kThreads);
    dim3 grid(ceil_div(out_vol, kThreads), numPlanes, numBatch);
    fractional_max_pool3d_bwd_kernel<storage_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
        grad_input.data_ptr<storage_t>(), goc.data_ptr<storage_t>(),
        ic.data_ptr<int64_t>(), numPlanes, inputT, inputH, inputW,
        outputT, outputH, outputW);
    return grad_input;
}

std::tuple<Tensor, Tensor> fractional_max_pool2d_cuda(
        const Tensor& self, const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& random_samples) {
    if (kernel_size.size() != 2 || output_size.size() != 2)
        TP_THROW(RuntimeError, "fractional_max_pool2d: kernel_size and output_size must have 2 elements");
    if (self.dim() != 3 && self.dim() != 4)
        TP_THROW(RuntimeError, "fractional_max_pool2d: expected 3D or 4D input");
    if (kernel_size[0] <= 0 || kernel_size[1] <= 0)
        TP_THROW(RuntimeError, "fractional_max_pool2d: kernel_size must be positive");
    const int64_t inputH = self.size(self.dim() - 2);
    const int64_t inputW = self.size(self.dim() - 1);
    if (output_size[0] + kernel_size[0] - 1 > inputH || output_size[1] + kernel_size[1] - 1 > inputW)
        TP_THROW(RuntimeError, "fractional_max_pool2d: kernel too large relative to input");
    fractional_pool_check(self, random_samples, 2, "fractional_max_pool2d");
    switch (self.dtype()) {
        case DType::Float32: return fractional_max_pool2d_cuda_impl<float, float>(self, kernel_size, output_size, random_samples);
        case DType::Float64: return fractional_max_pool2d_cuda_impl<double, double>(self, kernel_size, output_size, random_samples);
        case DType::Float16: return fractional_max_pool2d_cuda_impl<Half, float>(self, kernel_size, output_size, random_samples);
        case DType::BFloat16: return fractional_max_pool2d_cuda_impl<BFloat16, float>(self, kernel_size, output_size, random_samples);
        default: TP_THROW(TypeError, "fractional_max_pool2d: unsupported dtype");
    }
}

Tensor fractional_max_pool2d_backward_cuda(
        const Tensor& grad_output, const Tensor& self,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& indices) {
    if (output_size.size() != 2)
        TP_THROW(RuntimeError, "fractional_max_pool2d_backward: output_size must have 2 elements");
    if (self.dim() != 3 && self.dim() != 4)
        TP_THROW(RuntimeError, "fractional_max_pool2d_backward: expected 3D or 4D input");
    switch (self.dtype()) {
        case DType::Float32: return fractional_max_pool2d_backward_cuda_impl<float>(grad_output, self, output_size, indices);
        case DType::Float64: return fractional_max_pool2d_backward_cuda_impl<double>(grad_output, self, output_size, indices);
        case DType::Float16: return fractional_max_pool2d_backward_cuda_impl<Half>(grad_output, self, output_size, indices);
        case DType::BFloat16: return fractional_max_pool2d_backward_cuda_impl<BFloat16>(grad_output, self, output_size, indices);
        default: TP_THROW(TypeError, "fractional_max_pool2d_backward: unsupported dtype");
    }
}

std::tuple<Tensor, Tensor> fractional_max_pool3d_cuda(
        const Tensor& self, const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& random_samples) {
    if (kernel_size.size() != 3 || output_size.size() != 3)
        TP_THROW(RuntimeError, "fractional_max_pool3d: kernel_size and output_size must have 3 elements");
    if (self.dim() != 4 && self.dim() != 5)
        TP_THROW(RuntimeError, "fractional_max_pool3d: expected 4D or 5D input");
    if (kernel_size[0] <= 0 || kernel_size[1] <= 0 || kernel_size[2] <= 0)
        TP_THROW(RuntimeError, "fractional_max_pool3d: kernel_size must be positive");
    const int64_t inputT = self.size(self.dim() - 3);
    const int64_t inputH = self.size(self.dim() - 2);
    const int64_t inputW = self.size(self.dim() - 1);
    // ATen FractionalMaxPool3d.cpp:74-82 requires output + pool - 1 < input
    // (strict, unlike the 2D variant which allows equality).
    if (output_size[0] + kernel_size[0] - 1 >= inputT ||
        output_size[1] + kernel_size[1] - 1 >= inputH ||
        output_size[2] + kernel_size[2] - 1 >= inputW)
        TP_THROW(RuntimeError, "fractional_max_pool3d: kernel too large relative to input");
    fractional_pool_check(self, random_samples, 3, "fractional_max_pool3d");
    switch (self.dtype()) {
        case DType::Float32: return fractional_max_pool3d_cuda_impl<float, float>(self, kernel_size, output_size, random_samples);
        case DType::Float64: return fractional_max_pool3d_cuda_impl<double, double>(self, kernel_size, output_size, random_samples);
        case DType::Float16: return fractional_max_pool3d_cuda_impl<Half, float>(self, kernel_size, output_size, random_samples);
        case DType::BFloat16: return fractional_max_pool3d_cuda_impl<BFloat16, float>(self, kernel_size, output_size, random_samples);
        default: TP_THROW(TypeError, "fractional_max_pool3d: unsupported dtype");
    }
}

Tensor fractional_max_pool3d_backward_cuda(
        const Tensor& grad_output, const Tensor& self,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& output_size, const Tensor& indices) {
    if (output_size.size() != 3)
        TP_THROW(RuntimeError, "fractional_max_pool3d_backward: output_size must have 3 elements");
    if (self.dim() != 4 && self.dim() != 5)
        TP_THROW(RuntimeError, "fractional_max_pool3d_backward: expected 4D or 5D input");
    switch (self.dtype()) {
        case DType::Float32: return fractional_max_pool3d_backward_cuda_impl<float>(grad_output, self, output_size, indices);
        case DType::Float64: return fractional_max_pool3d_backward_cuda_impl<double>(grad_output, self, output_size, indices);
        case DType::Float16: return fractional_max_pool3d_backward_cuda_impl<Half>(grad_output, self, output_size, indices);
        case DType::BFloat16: return fractional_max_pool3d_backward_cuda_impl<BFloat16>(grad_output, self, output_size, indices);
        default: TP_THROW(TypeError, "fractional_max_pool3d_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CUDA, FractionalMaxPoolKernels) {
    m.impl("fractional_max_pool2d", fractional_max_pool2d_cuda);
    m.impl("fractional_max_pool2d_backward", fractional_max_pool2d_backward_cuda);
    m.impl("fractional_max_pool3d", fractional_max_pool3d_cuda);
    m.impl("fractional_max_pool3d_backward", fractional_max_pool3d_backward_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
