#include "QuantKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cuda_runtime.h>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

__global__ void quantize_per_tensor_kernel(
    int64_t numel,
    const float* input,
    int8_t* output,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float q = nearbyintf(input[i] / scale) + static_cast<float>(zero_point);
    const float clamped =
        fminf(static_cast<float>(quant_max),
              fmaxf(static_cast<float>(quant_min), q));
    output[i] = static_cast<int8_t>(clamped);
}

__global__ void dequantize_per_tensor_kernel(
    int64_t numel,
    const int8_t* input,
    float* output,
    float scale,
    int64_t zero_point) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    output[i] = (static_cast<float>(input[i]) - static_cast<float>(zero_point)) * scale;
}

__global__ void quantize_per_channel_kernel(
    int64_t numel,
    const float* input,
    int8_t* output,
    int64_t stride_on_axis,
    int64_t channels,
    const float* scales,
    const int64_t* zero_points) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    const float q = nearbyintf(input[i] / scales[c]) +
                    static_cast<float>(zero_points[c]);
    const float clamped = fminf(127.0f, fmaxf(-128.0f, q));
    output[i] = static_cast<int8_t>(clamped);
}

__global__ void dequantize_per_channel_kernel(
    int64_t numel,
    const int8_t* input,
    float* output,
    int64_t stride_on_axis,
    int64_t channels,
    const float* scales,
    const int64_t* zero_points) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    output[i] = (static_cast<float>(input[i]) -
                 static_cast<float>(zero_points[c])) * scales[c];
}

void check_qparams(double scale, int64_t zero_point, int64_t quant_min,
                   int64_t quant_max) {
    if (!(scale > 0.0)) {
        TP_THROW(ValueError, "quantize(): scale must be positive");
    }
    if (quant_min >= quant_max) {
        TP_THROW(ValueError, "quantize(): quant_min must be < quant_max");
    }
    if (zero_point < quant_min || zero_point > quant_max) {
        TP_THROW(ValueError, "quantize(): zero_point out of the quantized range");
    }
}

// Shared host-side preparation: validate dtypes and land the operands on
// Float32/Int8 compute layouts.  Half/BFloat16 promote to Float32 first.
struct QuantInputs {
    Tensor input;      // Float32 compute buffer
    int64_t stride_on_axis;
};

QuantInputs prepare_quantize(const Tensor& self, int64_t axis = 0) {
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float16 &&
        self.dtype() != DType::BFloat16 && self.dtype() != DType::Float64) {
        TP_THROW(TypeError, "quantize(): expected a floating point tensor");
    }
    QuantInputs out{};
    out.input = (self.dtype() == DType::Float32
                     ? self : self.to(DType::Float32)).contiguous();
    int64_t stride = 1;
    for (int64_t d = axis + 1; d < self.dim(); ++d) stride *= self.size(d);
    out.stride_on_axis = stride;
    return out;
}

} // namespace

Tensor quantize_per_tensor_cuda(const Tensor& self, double scale,
                                int64_t zero_point, int64_t quant_min,
                                int64_t quant_max) {
    check_qparams(scale, zero_point, quant_min, quant_max);
    QuantInputs prepared = prepare_quantize(self);
    Tensor out = Tensor::empty(self.shape(), DType::Int8, self.device());
    const int64_t numel = prepared.input.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantize_per_tensor_kernel<<<blocks, threads, 0, stream>>>(
        numel, prepared.input.data_ptr<float>(), out.data_ptr<int8_t>(),
        static_cast<float>(scale), zero_point,
        quant_min, quant_max);
    checkCuda(cudaGetLastError(), "CUDA quantize_per_tensor kernel");
    return out;
}

Tensor dequantize_per_tensor_cuda(const Tensor& self, double scale,
                                  int64_t zero_point) {
    if (self.dtype() != DType::Int8) {
        TP_THROW(TypeError, "dequantize(): expected an Int8 tensor");
    }
    if (!(scale > 0.0)) {
        TP_THROW(ValueError, "dequantize(): scale must be positive");
    }
    Tensor input = self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());
    const int64_t numel = input.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    dequantize_per_tensor_kernel<<<blocks, threads, 0, stream>>>(
        numel, input.data_ptr<int8_t>(), out.data_ptr<float>(),
        static_cast<float>(scale), zero_point);
    checkCuda(cudaGetLastError(), "CUDA dequantize_per_tensor kernel");
    return out;
}

Tensor quantize_per_channel_cuda(const Tensor& self, const Tensor& scales,
                                 const Tensor& zero_points, int64_t axis) {
    if (scales.dim() != 1 || zero_points.shape() != scales.shape()) {
        TP_THROW(ValueError,
                 "quantize(): scales/zero_points must be 1-D with equal sizes");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "quantize(): axis out of range");
    }
    if (scales.size(0) != self.size(axis)) {
        TP_THROW(ValueError,
                 "quantize(): scales size must match the quantized dimension");
    }
    QuantInputs prepared = prepare_quantize(self, axis);
    Tensor scales_f32 = scales.to(DType::Float32).contiguous();
    Tensor zps_i64 = zero_points.to(DType::Int64).contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Int8, self.device());
    const int64_t numel = prepared.input.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantize_per_channel_kernel<<<blocks, threads, 0, stream>>>(
        numel, prepared.input.data_ptr<float>(), out.data_ptr<int8_t>(),
        prepared.stride_on_axis, scales.size(0),
        scales_f32.data_ptr<float>(), zps_i64.data_ptr<int64_t>());
    checkCuda(cudaGetLastError(), "CUDA quantize_per_channel kernel");
    return out;
}

Tensor dequantize_per_channel_cuda(const Tensor& self, const Tensor& scales,
                                   const Tensor& zero_points, int64_t axis) {
    if (self.dtype() != DType::Int8) {
        TP_THROW(TypeError, "dequantize(): expected an Int8 tensor");
    }
    if (scales.dim() != 1 || zero_points.shape() != scales.shape()) {
        TP_THROW(ValueError,
                 "dequantize(): scales/zero_points must be 1-D with equal sizes");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "dequantize(): axis out of range");
    }
    if (scales.size(0) != self.size(axis)) {
        TP_THROW(ValueError,
                 "dequantize(): scales size must match the quantized dimension");
    }
    Tensor input = self.contiguous();
    Tensor scales_f32 = scales.to(DType::Float32).contiguous();
    Tensor zps_i64 = zero_points.to(DType::Int64).contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());

    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < input.dim(); ++d) stride_on_axis *= input.size(d);

    const int64_t numel = input.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    dequantize_per_channel_kernel<<<blocks, threads, 0, stream>>>(
        numel, input.data_ptr<int8_t>(), out.data_ptr<float>(),
        stride_on_axis, scales.size(0),
        scales_f32.data_ptr<float>(), zps_i64.data_ptr<int64_t>());
    checkCuda(cudaGetLastError(), "CUDA dequantize_per_channel kernel");
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantKernels) {
    m.impl("quantize_per_tensor", quantize_per_tensor_cuda);
    m.impl("dequantize_per_tensor", dequantize_per_tensor_cuda);
    m.impl("quantize_per_channel", quantize_per_channel_cuda);
    m.impl("dequantize_per_channel", dequantize_per_channel_cuda);
}

} // namespace cuda
} // namespace tensorplay
