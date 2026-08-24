#include "QuantKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cuda_runtime.h>
#include <vector>
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

__global__ void quantized_linear_kernel(int64_t total, int64_t k_size,
                                        int64_t out_features,
                                        const int8_t* x, const int8_t* w,
                                        double input_scale,
                                        int64_t input_zero_point,
                                        const float* w_scales,
                                        const int64_t* w_zps,
                                        const float* bias, float* out) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int64_t m = e / out_features;
    const int64_t n = e - m * out_features;
    const int8_t* x_row = x + m * k_size;
    const int8_t* w_row = w + n * k_size;
    const int64_t w_zp = w_zps[n];
    int64_t acc = 0;
    for (int64_t k = 0; k < k_size; ++k) {
        acc += static_cast<int64_t>(x_row[k] - input_zero_point) *
               static_cast<int64_t>(w_row[k] - w_zp);
    }
    out[e] = static_cast<float>(input_scale) * w_scales[n] *
                 static_cast<float>(acc) +
             bias[n];
}

Tensor quantized_linear_cuda(const Tensor& input, const Tensor& weight,
                             double input_scale, int64_t input_zero_point,
                             const Tensor& weight_scales,
                             const Tensor& weight_zero_points,
                             std::optional<Tensor> bias) {
    // Fused Int8 GEMM with per-channel weight requantization; one thread per
    // (m, n) output element streams both operand rows over K.
    if (input.dtype() != DType::Int8 || weight.dtype() != DType::Int8) {
        TP_THROW(TypeError,
                 "quantized_linear(): activations and weights must be Int8");
    }
    if (input.dim() != 2 || weight.dim() != 2) {
        TP_THROW(ValueError,
                 "quantized_linear(): expected 2-D [M,K] activations and "
                 "[N,K] weights");
    }
    if (!(input_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_linear(): scale must be positive");
    }
    if (input.size(1) != weight.size(1)) {
        TP_THROW(ValueError,
                 "quantized_linear(): incompatible K dimensions (" +
                     std::to_string(input.size(1)) + " vs " +
                     std::to_string(weight.size(1)) + ")");
    }
    const int64_t out_features = weight.size(0);
    if (weight_scales.dim() != 1 || weight_scales.size(0) != out_features ||
        weight_zero_points.shape() != weight_scales.shape()) {
        TP_THROW(ValueError,
                 "quantized_linear(): weight scales/zero_points must be 1-D "
                 "of length out_features");
    }

    Tensor x = input.contiguous();
    Tensor w = weight.contiguous();
    Tensor sc = weight_scales.to(DType::Float32).contiguous();
    Tensor zp = weight_zero_points.to(DType::Int64).contiguous();
    Tensor zps_host = zp.to(Device(DeviceType::CPU));
    const int64_t* host_zps = zps_host.data_ptr<int64_t>();
    for (int64_t n = 0; n < out_features; ++n) {
        if (host_zps[n] < -128 || host_zps[n] > 127) {
            TP_THROW(ValueError,
                     "quantized_linear(): zero_point out of the Int8 range");
        }
    }

    Tensor bias_f;
    if (bias.has_value()) {
        if (!isFloatingType(bias->dtype()) || bias->dim() != 1 ||
            bias->size(0) != out_features) {
            TP_THROW(ValueError,
                     "quantized_linear(): bias must be a 1-D floating tensor "
                     "of length out_features");
        }
        bias_f = bias->to(DType::Float32).contiguous();
    } else {
        bias_f = Tensor::zeros({out_features}, DType::Float32, x.device());
    }

    const int64_t m_size = x.size(0);
    const int64_t k_size = x.size(1);
    Tensor out = Tensor::empty({m_size, out_features}, DType::Float32,
                               x.device());
    const int64_t total = m_size * out_features;
    if (total == 0) return out;

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    quantized_linear_kernel<<<blocks, threads, 0, stream>>>(
        total, k_size, out_features, x.data_ptr<int8_t>(),
        w.data_ptr<int8_t>(), input_scale, input_zero_point,
        sc.data_ptr<float>(), zp.data_ptr<int64_t>(),
        bias_f.data_ptr<float>(), out.data_ptr<float>());
    checkCuda(cudaGetLastError(), "CUDA quantized_linear kernel");
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantKernels) {
    m.impl("quantize_per_tensor", quantize_per_tensor_cuda);
    m.impl("dequantize_per_tensor", dequantize_per_tensor_cuda);
    m.impl("quantize_per_channel", quantize_per_channel_cuda);
    m.impl("dequantize_per_channel", dequantize_per_channel_cuda);
    m.impl("quantized_linear", quantized_linear_cuda);
}

} // namespace cuda
} // namespace tensorplay
