#include "QuantKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"
#include "SizesAndStrides.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <optional>
#include <vector>
#include <cuda_runtime.h>
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
    Tensor out = Tensor::empty(self.shape(), DType::QInt8, self.device());
    const int64_t numel = prepared.input.numel();
    if (numel == 0) {
        out.impl()->set_quantizer(
            std::make_shared<PerTensorAffineQuantizer>(scale, zero_point));
        return out;
    }
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantize_per_tensor_kernel<<<blocks, threads, 0, stream>>>(
        numel, prepared.input.data_ptr<float>(), out.data_ptr<int8_t>(),
        static_cast<float>(scale), zero_point,
        quant_min, quant_max);
    checkCuda(cudaGetLastError(), "CUDA quantize_per_tensor kernel");
    out.impl()->set_quantizer(
        std::make_shared<PerTensorAffineQuantizer>(scale, zero_point));
    return out;
}

Tensor dequantize_per_tensor_cuda(const Tensor& self, double scale,
                                  int64_t zero_point) {
    if (self.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "dequantize(): expected a QInt8 tensor");
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
    Tensor out = Tensor::empty(self.shape(), DType::QInt8, self.device());
    out.impl()->set_quantizer(
        std::make_shared<PerChannelAffineQuantizer>(
            scales.to(DType::Float64).contiguous(),
            zps_i64, axis));
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
    if (self.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "dequantize(): expected a QInt8 tensor");
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
    if (input.dtype() != DType::QInt8 || weight.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "quantized_linear(): activations and weights must be QInt8");
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

// ---------------------------------------------------------------------------
// Quantized elementwise arithmetic over Int8 storage with explicit qparams:
// dequantize both operands as (q - zero_point) * scale, apply the float
// operation, requantize with round-to-nearest-even into [-128, 127] under
// the output qparams.  Division by zero follows IEEE float rules.
// ---------------------------------------------------------------------------

namespace {

__global__ void quantized_binary_kernel(
    int64_t numel,
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int8_t* __restrict__ out,
    const int64_t* __restrict__ a_strides,
    const int64_t* __restrict__ b_strides,
    const int64_t* __restrict__ out_strides,
    int rank,
    float a_scale, float a_zp,
    float b_scale, float b_zp,
    float inv_out_scale, float out_zp,
    int op) {
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= numel) return;

    int64_t rem = flat;
    int64_t ia = 0, ib = 0;
    for (int d = 0; d < rank; ++d) {
        const int64_t coord = rem / out_strides[d];
        rem -= coord * out_strides[d];
        ia += coord * a_strides[d];
        ib += coord * b_strides[d];
    }

    const float xa = (static_cast<float>(a[ia]) - a_zp) * a_scale;
    const float xb = (static_cast<float>(b[ib]) - b_zp) * b_scale;
    float y;
    switch (op) {
        case 0: y = xa + xb; break;
        case 1: y = xa - xb; break;
        case 2: y = xa * xb; break;
        default: y = xa / xb; break;
    }
    const float q = rintf(y * inv_out_scale) + out_zp;
    out[flat] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, q)));
}

void check_quantized_binary(
    const Tensor& a, const Tensor& b, double out_scale) {
    if (a.dtype() != DType::QInt8 || b.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "quantized op(): operands must be QInt8");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized op(): out_scale must be positive");
    }
}

Tensor quantized_binary_cuda_impl(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point,
    int op) {
    check_quantized_binary(a, b, out_scale);
    const std::vector<int64_t> out_sizes =
        broadcast_shapes(static_cast<std::vector<int64_t>>(a.shape()),
                         static_cast<std::vector<int64_t>>(b.shape()));
    Tensor out = Tensor::empty(out_sizes, DType::QInt8, a.device());

    // Broadcast strides pin singleton dimensions with a zero step, so one
    // flat index decomposition addresses both operands without materializing
    // an expanded copy.
    const std::vector<int64_t> sa = broadcast_strides(a, out_sizes);
    const std::vector<int64_t> sb = broadcast_strides(b, out_sizes);
    const std::vector<int64_t> so =
        SizesAndStrides::compute_contiguous_strides(out_sizes);
    const int rank = static_cast<int>(out_sizes.size());

    Tensor strides_gpu = Tensor::empty(
        {static_cast<int64_t>(rank) * 3}, DType::Int64, a.device());
    int64_t* host = static_cast<int64_t*>(
        malloc(sizeof(int64_t) * rank * 3));
    for (int d = 0; d < rank; ++d) {
        host[d] = sa[d];
        host[rank + d] = sb[d];
        host[2 * rank + d] = so[d];
    }
    cudaMemcpy(strides_gpu.data_ptr<int64_t>(), host,
               sizeof(int64_t) * rank * 3, cudaMemcpyHostToDevice);
    free(host);

    const Tensor ac = a.is_contiguous() ? a : a.contiguous();
    const Tensor bc = b.is_contiguous() ? b : b.contiguous();

    const int64_t numel = out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantized_binary_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, ac.data_ptr<int8_t>(), bc.data_ptr<int8_t>(),
        out.data_ptr<int8_t>(),
        strides_gpu.data_ptr<int64_t>(),
        strides_gpu.data_ptr<int64_t>() + rank,
        strides_gpu.data_ptr<int64_t>() + 2 * rank,
        rank,
        static_cast<float>(a_scale), static_cast<float>(a_zero_point),
        static_cast<float>(b_scale), static_cast<float>(b_zero_point),
        static_cast<float>(1.0 / out_scale),
        static_cast<float>(out_zero_point), op);
    checkCuda(cudaGetLastError(), "CUDA quantized binary kernel");
    out.impl()->set_quantizer(
        std::make_shared<PerTensorAffineQuantizer>(out_scale, out_zero_point));
    return out;
}

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

__global__ void quantized_clamp_kernel(
    int64_t numel,
    const int8_t* __restrict__ in,
    int8_t* __restrict__ out,
    float in_scale, float in_zp,
    float inv_out_scale, float out_zp,
    int has_min, int has_max,
    float min_value, float max_value) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    float y = (static_cast<float>(in[i]) - in_zp) * in_scale;
    if (has_min) y = fmaxf(y, min_value);
    if (has_max) y = fminf(y, max_value);
    const float q = rintf(y * inv_out_scale) + out_zp;
    out[i] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, q)));
}

} // namespace

Tensor quantized_add_cuda(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cuda_impl(a, b, a_scale, a_zero_point, b_scale,
                                      b_zero_point, out_scale, out_zero_point,
                                      0);
}

Tensor quantized_sub_cuda(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cuda_impl(a, b, a_scale, a_zero_point, b_scale,
                                      b_zero_point, out_scale, out_zero_point,
                                      1);
}

Tensor quantized_mul_cuda(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cuda_impl(a, b, a_scale, a_zero_point, b_scale,
                                      b_zero_point, out_scale, out_zero_point,
                                      2);
}

Tensor quantized_div_cuda(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cuda_impl(a, b, a_scale, a_zero_point, b_scale,
                                      b_zero_point, out_scale, out_zero_point,
                                      3);
}

Tensor quantized_clamp_cuda(
    const Tensor& self, double self_scale, int64_t self_zero_point,
    double out_scale, int64_t out_zero_point,
    std::optional<Scalar> min, std::optional<Scalar> max) {
    if (self.dtype() != DType::Int8) {
        TP_THROW(TypeError, "quantized_clamp(): expected an Int8 tensor");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_clamp(): out_scale must be positive");
    }
    const Tensor sc = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::QInt8, self.device());

    const int64_t numel = self.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantized_clamp_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, sc.data_ptr<int8_t>(), out.data_ptr<int8_t>(),
        static_cast<float>(self_scale), static_cast<float>(self_zero_point),
        static_cast<float>(1.0 / out_scale),
        static_cast<float>(out_zero_point),
        min.has_value() ? 1 : 0, max.has_value() ? 1 : 0,
        min.has_value() ? static_cast<float>(min->toDouble()) : 0.0f,
        max.has_value() ? static_cast<float>(max->toDouble()) : 0.0f);
    checkCuda(cudaGetLastError(), "CUDA quantized_clamp kernel");
    out.impl()->set_quantizer(
        std::make_shared<PerTensorAffineQuantizer>(out_scale, out_zero_point));
    return out;
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
    const Tensor& input, const Tensor& weight, std::optional<Tensor> bias,
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
    Tensor x = dequantize_per_tensor_cuda(input, input_scale, input_zero_point);
    Tensor w = dequantize_per_tensor_cuda(weight, weight_scale, weight_zero_point);
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
    out.impl()->set_quantizer(
        std::make_shared<PerTensorAffineQuantizer>(out_scale, out_zero_point));
    return out;
}

// ---------------------------------------------------------------------------
// Unsigned-byte and int32 quantization variants.  The rounding and clamping
// rules match the Int8 pair; only the storage width changes.
// ---------------------------------------------------------------------------

__global__ void quantize_codes_kernel(
    int64_t numel,
    const float* __restrict__ input,
    int64_t* __restrict__ codes,
    float scale,
    float zero_point,
    float quant_min,
    float quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float q = nearbyintf(input[i] / scale) + zero_point;
    codes[i] = static_cast<int64_t>(
        fminf(quant_max, fmaxf(quant_min, q)));
}

__global__ void cast_codes_to_bytes_kernel(
    int64_t numel,
    const int64_t* __restrict__ codes,
    uint8_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    out[i] = static_cast<uint8_t>(codes[i]);
}

__global__ void cast_codes_to_int32_kernel(
    int64_t numel,
    const int64_t* __restrict__ codes,
    int32_t* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    out[i] = static_cast<int32_t>(codes[i]);
}

__global__ void dequantize_uint8_kernel(
    int64_t numel,
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    float scale,
    float zero_point) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    output[i] = (static_cast<float>(input[i]) - zero_point) * scale;
}

__global__ void dequantize_int32_kernel(
    int64_t numel,
    const int32_t* __restrict__ input,
    float* __restrict__ output,
    float scale,
    float zero_point) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    output[i] = (static_cast<float>(input[i]) - zero_point) * scale;
}

Tensor quantize_per_tensor_quint8_cuda(const Tensor& self, double scale,
                                        int64_t zero_point, int64_t quant_min,
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
    Tensor input = self;
    if (input.dtype() != DType::Float32) {
        input = input.to(DType::Float32);
    }
    const Tensor ic = input.is_contiguous() ? input : input.contiguous();
    Tensor codes = Tensor::empty(self.shape(), DType::Int64, self.device());
    Tensor out = Tensor::empty(self.shape(), DType::QUInt8, self.device());
    const int64_t numel = self.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantize_codes_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, ic.data_ptr<float>(), codes.data_ptr<int64_t>(),
        static_cast<float>(scale), static_cast<float>(zero_point),
        static_cast<float>(quant_min), static_cast<float>(quant_max));
    checkCuda(cudaGetLastError(), "CUDA quantize_codes kernel");
    cast_codes_to_bytes_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, codes.data_ptr<int64_t>(), out.data_ptr<uint8_t>());
    checkCuda(cudaGetLastError(), "CUDA cast_codes_to_bytes kernel");
    out.impl()->set_quantizer(
        std::make_shared<PerTensorAffineQuantizer>(scale, zero_point));
    return out;
}

Tensor dequantize_per_tensor_quint8_cuda(const Tensor& self, double scale,
                                          int64_t zero_point) {
    if (self.dtype() != DType::QUInt8) {
        TP_THROW(TypeError, "dequantize(): expected a QUInt8 tensor");
    }
    if (!(scale > 0.0)) {
        TP_THROW(ValueError, "dequantize(): scale must be positive");
    }
    const Tensor ic = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());
    const int64_t numel = self.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    dequantize_uint8_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, ic.data_ptr<uint8_t>(), out.data_ptr<float>(),
        static_cast<float>(scale), static_cast<float>(zero_point));
    checkCuda(cudaGetLastError(), "CUDA dequantize_uint8 kernel");
    return out;
}

Tensor quantize_per_tensor_qint32_cuda(const Tensor& self, double scale,
                                        int64_t zero_point) {
    if (!(scale > 0.0)) {
        TP_THROW(ValueError, "quantize(): scale must be positive");
    }
    Tensor input = self;
    if (input.dtype() != DType::Float32) {
        input = input.to(DType::Float32);
    }
    const Tensor ic = input.is_contiguous() ? input : input.contiguous();
    Tensor codes = Tensor::empty(self.shape(), DType::Int64, self.device());
    Tensor out = Tensor::empty(self.shape(), DType::QInt32, self.device());
    const int64_t numel = self.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantize_codes_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, ic.data_ptr<float>(), codes.data_ptr<int64_t>(),
        static_cast<float>(scale), static_cast<float>(zero_point),
        -2147483648.0f, 2147483647.0f);
    checkCuda(cudaGetLastError(), "CUDA quantize_codes kernel");
    cast_codes_to_int32_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, codes.data_ptr<int64_t>(), out.data_ptr<int32_t>());
    checkCuda(cudaGetLastError(), "CUDA cast_codes_to_int32 kernel");
    out.impl()->set_quantizer(
        std::make_shared<PerTensorAffineQuantizer>(scale, zero_point));
    return out;
}

Tensor dequantize_per_tensor_qint32_cuda(const Tensor& self, double scale,
                                          int64_t zero_point) {
    if (self.dtype() != DType::QInt32) {
        TP_THROW(TypeError, "dequantize(): expected a QInt32 tensor");
    }
    if (!(scale > 0.0)) {
        TP_THROW(ValueError, "dequantize(): scale must be positive");
    }
    const Tensor ic = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());
    const int64_t numel = self.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    dequantize_int32_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, ic.data_ptr<int32_t>(), out.data_ptr<float>(),
        static_cast<float>(scale), static_cast<float>(zero_point));
    checkCuda(cudaGetLastError(), "CUDA dequantize_int32 kernel");
    return out;
}

// ---------------------------------------------------------------------------
// Fake quantization: map real values through the affine Int8 grid and back,
// with a cached in-range mask for the backward pass.  Rounding is
// round-half-even; the raw (pre-clamp) grid position decides the mask.
// Compute runs in float (double for Float64 inputs); the store type keeps
// the input dtype.
// ---------------------------------------------------------------------------

namespace {

constexpr double kSmallScaleThreshold = 6.1e-5;

struct QParams {
    double scale;
    int64_t zero_point;
};

// Host-side qparams derivation from a real range [min, max] over the grid
// [qmin, qmax]: widen to contain 0, repair degenerate/too-small scales,
// then nudge the zero point into the grid with round-half-even.
QParams choose_qparams_host(double min, double max, int64_t qmin,
                            int64_t qmax, bool preserve_sparsity) {
    TP_CHECK(min <= max, "choose qparams: min must be <= max");
    if (min < 0 && max > 0 && preserve_sparsity) {
        const int64_t symmetric_qmin = -((qmax - qmin) / 2 + 1);
        const int64_t symmetric_qmax = (qmax - qmin) / 2;
        const double max_scale = std::max(
            std::fabs(min / static_cast<double>(symmetric_qmin)),
            std::fabs(max / static_cast<double>(symmetric_qmax)));
        min = max_scale * static_cast<double>(symmetric_qmin);
        max = max_scale * static_cast<double>(symmetric_qmax);
    }
    min = std::min(min, 0.0);
    max = std::max(max, 0.0);
    TP_CHECK(qmin < qmax, "choose qparams: qmin must be < qmax");
    double scale = (max - min) / static_cast<double>(qmax - qmin);
    if (static_cast<float>(scale) == 0.0f ||
        std::isinf(1.0f / static_cast<float>(scale))) {
        scale = 0.1;
    }
    if (scale < kSmallScaleThreshold) {
        const double org_scale = scale;
        scale = kSmallScaleThreshold;
        if (min == 0.0) {
            max = kSmallScaleThreshold * static_cast<double>(qmax - qmin);
        } else if (max == 0.0) {
            min = -kSmallScaleThreshold * static_cast<double>(qmax - qmin);
        } else {
            const double amplifier = kSmallScaleThreshold / org_scale;
            min *= amplifier;
            max *= amplifier;
        }
    }
    const double zero_point_from_min =
        static_cast<double>(qmin) - min / scale;
    const double zero_point_from_max =
        static_cast<double>(qmax) - max / scale;
    const double zero_point_from_min_error =
        std::abs(static_cast<double>(qmin)) - std::abs(min / scale);
    const double zero_point_from_max_error =
        std::abs(static_cast<double>(qmax)) - std::abs(max / scale);
    double initial_zero_point =
        zero_point_from_min_error < zero_point_from_max_error
            ? zero_point_from_min
            : zero_point_from_max;
    if (min < 0 && max > 0 && preserve_sparsity) {
        initial_zero_point = static_cast<double>(qmin + qmax) / 2.0;
    }
    int64_t nudged_zero_point = 0;
    if (initial_zero_point < static_cast<double>(qmin)) {
        nudged_zero_point = qmin;
    } else if (initial_zero_point > static_cast<double>(qmax)) {
        nudged_zero_point = qmax;
    } else {
        nudged_zero_point =
            static_cast<int64_t>(std::nearbyint(initial_zero_point));
    }
    return {scale, nudged_zero_point};
}

void check_fake_quant_range(int64_t zero_point, int64_t quant_min,
                            int64_t quant_max) {
    if (quant_min > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): quant_min must be <= quant_max");
    }
    if (zero_point < quant_min || zero_point > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): zero_point must be between quant_min and "
                 "quant_max");
    }
}

void check_real_dtype(const Tensor& self, const char* op) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError,
                 std::string(op) + ": expected a floating point tensor, got " +
                     toString(self.dtype()));
    }
}

template <typename T>
__global__ void fake_quant_per_tensor_kernel(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, float scale, int64_t zero_point,
    int64_t quant_min, int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float inv_scale = 1.0f / scale;
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale) +
                      static_cast<float>(zero_point);
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) -
                                static_cast<float>(zero_point)) * scale);
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

__global__ void fake_quant_per_tensor_kernel_double(
    int64_t numel, const double* __restrict__ input,
    double* __restrict__ output, bool* __restrict__ mask, double scale,
    int64_t zero_point, int64_t quant_min, int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const double inv_scale = 1.0 / scale;
    const double raw = nearbyint(input[i] * inv_scale) +
                       static_cast<double>(zero_point);
    const int64_t q = static_cast<int64_t>(fmin(
        static_cast<double>(quant_max),
        fmax(static_cast<double>(quant_min), raw)));
    output[i] = (static_cast<double>(q) - static_cast<double>(zero_point)) *
                scale;
    mask[i] = raw >= static_cast<double>(quant_min) &&
              raw <= static_cast<double>(quant_max);
}

void launch_fake_quant_per_tensor(const Tensor& input, Tensor& out,
                                  Tensor& mask, double scale,
                                  int64_t zero_point, int64_t quant_min,
                                  int64_t quant_max) {
    const int64_t numel = input.numel();
    if (numel == 0) return;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            fake_quant_per_tensor_kernel<float><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<float>(), out.data_ptr<float>(),
                mask.data_ptr<bool>(), static_cast<float>(scale), zero_point,
                quant_min, quant_max);
            break;
        case DType::Float64:
            fake_quant_per_tensor_kernel_double<<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<double>(), out.data_ptr<double>(),
                mask.data_ptr<bool>(), scale, zero_point, quant_min,
                quant_max);
            break;
        case DType::Float16:
            fake_quant_per_tensor_kernel<Half><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                mask.data_ptr<bool>(), static_cast<float>(scale), zero_point,
                quant_min, quant_max);
            break;
        case DType::BFloat16:
            fake_quant_per_tensor_kernel<BFloat16><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<BFloat16>(), out.data_ptr<BFloat16>(),
                mask.data_ptr<bool>(), static_cast<float>(scale), zero_point,
                quant_min, quant_max);
            break;
        default:
            TP_THROW(TypeError, "fake_quantize_per_tensor_affine(): "
                                "unsupported input dtype");
    }
    checkCuda(cudaGetLastError(), "CUDA fake_quantize_per_tensor kernel");
}

template <typename T>
__global__ void fake_quant_tensor_qparams_kernel(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, const float* __restrict__ scale,
    const int32_t* __restrict__ zero_point,
    const int64_t* __restrict__ fake_quant_enabled, int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    if (*fake_quant_enabled == 0) {
        output[i] = input[i];
        mask[i] = true;
        return;
    }
    const float inv_scale = 1.0f / (*scale);
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale) +
                      static_cast<float>(*zero_point);
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) -
                                static_cast<float>(*zero_point)) * (*scale));
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

template <typename T>
__global__ void fake_quant_tensor_qparams_kernel_floatzp(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, const float* __restrict__ scale,
    const float* __restrict__ zero_point,
    const int64_t* __restrict__ fake_quant_enabled, int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    if (*fake_quant_enabled == 0) {
        output[i] = input[i];
        mask[i] = true;
        return;
    }
    const float inv_scale = 1.0f / (*scale);
    // A floating zero point folds the shift into the rounding itself.
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale +
                                 (*zero_point));
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) - (*zero_point)) *
                               (*scale));
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

template <typename T, typename ZPT>
__global__ void fake_quant_tensor_qparams_kernel_double_typed(
    int64_t numel, const double* __restrict__ input,
    double* __restrict__ output, bool* __restrict__ mask,
    const double* __restrict__ scale, const ZPT* __restrict__ zero_point,
    const int64_t* __restrict__ fake_quant_enabled, int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    if (*fake_quant_enabled == 0) {
        output[i] = input[i];
        mask[i] = true;
        return;
    }
    const double inv_scale = 1.0 / (*scale);
    const double zpv = static_cast<double>(*zero_point);
    const double raw = nearbyint(input[i] * inv_scale) + zpv;
    const int64_t q = static_cast<int64_t>(fmin(
        static_cast<double>(quant_max),
        fmax(static_cast<double>(quant_min), raw)));
    output[i] = (static_cast<double>(q) - zpv) * (*scale);
    mask[i] = raw >= static_cast<double>(quant_min) &&
              raw <= static_cast<double>(quant_max);
}

void launch_fake_quant_tensor_qparams(const Tensor& input, Tensor& out,
                                      Tensor& mask, const Tensor& scale,
                                      const Tensor& zero_point,
                                      const Tensor& fake_quant_enabled,
                                      int64_t quant_min, int64_t quant_max) {
    const int64_t numel = input.numel();
    if (numel == 0) return;
    Tensor sc = scale.to(DType::Float32).contiguous();
    const bool zp_float = !isIntegralType(zero_point.dtype());
    Tensor zpi = zp_float ? zero_point.to(DType::Float32).contiguous()
                          : zero_point.to(DType::Int32).contiguous();
    Tensor fq = fake_quant_enabled.to(DType::Int64).contiguous();
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (input.dtype() == DType::Float64) {
        Tensor sc64 = scale.to(DType::Float64).contiguous();
        Tensor zpi64 = zp_float
                           ? zero_point.to(DType::Float64).contiguous()
                           : zero_point.to(DType::Int64).contiguous();
        if (zp_float) {
            fake_quant_tensor_qparams_kernel_double_typed<
                double, double><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<double>(), out.data_ptr<double>(),
                mask.data_ptr<bool>(), sc64.data_ptr<double>(),
                zpi64.data_ptr<double>(), fq.data_ptr<int64_t>(), quant_min,
                quant_max);
        } else {
            fake_quant_tensor_qparams_kernel_double_typed<
                double, int64_t><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<double>(), out.data_ptr<double>(),
                mask.data_ptr<bool>(), sc64.data_ptr<double>(),
                zpi64.data_ptr<int64_t>(), fq.data_ptr<int64_t>(), quant_min,
                quant_max);
        }
        checkCuda(cudaGetLastError(), "CUDA fake_quantize tensor_qparams kernel");
        return;
    }
    switch (input.dtype()) {
        case DType::Float32:
            if (zp_float) {
                fake_quant_tensor_qparams_kernel_floatzp<float>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<float>(), out.data_ptr<float>(),
                        mask.data_ptr<bool>(), sc.data_ptr<float>(),
                        zpi.data_ptr<float>(), fq.data_ptr<int64_t>(),
                        quant_min, quant_max);
            } else {
                fake_quant_tensor_qparams_kernel<float>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<float>(), out.data_ptr<float>(),
                        mask.data_ptr<bool>(), sc.data_ptr<float>(),
                        zpi.data_ptr<int32_t>(), fq.data_ptr<int64_t>(),
                        quant_min, quant_max);
            }
            break;
        case DType::Float16:
            if (zp_float) {
                fake_quant_tensor_qparams_kernel_floatzp<Half>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                        mask.data_ptr<bool>(), sc.data_ptr<float>(),
                        zpi.data_ptr<float>(), fq.data_ptr<int64_t>(),
                        quant_min, quant_max);
            } else {
                fake_quant_tensor_qparams_kernel<Half>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                        mask.data_ptr<bool>(), sc.data_ptr<float>(),
                        zpi.data_ptr<int32_t>(), fq.data_ptr<int64_t>(),
                        quant_min, quant_max);
            }
            break;
        case DType::BFloat16:
            if (zp_float) {
                fake_quant_tensor_qparams_kernel_floatzp<BFloat16>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<BFloat16>(),
                        out.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                        sc.data_ptr<float>(), zpi.data_ptr<float>(),
                        fq.data_ptr<int64_t>(), quant_min, quant_max);
            } else {
                fake_quant_tensor_qparams_kernel<BFloat16>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<BFloat16>(),
                        out.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                        sc.data_ptr<float>(), zpi.data_ptr<int32_t>(),
                        fq.data_ptr<int64_t>(), quant_min, quant_max);
            }
            break;
        default:
            TP_THROW(TypeError, "fake_quantize_per_tensor_affine(): "
                                "unsupported input dtype");
    }
    checkCuda(cudaGetLastError(), "CUDA fake_quantize tensor_qparams kernel");
}

template <typename T>
__global__ void masked_grad_kernel(int64_t numel, const T* __restrict__ grad,
                                   const bool* __restrict__ mask,
                                   T* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    out[i] = mask[i] ? grad[i] : static_cast<T>(0);
}

Tensor masked_grad_cuda(const Tensor& grad, const Tensor& mask) {
    Tensor out = Tensor::empty(grad.shape(), grad.dtype(), grad.device());
    const int64_t numel = grad.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    switch (grad.dtype()) {
        case DType::Float32:
            masked_grad_kernel<float><<<blocks, threads, 0, stream>>>(
                numel, grad.data_ptr<float>(), mask.data_ptr<bool>(),
                out.data_ptr<float>());
            break;
        case DType::Float64:
            masked_grad_kernel<double><<<blocks, threads, 0, stream>>>(
                numel, grad.data_ptr<double>(), mask.data_ptr<bool>(),
                out.data_ptr<double>());
            break;
        case DType::Float16:
            masked_grad_kernel<Half><<<blocks, threads, 0, stream>>>(
                numel, grad.data_ptr<Half>(), mask.data_ptr<bool>(),
                out.data_ptr<Half>());
            break;
        case DType::BFloat16:
            masked_grad_kernel<BFloat16><<<blocks, threads, 0, stream>>>(
                numel, grad.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                out.data_ptr<BFloat16>());
            break;
        default:
            TP_THROW(TypeError, "fake_quantize backward: expected a "
                                "floating point gradient");
    }
    checkCuda(cudaGetLastError(), "CUDA fake_quantize backward kernel");
    return out;
}

// Learnable-qparams backward: dX is a straight-through inside the
// representable range and zero outside; dScale and dZeroPoint collect one
// contribution per element, scaled by grad_factor.
template <typename T>
__global__ void learnable_backward_kernel(
    int64_t numel, const T* __restrict__ x, const T* __restrict__ dy,
    T* __restrict__ dx, T* __restrict__ dscale, T* __restrict__ dzp,
    float scale, int64_t zero_point, int64_t quant_min, int64_t quant_max,
    float grad_factor) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float inv_scale = 1.0f / scale;
    const float dscale_small = static_cast<float>(quant_min - zero_point);
    const float dscale_big = static_cast<float>(quant_max - zero_point);
    const float xf = static_cast<float>(x[i]);
    const float dyf = static_cast<float>(dy[i]);
    const int64_t xq = static_cast<int64_t>(
                           nearbyintf(xf * inv_scale)) + zero_point;
    dx[i] = static_cast<T>(dyf * (xq >= quant_min && xq <= quant_max));
    const float xfq = static_cast<float>(
        (std::min<int64_t>(std::max<int64_t>(xq, quant_min), quant_max) -
         zero_point) * scale);
    if (xq < quant_min || xq > quant_max) {
        dscale[i] = static_cast<T>(
            (dyf * ((xq < quant_min) ? dscale_small : dscale_big)) *
            grad_factor);
        dzp[i] = static_cast<T>(dyf * (-1.0f) * scale * grad_factor);
    } else {
        dscale[i] = static_cast<T>(dyf * (xfq - xf) * inv_scale *
                                   grad_factor);
        dzp[i] = static_cast<T>(0);
    }
}

std::tuple<Tensor, Tensor, Tensor> learnable_backward_cuda(
    const Tensor& grad, const Tensor& x, double scale_val,
    int64_t zero_point_val, int64_t quant_min, int64_t quant_max,
    double grad_factor, DType out_dtype) {
    const int64_t numel = x.numel();
    Tensor dx = Tensor::empty(x.shape(), out_dtype, x.device());
    Tensor dscale_vec = Tensor::empty(x.shape(), out_dtype, x.device());
    Tensor dzp_vec = Tensor::empty(x.shape(), out_dtype, x.device());
    if (numel == 0) {
        return {std::move(dx), std::move(dscale_vec.sum().reshape({1})),
                std::move(dzp_vec.sum().reshape({1}))};
    }
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (out_dtype == DType::Float64) {
        learnable_backward_kernel<double><<<blocks, threads, 0, stream>>>(
            numel, x.data_ptr<double>(), grad.data_ptr<double>(),
            dx.data_ptr<double>(), dscale_vec.data_ptr<double>(),
            dzp_vec.data_ptr<double>(), static_cast<float>(scale_val),
            zero_point_val, quant_min, quant_max,
            static_cast<float>(grad_factor));
    } else {
        learnable_backward_kernel<float><<<blocks, threads, 0, stream>>>(
            numel, x.data_ptr<float>(), grad.data_ptr<float>(),
            dx.data_ptr<float>(), dscale_vec.data_ptr<float>(),
            dzp_vec.data_ptr<float>(), static_cast<float>(scale_val),
            zero_point_val, quant_min, quant_max,
            static_cast<float>(grad_factor));
    }
    checkCuda(cudaGetLastError(), "CUDA learnable backward kernel");
    return {std::move(dx), std::move(dscale_vec.sum().reshape({1})),
            std::move(dzp_vec.sum().reshape({1}))};
}

std::tuple<Tensor, Tensor, DType> promote_learnable_pair(const Tensor& grad,
                                                         const Tensor& x) {
    check_real_dtype(x, "fake_quantize backward");
    if (!isFloatingType(grad.dtype())) {
        TP_THROW(TypeError,
                 "fake_quantize backward: expected a floating point grad");
    }
    DType compute = x.dtype();
    if (compute == DType::Float16 || compute == DType::BFloat16) {
        compute = DType::Float32;
    }
    Tensor xc = (x.dtype() == compute) ? x : x.to(compute);
    Tensor gc = (grad.dtype() == compute) ? grad : grad.to(compute);
    xc = xc.is_contiguous() ? xc : xc.contiguous();
    gc = gc.is_contiguous() ? gc : gc.contiguous();
    if (xc.numel() != gc.numel()) {
        TP_THROW(ValueError,
                 "fake_quantize backward: X and dY must have the same number "
                 "of elements");
    }
    return {gc, xc, compute};
}

template <typename T>
__global__ void learnable_backward_per_channel_kernel(
    int64_t numel, const T* __restrict__ x, const T* __restrict__ dy,
    T* __restrict__ dx, T* __restrict__ dscale, T* __restrict__ dzp,
    int64_t stride_on_axis, int64_t channels, const float* __restrict__ scales,
    const float* __restrict__ zero_points, int64_t quant_min,
    int64_t quant_max, float grad_factor) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    const float inv_scale = 1.0f / scales[c];
    const float zpf = zero_points[c];
    const float dscale_small = static_cast<float>(quant_min) - zpf;
    const float dscale_big = static_cast<float>(quant_max) - zpf;
    const float xf = static_cast<float>(x[i]);
    const float dyf = static_cast<float>(dy[i]);
    const int64_t xq = static_cast<int64_t>(nearbyintf(xf * inv_scale)) +
                       static_cast<int64_t>(zpf);
    dx[i] = static_cast<T>(dyf * (xq >= quant_min && xq <= quant_max));
    const float xfq = static_cast<float>(
        (std::min<int64_t>(std::max<int64_t>(xq, quant_min), quant_max)) -
        static_cast<int64_t>(zpf)) * scales[c];
    if (xq < quant_min || xq > quant_max) {
        dscale[i] = static_cast<T>(
            (dyf * ((xq < quant_min) ? dscale_small : dscale_big)) *
            grad_factor);
        dzp[i] = static_cast<T>(dyf * (-1.0f) * scales[c] * grad_factor);
    } else {
        dscale[i] = static_cast<T>(dyf * (xfq - xf) * inv_scale *
                                   grad_factor);
        dzp[i] = static_cast<T>(0);
    }
}

} // namespace

std::tuple<Tensor, Tensor> fake_quantize_per_tensor_affine_cachemask_cuda(
    const Tensor& self, double scale, int64_t zero_point, int64_t quant_min,
    int64_t quant_max) {
    check_real_dtype(self, "fake_quantize_per_tensor_affine");
    check_fake_quant_range(zero_point, quant_min, quant_max);
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), self.dtype(), self.device());
    Tensor mask = Tensor::empty(self.shape(), DType::Bool, self.device());
    launch_fake_quant_per_tensor(input, out, mask, scale, zero_point,
                                 quant_min, quant_max);
    return {std::move(out), std::move(mask)};
}

Tensor fake_quantize_per_tensor_affine_cuda(const Tensor& self, double scale,
                                            int64_t zero_point,
                                            int64_t quant_min,
                                            int64_t quant_max) {
    return std::get<0>(fake_quantize_per_tensor_affine_cachemask_cuda(
        self, scale, zero_point, quant_min, quant_max));
}

std::tuple<Tensor, Tensor>
_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    const Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max) {
    check_real_dtype(self, "fake_quantize_per_tensor_affine");
    if (quant_min > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): quant_min must be <= quant_max");
    }
    TP_CHECK(scale.numel() == 1 && zero_point.numel() == 1 &&
                 fake_quant_enabled.numel() == 1,
             "fake_quantize(): scale, zero_point and the fake-quant flag "
             "must be 1-element tensors");
    Tensor out = Tensor::empty(self.shape(), self.dtype(), self.device());
    Tensor mask = Tensor::empty(self.shape(), DType::Bool, self.device());
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    const int64_t numel = input.numel();
    if (numel == 0) return {std::move(out), std::move(mask)};
    launch_fake_quant_tensor_qparams(input, out, mask, scale, zero_point,
                                     fake_quant_enabled, quant_min,
                                     quant_max);
    return {std::move(out), std::move(mask)};
}

Tensor fake_quantize_per_tensor_affine_tensor_qparams_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max) {
    Tensor enabled = Tensor::full({1}, Scalar(static_cast<int64_t>(1)),
                                  DType::Int64, self.device());
    return std::get<0>(
        _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda(
            self, scale, zero_point, enabled, quant_min, quant_max));
}

Tensor fake_quantize_per_tensor_affine_cachemask_backward_cuda(
    const Tensor& grad, const Tensor& mask) {
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "fake_quantize backward: mask must be Bool");
    }
    if (mask.numel() != grad.numel()) {
        TP_THROW(ValueError,
                 "fake_quantize backward: mask and grad must have the same "
                 "number of elements");
    }
    if (grad.numel() == 0) return grad;
    const Tensor gc = grad.is_contiguous() ? grad : grad.contiguous();
    const Tensor mc = mask.is_contiguous() ? mask : mask.contiguous();
    return masked_grad_cuda(gc, mc);
}

Tensor _fake_quantize_learnable_per_tensor_affine_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max, double grad_factor) {
    (void)grad_factor;
    check_real_dtype(self, "fake_quantize_per_tensor_affine");
    TP_CHECK(scale.numel() == 1 && zero_point.numel() == 1,
             "fake_quantize(): scale and zero_point must be 1-element "
             "tensors");
    const double scale_val = scale.item().toDouble();
    double zp_fp = std::nearbyint(zero_point.item().toDouble());
    zp_fp = std::min(static_cast<double>(quant_max),
                     std::max(static_cast<double>(quant_min), zp_fp));
    return fake_quantize_per_tensor_affine_cuda(
        self, scale_val, static_cast<int64_t>(zp_fp), quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_tensor_affine_backward_cuda(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t quant_min, int64_t quant_max,
    double grad_factor) {
    TP_CHECK(scale.numel() == 1 && zero_point.numel() == 1,
             "fake_quantize backward: scale and zero_point must be "
             "1-element tensors");
    double zp_fp = zero_point.item().toDouble() + 0.5;
    zp_fp = std::min(static_cast<double>(quant_max),
                     std::max(static_cast<double>(quant_min), zp_fp));
    if (quant_min > 0 || quant_max < 0) {
        TP_THROW(ValueError,
                 "fake_quantize backward: the quantization range must "
                 "include 0");
    }
    const int64_t zero_point_val = static_cast<int64_t>(zp_fp);
    if (zero_point_val < quant_min || zero_point_val > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize backward: zero_point out of the quantized "
                 "range");
    }
    if (self.numel() == 0) {
        return {self, scale, zero_point};
    }
    auto promoted = promote_learnable_pair(grad, self);
    return learnable_backward_cuda(
        std::get<0>(promoted), std::get<1>(promoted),
        scale.item().toDouble(), zero_point_val, quant_min, quant_max,
        grad_factor, std::get<2>(promoted));
}

// ---------------------------------------------------------------------------
// Per-channel fake quantization: scale/zero_point arrays are indexed by the
// channel of each element under the axis-major layout.
// ---------------------------------------------------------------------------

namespace {

template <typename T>
__global__ void fake_quant_per_channel_kernel(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, int64_t stride_on_axis, int64_t channels,
    const float* __restrict__ scales,
    const int64_t* __restrict__ zero_points, int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    const float inv_scale = 1.0f / scales[c];
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale) +
                      static_cast<float>(zero_points[c]);
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) -
                                static_cast<float>(zero_points[c])) *
                               scales[c]);
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

template <typename T>
__global__ void fake_quant_per_channel_kernel_floatzp(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, int64_t stride_on_axis, int64_t channels,
    const float* __restrict__ scales, const float* __restrict__ zero_points,
    int64_t quant_min, int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    const float inv_scale = 1.0f / scales[c];
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale +
                                 zero_points[c]);
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) - zero_points[c]) *
                               scales[c]);
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

} // namespace

std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max) {
    check_real_dtype(self, "fake_quantize_per_channel_affine");
    TP_CHECK(scale.dim() == 1 && zero_point.dim() == 1,
             "fake_quantize(): scale and zero_point must be 1-D tensors");
    TP_CHECK(scale.numel() == zero_point.numel(),
             "fake_quantize(): scale and zero_point must have the same "
             "size");
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "fake_quantize(): axis out of range");
    }
    TP_CHECK(scale.numel() == self.size(axis),
             "fake_quantize(): scale size must match the quantized "
             "dimension");
    if (quant_min > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): quant_min must be <= quant_max");
    }
    if (isIntegralType(zero_point.dtype())) {
        Tensor zpc = zero_point.to(DType::Int64).contiguous();
        const int64_t* zp = zpc.data_ptr<int64_t>();
        for (int64_t i = 0; i < zpc.numel(); ++i) {
            if (zp[i] < quant_min || zp[i] > quant_max) {
                TP_THROW(ValueError,
                         "fake_quantize(): zero_point out of the quantized "
                         "range");
            }
        }
    }

    Tensor out = Tensor::empty(self.shape(), self.dtype(), self.device());
    Tensor mask = Tensor::empty(self.shape(), DType::Bool, self.device());
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor sc = scale.to(DType::Float32).contiguous();
    const bool zp_float = !isIntegralType(zero_point.dtype());
    Tensor zpf = zp_float ? zero_point.to(DType::Float32).contiguous()
                          : zero_point.to(DType::Int64).contiguous();
    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < input.dim(); ++d) {
        stride_on_axis *= input.size(d);
    }
    const int64_t channels = scale.numel();
    const int64_t numel = input.numel();
    if (numel == 0) return {std::move(out), std::move(mask)};
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            if (zp_float) {
                fake_quant_per_channel_kernel_floatzp<float>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<float>(), out.data_ptr<float>(),
                        mask.data_ptr<bool>(), stride_on_axis, channels,
                        sc.data_ptr<float>(), zpf.data_ptr<float>(),
                        quant_min, quant_max);
            } else {
                fake_quant_per_channel_kernel<float>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<float>(), out.data_ptr<float>(),
                        mask.data_ptr<bool>(), stride_on_axis, channels,
                        sc.data_ptr<float>(), zpf.data_ptr<int64_t>(),
                        quant_min, quant_max);
            }
            break;
        case DType::Float16:
            if (zp_float) {
                fake_quant_per_channel_kernel_floatzp<Half>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                        mask.data_ptr<bool>(), stride_on_axis, channels,
                        sc.data_ptr<float>(), zpf.data_ptr<float>(),
                        quant_min, quant_max);
            } else {
                fake_quant_per_channel_kernel<Half>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                        mask.data_ptr<bool>(), stride_on_axis, channels,
                        sc.data_ptr<float>(), zpf.data_ptr<int64_t>(),
                        quant_min, quant_max);
            }
            break;
        case DType::BFloat16:
            if (zp_float) {
                fake_quant_per_channel_kernel_floatzp<BFloat16>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<BFloat16>(),
                        out.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                        stride_on_axis, channels, sc.data_ptr<float>(),
                        zpf.data_ptr<float>(), quant_min, quant_max);
            } else {
                fake_quant_per_channel_kernel<BFloat16>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<BFloat16>(),
                        out.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                        stride_on_axis, channels, sc.data_ptr<float>(),
                        zpf.data_ptr<int64_t>(), quant_min, quant_max);
            }
            break;
        default:
            TP_THROW(TypeError, "fake_quantize_per_channel_affine(): "
                                "unsupported input dtype");
    }
    checkCuda(cudaGetLastError(), "CUDA fake_quantize_per_channel kernel");
    return {std::move(out), std::move(mask)};
}

Tensor fake_quantize_per_channel_affine_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max) {
    return std::get<0>(fake_quantize_per_channel_affine_cachemask_cuda(
        self, scale, zero_point, axis, quant_min, quant_max));
}

Tensor fake_quantize_per_channel_affine_cachemask_backward_cuda(
    const Tensor& grad, const Tensor& mask) {
    return fake_quantize_per_tensor_affine_cachemask_backward_cuda(grad,
                                                                   mask);
}

Tensor _fake_quantize_learnable_per_channel_affine_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
    (void)grad_factor;
    Tensor zp = zero_point.to(DType::Float32)
                    .round()
                    .clamp(Scalar(static_cast<int64_t>(quant_min)),
                           Scalar(static_cast<int64_t>(quant_max)))
                    .to(DType::Int64);
    return fake_quantize_per_channel_affine_cuda(
        self, scale.to(DType::Float32), zp, axis, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_channel_affine_backward_cuda(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t axis, int64_t quant_min,
    int64_t quant_max, double grad_factor) {
    TP_CHECK(scale.dim() == 1 && zero_point.dim() == 1 &&
                 scale.numel() == zero_point.numel(),
             "fake_quantize backward: scale and zero_point must be 1-D "
             "tensors of the same size");
    if (quant_min > 0 || quant_max < 0) {
        TP_THROW(ValueError,
                 "fake_quantize backward: the quantization range must "
                 "include 0");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "fake_quantize backward: axis out of range");
    }
    TP_CHECK(scale.numel() == self.size(axis),
             "fake_quantize backward: scale size must match the quantized "
             "dimension");
    if (self.numel() == 0) {
        return {self, scale, zero_point};
    }

    auto promoted = promote_learnable_pair(grad, self);
    const Tensor& x = std::get<1>(promoted);
    const DType compute = std::get<2>(promoted);

    Tensor zp_f = zero_point.to(DType::Float32)
                      .round()
                      .clamp(Scalar(static_cast<int64_t>(quant_min)),
                             Scalar(static_cast<int64_t>(quant_max)))
                      .contiguous();
    Tensor sc = scale.to(DType::Float32).contiguous();
    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < x.dim(); ++d) stride_on_axis *= x.size(d);
    const int64_t channels = scale.numel();
    const int64_t numel = x.numel();

    Tensor dx = Tensor::empty(x.shape(), compute, x.device());
    Tensor dscale_vec = Tensor::empty(x.shape(), compute, x.device());
    Tensor dzp_vec = Tensor::empty(x.shape(), compute, x.device());
    if (numel > 0) {
        const cudaStream_t stream = getCurrentCUDAStream().stream();
        const int threads = 256;
        const int blocks = static_cast<int>((numel + threads - 1) / threads);
        if (compute == DType::Float64) {
            TP_THROW(TypeError,
                     "fake_quantize backward: Float64 per-channel backward "
                     "is not supported on this device");
        }
        learnable_backward_per_channel_kernel<float>
            <<<blocks, threads, 0, stream>>>(
                numel, x.data_ptr<float>(),
                std::get<0>(promoted).data_ptr<float>(),
                dx.data_ptr<float>(), dscale_vec.data_ptr<float>(),
                dzp_vec.data_ptr<float>(), stride_on_axis, channels,
                sc.data_ptr<float>(), zp_f.data_ptr<float>(), quant_min,
                quant_max, static_cast<float>(grad_factor));
        checkCuda(cudaGetLastError(),
                  "CUDA learnable per-channel backward kernel");
    }
    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < x.dim(); ++d) {
        if (d != axis) reduce_dims.push_back(d);
    }
    Tensor dscale = dscale_vec.sum(reduce_dims, false);
    Tensor dzp = dzp_vec.sum(reduce_dims, false);
    return {std::move(dx), std::move(dscale), std::move(dzp)};
}

// ---------------------------------------------------------------------------
// Dynamic quantization and fused moving-average observer + fake quant.
// ---------------------------------------------------------------------------

namespace {

template <typename T>
__global__ void dynamic_quantize_kernel(int64_t numel,
                                        const float* __restrict__ input,
                                        T* __restrict__ output, float scale,
                                        int64_t zero_point, int64_t qmin,
                                        int64_t qmax) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float q = nearbyintf(input[i] / scale) +
                    static_cast<float>(zero_point);
    output[i] = static_cast<T>(fminf(static_cast<float>(qmax),
                                     fmaxf(static_cast<float>(qmin), q)));
}

std::pair<double, double> tensor_min_max(const Tensor& self) {
    const Tensor input = self.contiguous();
    auto mm = Tensor::aminmax(input.reshape({input.numel()}), {}, false);
    return {std::get<0>(mm).item().toDouble(),
            std::get<1>(mm).item().toDouble()};
}

// Device-side moving average of the running min/max state.
__global__ void moving_average_minmax_kernel(
    int64_t size, const float* __restrict__ x_min,
    const float* __restrict__ x_max, float* __restrict__ running_min,
    float* __restrict__ running_max, float averaging_const) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    running_min[i] = ::isinf(running_min[i])
                         ? x_min[i]
                         : running_min[i] +
                               averaging_const * (x_min[i] - running_min[i]);
    running_max[i] = ::isinf(running_max[i])
                         ? x_max[i]
                         : running_max[i] +
                               averaging_const * (x_max[i] - running_max[i]);
}

// Derives qparams from the running range on device; entries are only
// written while the fake-quant flag is on.
__global__ void choose_qparams_kernel(const int64_t* __restrict__ fake_on,
                                      const float* __restrict__ x_min,
                                      const float* __restrict__ x_max,
                                      int64_t qmin, int64_t qmax,
                                      bool preserve_sparsity, int64_t size,
                                      float* __restrict__ scale,
                                      int64_t* __restrict__ zero_point) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size || *fake_on == 0) return;
    float min_val = x_min[i];
    float max_val = x_max[i];
    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
        const int64_t symmetric_qmin = -((qmax - qmin) / 2 + 1);
        const int64_t symmetric_qmax = (qmax - qmin) / 2;
        const double max_scale = fmax(
            fabs(min_val / static_cast<double>(symmetric_qmin)),
            fabs(max_val / static_cast<double>(symmetric_qmax)));
        min_val = static_cast<float>(max_scale *
                                     static_cast<double>(symmetric_qmin));
        max_val = static_cast<float>(max_scale *
                                     static_cast<double>(symmetric_qmax));
    }
    min_val = fminf(min_val, 0.0f);
    max_val = fmaxf(max_val, 0.0f);
    float sc = static_cast<float>(
        (static_cast<double>(max_val) - static_cast<double>(min_val)) /
        static_cast<double>(qmax - qmin));
    if (sc == 0.0f || ::isinf(1.0f / sc)) sc = 0.1f;
    if (sc < static_cast<float>(kSmallScaleThreshold)) {
        const float org = sc;
        sc = static_cast<float>(kSmallScaleThreshold);
        if (min_val == 0.0f) {
            max_val = sc * static_cast<float>(qmax - qmin);
        } else if (max_val == 0.0f) {
            min_val = -sc * static_cast<float>(qmax - qmin);
        } else {
            const float amplifier = sc / org;
            min_val *= amplifier;
            max_val *= amplifier;
        }
    }
    const double zp_from_min =
        static_cast<double>(qmin) -
        static_cast<double>(min_val) / static_cast<double>(sc);
    const double zp_from_max =
        static_cast<double>(qmax) -
        static_cast<double>(max_val) / static_cast<double>(sc);
    const double err_min = std::abs(static_cast<double>(qmin)) -
                           std::abs(static_cast<double>(min_val) /
                                    static_cast<double>(sc));
    const double err_max = std::abs(static_cast<double>(qmax)) -
                           std::abs(static_cast<double>(max_val) /
                                    static_cast<double>(sc));
    double initial_zp = err_min < err_max ? zp_from_min : zp_from_max;
    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
        initial_zp = static_cast<double>(qmin + qmax) / 2.0;
    }
    int64_t nudged = 0;
    if (initial_zp < static_cast<double>(qmin)) {
        nudged = qmin;
    } else if (initial_zp > static_cast<double>(qmax)) {
        nudged = qmax;
    } else {
        nudged = static_cast<int64_t>(nearbyint(initial_zp));
    }
    scale[i] = sc;
    zero_point[i] = nudged;
}

void fill_or_resize_f32(Tensor& t, int64_t size, float value,
                        const Device& device) {
    if (t.numel() == 0) {
        t = Tensor::full({size}, Scalar(value), DType::Float32, device);
        return;
    }
    if (t.numel() != size) {
        t.resize_({size});
        t.fill_(Scalar(value));
        return;
    }
    t.fill_(Scalar(value));
}

} // namespace

Tensor quantize_per_tensor_dynamic_cuda(const Tensor& self, DType dtype,
                                        bool reduce_range) {
    if (dtype != DType::QInt8 && dtype != DType::QUInt8 &&
        dtype != DType::Float16) {
        TP_THROW(TypeError,
                 "quantize_per_tensor_dynamic(): only QInt8, QUInt8 and "
                 "Float16 outputs are supported");
    }
    check_real_dtype(self, "quantize_per_tensor_dynamic");
    if (dtype == DType::Float16) {
        return self.contiguous().to(DType::Float16);
    }
    const auto mm = tensor_min_max(self);
    int64_t qmin = (dtype == DType::QInt8) ? -128 : 0;
    int64_t qmax = (dtype == DType::QInt8) ? 127 : 255;
    if (reduce_range) {
        qmin /= 2;
        qmax /= 2;
    }
    const QParams qp = choose_qparams_host(mm.first, mm.second, qmin, qmax,
                                           /*preserve_sparsity=*/false);

    const Tensor input = self.contiguous();
    const Tensor input_f =
        (input.dtype() == DType::Float32) ? input : input.to(DType::Float32);
    Tensor out = Tensor::empty(self.shape(), dtype, self.device());
    const int64_t numel = input.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (dtype == DType::QInt8) {
        dynamic_quantize_kernel<int8_t><<<blocks, threads, 0, stream>>>(
            numel, input_f.data_ptr<float>(), out.data_ptr<int8_t>(),
            static_cast<float>(qp.scale), qp.zero_point, qmin, qmax);
    } else {
        dynamic_quantize_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
            numel, input_f.data_ptr<float>(), out.data_ptr<uint8_t>(),
            static_cast<float>(qp.scale), qp.zero_point, qmin, qmax);
    }
    checkCuda(cudaGetLastError(), "CUDA quantize_per_tensor_dynamic kernel");
    out.impl()->set_quantizer(
        std::make_shared<PerTensorAffineQuantizer>(qp.scale, qp.zero_point));
    return out;
}

std::tuple<double, int64_t> _choose_qparams_per_tensor_cuda(
    const Tensor& self, bool reduce_range) {
    check_real_dtype(self, "_choose_qparams_per_tensor");
    const auto mm = tensor_min_max(self);
    int64_t qmin = 0;
    int64_t qmax = 255;
    if (reduce_range) {
        qmin /= 2;
        qmax /= 2;
    }
    const QParams qp =
        choose_qparams_host(mm.first, mm.second, qmin, qmax, false);
    return {qp.scale, qp.zero_point};
}

// ---------------------------------------------------------------------------
// Tensor-level quantization metadata on CUDA.  The quantizer lives in host
// memory on the TensorImpl; only int_repr and _make_per_* touch codes.
// ---------------------------------------------------------------------------

bool is_quantized_cuda(const Tensor& self) {
    return quantized::is_quantized(self);
}

int64_t qscheme_cuda(const Tensor& self) {
    quantized::require_quantized(self, "qscheme");
    return static_cast<int64_t>(
        quantized::quantizer_of(self)->qscheme());
}

double q_scale_cuda(const Tensor& self) {
    return quantized::q_scale(self);
}

int64_t q_zero_point_cuda(const Tensor& self) {
    return quantized::q_zero_point(self);
}

Tensor q_per_channel_scales_cuda(const Tensor& self) {
    return quantized::q_per_channel_scales(self);
}

Tensor q_per_channel_zero_points_cuda(const Tensor& self) {
    return quantized::q_per_channel_zero_points(self);
}

int64_t q_per_channel_axis_cuda(const Tensor& self) {
    return quantized::q_per_channel_axis(self);
}

Tensor int_repr_cuda(const Tensor& self) {
    quantized::require_quantized(self, "int_repr");
    return quantized::strip_quantizer(self).clone();
}

Tensor dequantize_self_cuda(const Tensor& self) {
    if (!quantized::is_quantized(self)) {
        return self;
    }
    const auto q = quantized::quantizer_of(self);
    if (q->qscheme() == QScheme::PerChannelAffine) {
        return dequantize_per_channel_cuda(self, q->scales(),
                                           q->zero_points(), q->axis());
    }
    switch (self.dtype()) {
        case DType::QInt8:
            return dequantize_per_tensor_cuda(self, q->scale(),
                                              q->zero_point());
        case DType::QUInt8:
            return dequantize_per_tensor_quint8_cuda(self, q->scale(),
                                                     q->zero_point());
        case DType::QInt32:
            return dequantize_per_tensor_qint32_cuda(self, q->scale(),
                                                     q->zero_point());
        default:
            TP_THROW(TypeError,
                     std::string("dequantize(): unsupported quantized dtype ") +
                         toString(self.dtype()));
    }
}

namespace {

DType quantized_dtype_of_codes_cuda(const Tensor& codes, const char* op) {
    switch (codes.dtype()) {
        case DType::Int8:
            return DType::QInt8;
        case DType::UInt8:
            return DType::QUInt8;
        case DType::Int32:
            return DType::QInt32;
        default:
            TP_THROW(TypeError,
                     std::string(op) +
                         ": expected an Int8/UInt8/Int32 code tensor, got " +
                         toString(codes.dtype()));
    }
}

} // namespace

Tensor _make_per_tensor_quantized_tensor_cuda(const Tensor& self, double scale,
                                              int64_t zero_point) {
    if (!(scale > 0.0)) {
        TP_THROW(ValueError,
                 "_make_per_tensor_quantized_tensor(): scale must be positive");
    }
    const DType qdt = quantized_dtype_of_codes_cuda(
        self, "_make_per_tensor_quantized_tensor");
    return quantized::make_qtensor(
        self.clone(),
        std::make_shared<PerTensorAffineQuantizer>(scale, zero_point), qdt);
}

Tensor _make_per_channel_quantized_tensor_cuda(const Tensor& self,
                                               const Tensor& scale,
                                               const Tensor& zero_point,
                                               int64_t axis) {
    if (scale.dim() != 1 || zero_point.shape() != scale.shape()) {
        TP_THROW(ValueError,
                 "_make_per_channel_quantized_tensor(): scales/zero_points "
                 "must be 1-D with equal sizes");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError,
                 "_make_per_channel_quantized_tensor(): axis out of range");
    }
    if (scale.size(0) != self.size(axis)) {
        TP_THROW(ValueError,
                 "_make_per_channel_quantized_tensor(): scales size must "
                 "match the quantized dimension");
    }
    Tensor sc = scale.to(DType::Float64).contiguous();
    const double* sp = sc.data_ptr<double>();
    for (int64_t i = 0; i < sc.numel(); ++i) {
        if (!(sp[i] > 0.0)) {
            TP_THROW(ValueError,
                     "_make_per_channel_quantized_tensor(): scales must be "
                     "positive");
        }
    }
    const DType qdt = quantized_dtype_of_codes_cuda(
        self, "_make_per_channel_quantized_tensor");
    return quantized::make_qtensor(
        self.clone(),
        std::make_shared<PerChannelAffineQuantizer>(
            sc, zero_point.to(DType::Int64).contiguous(), axis), qdt);
}

std::tuple<Tensor, Tensor> _fused_moving_avg_obs_fq_helper_cuda(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant) {
    if (ch_axis >= self.dim()) {
        TP_THROW(ValueError,
                 "fused_moving_avg_obs_fake_quant(): ch_axis must be < "
                 "self.dim()");
    }
    check_real_dtype(self, "fused_moving_avg_obs_fake_quant");
    const bool observe = observer_on.item().to<int64_t>() != 0;
    const bool fake_on = fake_quant_on.item().to<int64_t>() != 0;

    if (per_row_fake_quant) {
        Tensor y = self;
        if (self.dim() != 2) {
            std::vector<int64_t> dims(self.dim());
            for (int64_t d = 0; d < self.dim(); ++d) dims[d] = d;
            dims[ch_axis] = 0;
            dims[0] = ch_axis;
            y = self.permute(dims).flatten(1);
        }
        const int64_t size = self.size(ch_axis);
        if (running_min.numel() == 0) {
            fill_or_resize_f32(running_min, size,
                               std::numeric_limits<float>::infinity(),
                               self.device());
            fill_or_resize_f32(running_max, size,
                               -std::numeric_limits<float>::infinity(),
                               self.device());
            scale.resize_({size});
            zero_point.resize_({size});
        }
        if (observe) {
            auto mm = Tensor::aminmax(y.contiguous(), {1}, false);
            Tensor mn = std::get<0>(mm).to(DType::Float32).contiguous();
            Tensor mx = std::get<1>(mm).to(DType::Float32).contiguous();
            const cudaStream_t stream = getCurrentCUDAStream().stream();
            const int threads = 256;
            const int blocks = static_cast<int>((size + threads - 1) / threads);
            moving_average_minmax_kernel<<<blocks, threads, 0, stream>>>(
                size, mn.data_ptr<float>(), mx.data_ptr<float>(),
                running_min.data_ptr<float>(), running_max.data_ptr<float>(),
                static_cast<float>(averaging_const));
            checkCuda(cudaGetLastError(), "CUDA moving average kernel");
        }
        if (!fake_on) {
            Tensor mask = Tensor::full_like(self, 1, DType::Bool);
            return {self.clone(), std::move(mask)};
        }
        Tensor sc = Tensor::empty({size}, DType::Float32, self.device());
        Tensor zp = Tensor::empty({size}, DType::Int64, self.device());
        {
            const cudaStream_t stream = getCurrentCUDAStream().stream();
            const int threads = 256;
            const int blocks =
                static_cast<int>((size + threads - 1) / threads);
            Tensor fq = fake_quant_on.to(DType::Int64).contiguous();
            choose_qparams_kernel<<<blocks, threads, 0, stream>>>(
                fq.data_ptr<int64_t>(), running_min.data_ptr<float>(),
                running_max.data_ptr<float>(), quant_min, quant_max,
                symmetric_quant, size, sc.data_ptr<float>(),
                zp.data_ptr<int64_t>());
            checkCuda(cudaGetLastError(), "CUDA choose qparams kernel");
        }
        scale.copy_(sc);
        zero_point.copy_(zp);
        return fake_quantize_per_channel_affine_cachemask_cuda(
            self, sc, zp, ch_axis, quant_min, quant_max);
    }

    if (observe) {
        auto mm = Tensor::aminmax(self.reshape({self.numel()}), {}, false);
        Tensor mn = std::get<0>(mm).to(DType::Float32).contiguous();
        Tensor mx = std::get<1>(mm).to(DType::Float32).contiguous();
        const cudaStream_t stream = getCurrentCUDAStream().stream();
        moving_average_minmax_kernel<<<1, 1, 0, stream>>>(
            1, mn.data_ptr<float>(), mx.data_ptr<float>(),
            running_min.data_ptr<float>(), running_max.data_ptr<float>(),
            static_cast<float>(averaging_const));
        checkCuda(cudaGetLastError(), "CUDA moving average kernel");
    }
    if (!fake_on) {
        Tensor mask = Tensor::full_like(self, 1, DType::Bool);
        return {self.clone(), std::move(mask)};
    }
    const double mn = running_min.item().toDouble();
    const double mx = running_max.item().toDouble();
    const QParams qp =
        choose_qparams_host(mn, mx, quant_min, quant_max, symmetric_quant);
    Tensor sc = Tensor::full({1}, Scalar(static_cast<float>(qp.scale)),
                             DType::Float32, self.device());
    Tensor zp = Tensor::full({1}, Scalar(qp.zero_point), DType::Int64,
                             self.device());
    scale.copy_(sc);
    zero_point.copy_(zp);
    Tensor enabled = Tensor::full({1}, Scalar(static_cast<int64_t>(1)),
                                  DType::Int64, self.device());
    return _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda(
        self, sc, zp, enabled, quant_min, quant_max);
}

Tensor fused_moving_avg_obs_fake_quant_cuda(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant) {
    if (self.numel() == 0) {
        return self.clone();
    }
    return std::get<0>(_fused_moving_avg_obs_fq_helper_cuda(
        self, observer_on, fake_quant_on, running_min, running_max, scale,
        zero_point, averaging_const, quant_min, quant_max, ch_axis,
        per_row_fake_quant, symmetric_quant));
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantKernels) {
    m.impl("quantize_per_tensor", quantize_per_tensor_cuda);
    m.impl("dequantize_per_tensor", dequantize_per_tensor_cuda);
    m.impl("quantize_per_channel", quantize_per_channel_cuda);
    m.impl("dequantize_per_channel", dequantize_per_channel_cuda);
    m.impl("quantized_linear", quantized_linear_cuda);
    m.impl("quantized_add", quantized_add_cuda);
    m.impl("quantized_sub", quantized_sub_cuda);
    m.impl("quantized_mul", quantized_mul_cuda);
    m.impl("quantized_div", quantized_div_cuda);
    m.impl("quantized_clamp", quantized_clamp_cuda);
    m.impl("quantized_max_pool2d", quantized_max_pool2d_cuda);
    m.impl("quantized_conv2d", quantized_conv2d_cuda);
    m.impl("quantize_per_tensor_quint8", quantize_per_tensor_quint8_cuda);
    m.impl("dequantize_per_tensor_quint8", dequantize_per_tensor_quint8_cuda);
    m.impl("quantize_per_tensor_qint32", quantize_per_tensor_qint32_cuda);
    m.impl("dequantize_per_tensor_qint32", dequantize_per_tensor_qint32_cuda);
    m.impl("fake_quantize_per_tensor_affine",
           fake_quantize_per_tensor_affine_cuda);
    m.impl("fake_quantize_per_tensor_affine.tensor_qparams",
           fake_quantize_per_tensor_affine_tensor_qparams_cuda);
    m.impl("fake_quantize_per_tensor_affine_cachemask",
           fake_quantize_per_tensor_affine_cachemask_cuda);
    m.impl("_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
           _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda);
    m.impl("fake_quantize_per_tensor_affine_cachemask_backward",
           fake_quantize_per_tensor_affine_cachemask_backward_cuda);
    m.impl("_fake_quantize_learnable_per_tensor_affine",
           _fake_quantize_learnable_per_tensor_affine_cuda);
    m.impl("_fake_quantize_learnable_per_tensor_affine_backward",
           _fake_quantize_learnable_per_tensor_affine_backward_cuda);
    m.impl("fake_quantize_per_channel_affine",
           fake_quantize_per_channel_affine_cuda);
    m.impl("fake_quantize_per_channel_affine_cachemask",
           fake_quantize_per_channel_affine_cachemask_cuda);
    m.impl("fake_quantize_per_channel_affine_cachemask_backward",
           fake_quantize_per_channel_affine_cachemask_backward_cuda);
    m.impl("_fake_quantize_learnable_per_channel_affine",
           _fake_quantize_learnable_per_channel_affine_cuda);
    m.impl("_fake_quantize_learnable_per_channel_affine_backward",
           _fake_quantize_learnable_per_channel_affine_backward_cuda);
    m.impl("quantize_per_tensor_dynamic", quantize_per_tensor_dynamic_cuda);
    m.impl("_choose_qparams_per_tensor", _choose_qparams_per_tensor_cuda);
    m.impl("fused_moving_avg_obs_fake_quant",
           fused_moving_avg_obs_fake_quant_cuda);
    m.impl("_fused_moving_avg_obs_fq_helper",
           _fused_moving_avg_obs_fq_helper_cuda);
    m.impl("is_quantized", is_quantized_cuda);
    m.impl("qscheme", qscheme_cuda);
    m.impl("q_scale", q_scale_cuda);
    m.impl("q_zero_point", q_zero_point_cuda);
    m.impl("q_per_channel_scales", q_per_channel_scales_cuda);
    m.impl("q_per_channel_zero_points", q_per_channel_zero_points_cuda);
    m.impl("q_per_channel_axis", q_per_channel_axis_cuda);
    m.impl("int_repr", int_repr_cuda);
    m.impl("dequantize.self", dequantize_self_cuda);
    m.impl("_make_per_tensor_quantized_tensor",
           _make_per_tensor_quantized_tensor_cuda);
    m.impl("_make_per_channel_quantized_tensor",
           _make_per_channel_quantized_tensor_cuda);
}

} // namespace cuda
} // namespace tensorplay
