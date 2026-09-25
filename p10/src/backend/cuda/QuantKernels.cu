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

std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max);

std::tuple<Tensor, Tensor>
_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    const Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max);

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

template <typename Storage>
__global__ void quantize_per_channel_kernel(
    int64_t numel,
    const float* input,
    Storage* output,
    int64_t stride_on_axis,
    int64_t channels,
    const float* scales,
    const int64_t* zero_points,
    int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    const int64_t rounded = static_cast<int64_t>(
        nearbyintf(input[i] / scales[c]) +
        static_cast<float>(zero_points[c]));
    const int64_t clamped = rounded < quant_min
        ? quant_min
        : (rounded > quant_max ? quant_max : rounded);
    output[i] = static_cast<Storage>(clamped);
}

template <typename Storage>
__global__ void quantize_per_channel_float_qparams_kernel(
    int64_t numel,
    const float* input,
    Storage* output,
    int64_t stride_on_axis,
    int64_t channels,
    const float* scales,
    const float* zero_points,
    int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    const int64_t rounded = static_cast<int64_t>(
        nearbyintf(input[i] * (1.0f / scales[c]) + zero_points[c]));
    const int64_t clamped = rounded < quant_min
        ? quant_min
        : (rounded > quant_max ? quant_max : rounded);
    output[i] = static_cast<Storage>(clamped);
}

template <typename Storage>
__global__ void dequantize_per_channel_kernel(
    int64_t numel,
    const Storage* input,
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

template <typename Storage>
__global__ void dequantize_per_channel_float_qparams_kernel(
    int64_t numel,
    const Storage* input,
    float* output,
    int64_t stride_on_axis,
    int64_t channels,
    const float* scales,
    const float* zero_points) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    output[i] = (static_cast<float>(input[i]) - zero_points[c]) * scales[c];
}

void check_qparams(double scale, int64_t zero_point, int64_t quant_min,
                   int64_t quant_max) {
    if (!std::isfinite(scale) || !(scale > 0.0)) {
        TP_THROW(ValueError, "quantize(): scale must be positive");
    }
    if (quant_min >= quant_max) {
        TP_THROW(ValueError, "quantize(): quant_min must be < quant_max");
    }
    if (zero_point < quant_min || zero_point > quant_max) {
        TP_THROW(ValueError, "quantize(): zero_point out of the quantized range");
    }
}

void check_storage_range(int64_t quant_min, int64_t quant_max,
                         int64_t storage_min, int64_t storage_max) {
    if (quant_min < storage_min || quant_max > storage_max) {
        TP_THROW(ValueError,
                 "quantize(): quantization range does not fit the output storage");
    }
}

// Shared host-side preparation: validate dtypes and land the operands on
// Float32/Int8 compute layouts.  Half/BFloat16 promote to Float32 first.
struct QuantInputs {
    Tensor input;      // Float32 compute buffer
    int64_t stride_on_axis;
};

QuantInputs prepare_quantize(const Tensor& self, int64_t axis = 0) {
    if (!isFloatingType(self.dtype())) {
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

Tensor empty_quantized_codes(const std::vector<int64_t>& size, DType dtype,
                             const Device& device,
                             std::optional<int64_t> memory_format) {
    Tensor codes(size, underlying_storage_type(dtype), device);
    const MemoryFormat format = static_cast<MemoryFormat>(
        memory_format.value_or(static_cast<int64_t>(MemoryFormat::Contiguous)));
    if (format == MemoryFormat::ChannelsLast ||
        format == MemoryFormat::ChannelsLast3d) {
        return codes.as_strided(size, get_channels_last_strides(size), 0);
    }
    return codes;
}

Tensor empty_affine_quantized_cuda(const std::vector<int64_t>& size,
                                   std::optional<DType> dtype,
                                   std::optional<int64_t> layout,
                                   std::optional<Device> device,
                                   std::optional<bool> pin_memory,
                                   double scale, int64_t zero_point,
                                   std::optional<int64_t> memory_format) {
    if (layout.has_value() && *layout != 5) {
        TP_THROW(NotImplementedError,
                 "_empty_affine_quantized is only implemented for strided layout tensors");
    }
    if (pin_memory.value_or(false)) {
        TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
    }
    if (!dtype.has_value() || *dtype == DType::Undefined) {
        TP_THROW(RuntimeError, "Must provide data type for Tensor creation functions.");
    }
    const DType dt = *dtype;
    if (!isQuantizedType(dt)) {
        TP_THROW(TypeError,
                 "_empty_affine_quantized(): dtype must be a quantized dtype, got ",
                 toString(dt));
    }
    const Device target = device.value_or(Device(DeviceType::CUDA));
    Tensor codes = empty_quantized_codes(size, dt, target, memory_format);
    return quantized::make_qtensor(
        codes, make_per_tensor_affine_quantizer(scale, zero_point, dt), dt);
}

Tensor empty_per_channel_affine_quantized_cuda(
    const std::vector<int64_t>& size, const Tensor& scales,
    const Tensor& zero_points, int64_t axis, std::optional<DType> dtype,
    std::optional<int64_t> layout, std::optional<Device> device,
    std::optional<bool> pin_memory, std::optional<int64_t> memory_format) {
    if (layout.has_value() && *layout != 5) {
        TP_THROW(NotImplementedError,
                 "_empty_per_channel_affine_quantized is only implemented for strided layout tensors");
    }
    if (pin_memory.value_or(false)) {
        TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
    }
    if (!dtype.has_value() || *dtype == DType::Undefined) {
        TP_THROW(RuntimeError, "Must provide data type for Tensor creation functions.");
    }
    const DType dt = *dtype;
    if (!isQuantizedType(dt)) {
        TP_THROW(TypeError,
                 "_empty_per_channel_affine_quantized(): dtype must be a quantized dtype, got ",
                 toString(dt));
    }
    const Device target = device.value_or(Device(DeviceType::CUDA));
    Tensor target_scales = scales.device() == target ? scales : scales.to(target);
    Tensor target_zero_points =
        zero_points.device() == target ? zero_points : zero_points.to(target);
    Tensor codes = empty_quantized_codes(size, dt, target, memory_format);
    return quantized::make_qtensor(
        codes,
        make_per_channel_affine_quantizer(target_scales, target_zero_points,
                                          axis, dt),
        dt);
}

Tensor empty_quantized_cuda(const std::vector<int64_t>& size,
                            const Tensor& qtensor,
                            std::optional<DType> dtype,
                            std::optional<int64_t> layout,
                            std::optional<Device> device,
                            std::optional<bool> pin_memory,
                            std::optional<int64_t> memory_format) {
    if (layout.has_value() && *layout != 5) {
        TP_THROW(NotImplementedError,
                 "empty_quantized is only implemented for strided layout tensors");
    }
    if (pin_memory.value_or(false)) {
        TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
    }
    quantized::require_quantized(qtensor, "empty_quantized");
    const DType dt = (dtype.has_value() && *dtype != DType::Undefined)
        ? *dtype : qtensor.dtype();
    if (dt != qtensor.dtype()) {
        TP_THROW(RuntimeError,
                 "empty_quantized(): dtype must match the source quantized tensor");
    }

    const Device target = device.value_or(qtensor.device());
    Tensor codes = empty_quantized_codes(size, dt, target, memory_format);

    const QuantizerPtr source_quantizer = quantized::quantizer_of(qtensor);
    QuantizerPtr output_quantizer;
    switch (source_quantizer->qscheme()) {
        case kPerTensorAffine:
            output_quantizer = make_per_tensor_affine_quantizer(
                source_quantizer->scale(), source_quantizer->zero_point(), dt);
            break;
        case kPerChannelAffine:
        case kPerChannelAffineFloatQParams: {
            Tensor scales = source_quantizer->scales();
            Tensor zero_points = source_quantizer->zero_points();
            if (scales.device() != target) scales = scales.to(target);
            if (zero_points.device() != target) zero_points = zero_points.to(target);
            output_quantizer = make_per_channel_affine_quantizer(
                scales, zero_points, source_quantizer->axis(), dt);
            break;
        }
        default:
            TP_THROW(ValueError,
                     "empty_quantized(): unsupported quantization scheme");
    }
    return quantized::make_qtensor(codes, std::move(output_quantizer), dt);
}

Tensor quantize_per_tensor_qint8_cuda(const Tensor& self, double scale,
                                      int64_t zero_point, int64_t quant_min,
                                      int64_t quant_max) {
    check_qparams(scale, zero_point, quant_min, quant_max);
    check_storage_range(quant_min, quant_max, -128, 127);
    QuantInputs prepared = prepare_quantize(self);
    Tensor out = Tensor::empty(self.shape(), DType::QInt8, self.device());
    const int64_t numel = prepared.input.numel();
    if (numel == 0) {
        out.impl()->set_quantizer(
            make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt8));
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
        make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt8));
    return out;
}

Tensor dequantize_per_tensor_qint8_cuda(const Tensor& self, double scale,
                                        int64_t zero_point) {
    if (self.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "dequantize(): expected a QInt8 tensor");
    }
    make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt8);
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

namespace {

std::pair<int64_t, int64_t> quantized_bounds_cuda(DType dtype) {
    switch (dtype) {
        case DType::QInt8:
            return {-128, 127};
        case DType::QUInt8:
            return {0, 255};
        case DType::QInt32:
            return {std::numeric_limits<int32_t>::min(),
                    std::numeric_limits<int32_t>::max()};
        default:
            TP_THROW(TypeError,
                     "per-channel quantization requires a quantized dtype");
    }
}

void check_per_channel_inputs_cuda(const Tensor& self, const Tensor& scales,
                                   const Tensor& zero_points, int64_t& axis,
                                   const char* op) {
    if (scales.dim() != 1 || zero_points.shape() != scales.shape()) {
        TP_THROW(ValueError, op,
                 ": scales/zero_points must be 1-D with equal sizes");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, op, ": axis out of range");
    }
    if (scales.size(0) != self.size(axis)) {
        TP_THROW(ValueError, op,
                 ": scales size must match the quantized dimension");
    }
    if (scales.device() != self.device() ||
        zero_points.device() != self.device()) {
        TP_THROW(RuntimeError, op,
                 ": scales and zero_points must be on the input device");
    }
}

template <typename Storage>
Tensor quantize_per_channel_cuda_impl(const Tensor& self, const Tensor& scales,
                                      const Tensor& zero_points, int64_t axis,
                                      DType dtype) {
    check_per_channel_inputs_cuda(self, scales, zero_points, axis, "quantize()");
    const auto bounds = quantized_bounds_cuda(dtype);
    QuantizerPtr quantizer = make_per_channel_affine_quantizer(
        scales, zero_points, axis, dtype);
    QuantInputs prepared = prepare_quantize(self, axis);
    Tensor scales_f32 = scales.to(DType::Float32).contiguous();
    Tensor out = Tensor::empty(self.shape(), dtype, self.device());
    out.impl()->set_quantizer(quantizer);
    const int64_t numel = prepared.input.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (isFloatingType(zero_points.dtype())) {
        Tensor zps_f32 = zero_points.to(DType::Float32).contiguous();
        quantize_per_channel_float_qparams_kernel<Storage>
            <<<blocks, threads, 0, stream>>>(
                numel, prepared.input.data_ptr<float>(), out.data_ptr<Storage>(),
                prepared.stride_on_axis, scales.size(0),
                scales_f32.data_ptr<float>(), zps_f32.data_ptr<float>(),
                bounds.first, bounds.second);
    } else {
        Tensor zps_i64 = zero_points.to(DType::Int64).contiguous();
        quantize_per_channel_kernel<Storage><<<blocks, threads, 0, stream>>>(
            numel, prepared.input.data_ptr<float>(), out.data_ptr<Storage>(),
            prepared.stride_on_axis, scales.size(0),
            scales_f32.data_ptr<float>(), zps_i64.data_ptr<int64_t>(),
            bounds.first, bounds.second);
    }
    checkCuda(cudaGetLastError(), "CUDA quantize_per_channel kernel");
    return out;
}

template <typename Storage>
Tensor dequantize_per_channel_cuda_impl(const Tensor& self,
                                        const Tensor& scales,
                                        const Tensor& zero_points, int64_t axis,
                                        DType dtype) {
    if (self.dtype() != dtype) {
        TP_THROW(TypeError, "dequantize(): quantized dtype does not match ",
                 toString(dtype));
    }
    check_per_channel_inputs_cuda(self, scales, zero_points, axis,
                                  "dequantize()");
    QuantizerPtr quantizer = make_per_channel_affine_quantizer(
        scales, zero_points, axis, dtype);
    Tensor input = self.contiguous();
    Tensor scales_f32 = scales.to(DType::Float32).contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());

    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < input.dim(); ++d) stride_on_axis *= input.size(d);

    const int64_t numel = input.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (isFloatingType(zero_points.dtype())) {
        Tensor zps_f32 = zero_points.to(DType::Float32).contiguous();
        dequantize_per_channel_float_qparams_kernel<Storage>
            <<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<Storage>(), out.data_ptr<float>(),
                stride_on_axis, scales.size(0), scales_f32.data_ptr<float>(),
                zps_f32.data_ptr<float>());
    } else {
        Tensor zps_i64 = zero_points.to(DType::Int64).contiguous();
        dequantize_per_channel_kernel<Storage><<<blocks, threads, 0, stream>>>(
            numel, input.data_ptr<Storage>(), out.data_ptr<float>(),
            stride_on_axis, scales.size(0), scales_f32.data_ptr<float>(),
            zps_i64.data_ptr<int64_t>());
    }
    checkCuda(cudaGetLastError(), "CUDA dequantize_per_channel kernel");
    (void)quantizer;
    return out;
}

} // namespace

Tensor quantize_per_channel_cuda(const Tensor& self, const Tensor& scales,
                                 const Tensor& zero_points, int64_t axis,
                                 DType dtype) {
    return quantize_per_channel_dtype_cuda(
        self, scales, zero_points, axis, dtype);
}

Tensor quantize_per_channel_dtype_cuda(const Tensor& self,
                                       const Tensor& scales,
                                       const Tensor& zero_points, int64_t axis,
                                       DType dtype) {
    if (self.dtype() != DType::Float32) {
        TP_THROW(TypeError, "quantize(): expected a Float32 tensor, got ",
                 toString(self.dtype()));
    }
    switch (dtype) {
        case DType::QInt8:
            return quantize_per_channel_cuda_impl<int8_t>(
                self, scales, zero_points, axis, dtype);
        case DType::QUInt8:
            return quantize_per_channel_cuda_impl<uint8_t>(
                self, scales, zero_points, axis, dtype);
        case DType::QInt32:
            return quantize_per_channel_cuda_impl<int32_t>(
                self, scales, zero_points, axis, dtype);
        default:
            TP_THROW(TypeError, "quantize(): unsupported quantized dtype ",
                     toString(dtype));
    }
}

Tensor dequantize_per_channel_dtype_cuda(const Tensor& self,
                                         const Tensor& scales,
                                         const Tensor& zero_points,
                                         int64_t axis, DType dtype) {
    switch (dtype) {
        case DType::QInt8:
            return dequantize_per_channel_cuda_impl<int8_t>(
                self, scales, zero_points, axis, dtype);
        case DType::QUInt8:
            return dequantize_per_channel_cuda_impl<uint8_t>(
                self, scales, zero_points, axis, dtype);
        case DType::QInt32:
            return dequantize_per_channel_cuda_impl<int32_t>(
                self, scales, zero_points, axis, dtype);
        default:
            TP_THROW(TypeError, "dequantize(): unsupported quantized dtype ",
                     toString(dtype));
    }
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
    check_qparams(scale, zero_point, quant_min, quant_max);
    check_storage_range(quant_min, quant_max, 0, 255);
    const Tensor ic = prepare_quantize(self).input;
    Tensor codes = Tensor::empty(self.shape(), DType::Int64, self.device());
    Tensor out = Tensor::empty(self.shape(), DType::QUInt8, self.device());
    const int64_t numel = self.numel();
    if (numel == 0) {
        out.impl()->set_quantizer(
            make_per_tensor_affine_quantizer(scale, zero_point, DType::QUInt8));
        return out;
    }
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
        make_per_tensor_affine_quantizer(scale, zero_point, DType::QUInt8));
    return out;
}

Tensor dequantize_per_tensor_quint8_cuda(const Tensor& self, double scale,
                                          int64_t zero_point) {
    if (self.dtype() != DType::QUInt8) {
        TP_THROW(TypeError, "dequantize(): expected a QUInt8 tensor");
    }
    make_per_tensor_affine_quantizer(scale, zero_point, DType::QUInt8);
    const Tensor ic = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());
    const int64_t numel = self.numel();
    if (numel == 0) return out;
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
    check_qparams(scale, zero_point, std::numeric_limits<int32_t>::min(),
                  std::numeric_limits<int32_t>::max());
    const Tensor ic = prepare_quantize(self).input;
    Tensor codes = Tensor::empty(self.shape(), DType::Int64, self.device());
    Tensor out = Tensor::empty(self.shape(), DType::QInt32, self.device());
    const int64_t numel = self.numel();
    if (numel == 0) {
        out.impl()->set_quantizer(
            make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt32));
        return out;
    }
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
        make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt32));
    return out;
}

Tensor dequantize_per_tensor_qint32_cuda(const Tensor& self, double scale,
                                          int64_t zero_point) {
    if (self.dtype() != DType::QInt32) {
        TP_THROW(TypeError, "dequantize(): expected a QInt32 tensor");
    }
    make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt32);
    const Tensor ic = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());
    const int64_t numel = self.numel();
    if (numel == 0) return out;
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    dequantize_int32_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, ic.data_ptr<int32_t>(), out.data_ptr<float>(),
        static_cast<float>(scale), static_cast<float>(zero_point));
    checkCuda(cudaGetLastError(), "CUDA dequantize_int32 kernel");
    return out;
}

Tensor quantize_per_tensor_dtype_cuda(const Tensor& self, double scale,
                                      int64_t zero_point, DType dtype) {
    if (self.dtype() != DType::Float32) {
        TP_THROW(TypeError, "quantize(): expected a Float32 tensor, got ",
                 toString(self.dtype()));
    }
    switch (dtype) {
        case DType::QInt8:
            return quantize_per_tensor_qint8_cuda(
                self, scale, zero_point, -128, 127);
        case DType::QUInt8:
            return quantize_per_tensor_quint8_cuda(
                self, scale, zero_point, 0, 255);
        case DType::QInt32:
            return quantize_per_tensor_qint32_cuda(self, scale, zero_point);
        default:
            TP_THROW(TypeError, "quantize(): unsupported quantized dtype ",
                     toString(dtype));
    }
}

Tensor quantize_per_tensor_cuda(const Tensor& self, double scale,
                                int64_t zero_point, DType dtype) {
    return quantize_per_tensor_dtype_cuda(self, scale, zero_point, dtype);
}

Tensor dequantize_per_tensor_dtype_cuda(const Tensor& self, double scale,
                                        int64_t zero_point, DType dtype) {
    switch (dtype) {
        case DType::QInt8:
            return dequantize_per_tensor_qint8_cuda(self, scale, zero_point);
        case DType::QUInt8:
            return dequantize_per_tensor_quint8_cuda(self, scale, zero_point);
        case DType::QInt32:
            return dequantize_per_tensor_qint32_cuda(self, scale, zero_point);
        default:
            TP_THROW(TypeError, "dequantize(): unsupported quantized dtype ",
                     toString(dtype));
    }
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

void check_real_dtype(const Tensor& self, const char* op) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError,
                 std::string(op) + ": expected a floating point tensor, got " +
                     toString(self.dtype()));
    }
}

} // namespace

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
        make_per_tensor_affine_quantizer(qp.scale, qp.zero_point, dtype));
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
        return self.to(DType::Float32);
    }
    return quantized::quantizer_of(self)->dequantize(self);
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
        self.clone(), make_per_tensor_affine_quantizer(scale, zero_point, qdt),
        qdt);
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
        make_per_channel_affine_quantizer(scale, zero_point, axis, qdt), qdt);
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
    m.impl("_empty_affine_quantized", empty_affine_quantized_cuda);
    m.impl("_empty_per_channel_affine_quantized",
           empty_per_channel_affine_quantized_cuda);
    m.impl("empty_quantized", empty_quantized_cuda);
    m.impl("quantize_per_tensor", quantize_per_tensor_cuda);
    m.impl("quantize_per_channel", quantize_per_channel_cuda);
    m.impl("quantized_max_pool2d", quantized_max_pool2d_cuda);
    m.impl("quantized_conv2d", quantized_conv2d_cuda);
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
