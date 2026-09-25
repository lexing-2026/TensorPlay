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

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantKernels) {
    m.impl("_empty_affine_quantized", empty_affine_quantized_cuda);
    m.impl("_empty_per_channel_affine_quantized",
           empty_per_channel_affine_quantized_cuda);
    m.impl("empty_quantized", empty_quantized_cuda);
    m.impl("quantize_per_tensor", quantize_per_tensor_cuda);
    m.impl("quantize_per_channel", quantize_per_channel_cuda);
    m.impl("quantized_max_pool2d", quantized_max_pool2d_cuda);
    m.impl("quantized_conv2d", quantized_conv2d_cuda);

}

} // namespace cuda
} // namespace tensorplay
