#include "QuantKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"
#include "SizesAndStrides.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

template<int Bits,bool Quantize,bool Channel>
__global__ void subbyte_affine_kernel(const float* real,const uint8_t* packed,float* output,uint8_t* codes,
        int64_t size,float scale,float zp,const float* scales,const float* zeros,int64_t stride,int64_t channels) {
    constexpr int pack=8/Bits,mask=(1<<Bits)-1;
    const int64_t bytes=(size+pack-1)/pack;
    for (int64_t byte=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
            byte<bytes;byte+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        uint8_t value=0;
        for (int lane=0;lane<pack && byte*pack+lane<size;++lane) {
            const int64_t i=byte*pack+lane;
            float s=scale,z=zp;
            if constexpr (Channel) { const int64_t c=(i/stride)%channels; s=scales[c]; z=zeros[c]; }
            if constexpr (Quantize) {
                const float inverse=s==0.f ? 1.f : 1.f/s;
                const float position=__fmul_rn(real[i],inverse);
                const float q=Channel ? nearbyintf(__fadd_rn(position,z)) : __fadd_rn(nearbyintf(position),z);
                uint8_t code=0;
                if constexpr (Channel) {
                    const int64_t wide=q>=-0x1p63f && q<0x1p63f ? static_cast<int64_t>(q) : INT64_MIN;
                    const int32_t narrow=static_cast<int32_t>(static_cast<uint32_t>(wide));
                    code=static_cast<uint8_t>(narrow<0 ? 0 : (narrow>mask ? mask : narrow));
                } else if (q>=-0x1p63f && q<0x1p63f) code=static_cast<uint8_t>(fminf(mask,fmaxf(0.f,q)));
                value|=code<<(lane*Bits);
            } else output[i]=__fmul_rn(__fadd_rn(static_cast<float>((packed[byte]>>(lane*Bits))&mask),-z),s);
        }
        if constexpr (Quantize) codes[byte]=value;
    }
}

template<bool Quantize,bool Channel>
Tensor subbyte_affine_cuda(const Tensor& self,DType dtype,double scale,int64_t zp,
        const Tensor& scales=Tensor(),const Tensor& zeros=Tensor(),int64_t axis=0) {
    const bool channels_last=!Channel && ((self.dim()==4 && self.is_contiguous(MemoryFormat::ChannelsLast))
        || (self.dim()==5 && self.is_contiguous(MemoryFormat::ChannelsLast3d)));
    const Tensor input=channels_last ? self : self.contiguous();
    Tensor out=Tensor::empty(self.shape(),Quantize ? dtype : DType::Float32,self.device());
    if (channels_last) out=out.as_strided(self.shape(),get_channels_last_strides(self.shape()),0);
    Tensor sc,z;
    int64_t stride=1,channels=0;
    if constexpr (Channel) {
        sc=scales.to(DType::Float32).contiguous(); z=zeros.to(DType::Float32).contiguous();
        channels=sc.numel();
        for (int64_t d=axis+1;d<self.dim();++d) stride*=self.size(d);
    }
    if (self.numel()) {
        const int bits=dtype==DType::QUInt4x2 ? 4 : (dtype==DType::QUInt2x4 ? 2 : 1),pack=8/bits;
        const int blocks=static_cast<int>(std::min<int64_t>(((self.numel()+pack-1)/pack+255)/256,65535));
        auto launch=[&]<int Bits>() {
            subbyte_affine_kernel<Bits,Quantize,Channel><<<blocks,256,0,getCurrentCUDAStream().stream()>>>(
                Quantize ? input.data_ptr<float>() : nullptr,Quantize ? nullptr : static_cast<const uint8_t*>(input.data_ptr()),
                Quantize ? nullptr : out.data_ptr<float>(),Quantize ? static_cast<uint8_t*>(out.data_ptr()) : nullptr,
                self.numel(),static_cast<float>(scale),static_cast<float>(zp),Channel ? sc.data_ptr<float>() : nullptr,
                Channel ? z.data_ptr<float>() : nullptr,stride,channels);
        };
        if (bits==4) launch.template operator()<4>();
        else if (bits==2) launch.template operator()<2>();
        else launch.template operator()<1>();
        checkCuda(cudaGetLastError(),"Sub-byte affine kernel");
    }
    if constexpr (Quantize) {
        if constexpr (Channel) out.impl()->set_quantizer(make_per_channel_affine_quantizer(sc,z,axis,dtype));
        else out.impl()->set_quantizer(make_per_tensor_affine_quantizer(scale,zp,dtype));
    }
    return out;
}

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
    Tensor allocation(size,dtype,device);
    Tensor codes(allocation.impl()->storage(),size,allocation.strides(),underlying_storage_type(dtype),0);
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
        case DType::QUInt2x4:
            return {0, 3};
        case DType::QUInt4x2:
            return {0, 15};
        case DType::QUInt1x8:
            return {0, 1};
        default:
            TP_THROW(TypeError,
                     "quantize_affine(): unsupported quantized dtype");
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
        case DType::QUInt2x4:
        case DType::QUInt4x2:
        case DType::QUInt1x8:
            check_per_channel_inputs_cuda(self,scales,zero_points,axis,"quantize()");
            TP_CHECK(isFloatingType(zero_points.dtype()),"Sub-byte per-channel quantization requires floating zero points");
            make_per_channel_affine_quantizer(scales,zero_points,axis,dtype);
            return subbyte_affine_cuda<true,true>(self,dtype,0.,0,scales,zero_points,axis);
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
        case DType::QUInt2x4:
        case DType::QUInt4x2:
        case DType::QUInt1x8:
            check_per_channel_inputs_cuda(self,scales,zero_points,axis,"dequantize()");
            TP_CHECK(isFloatingType(zero_points.dtype()),"Sub-byte per-channel quantization requires floating zero points");
            return subbyte_affine_cuda<false,true>(self,dtype,0.,0,scales,zero_points,axis);
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
        case DType::QUInt2x4:
        case DType::QUInt4x2:
        case DType::QUInt1x8:
            make_per_tensor_affine_quantizer(scale,zero_point,dtype);
            return subbyte_affine_cuda<true,false>(self,dtype,scale,zero_point);
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
        case DType::QUInt2x4:
        case DType::QUInt4x2:
        case DType::QUInt1x8:
            return subbyte_affine_cuda<false,false>(self,dtype,scale,zero_point);
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

// ---- Block-wise affine quantization (quantize_affine/dequantize_affine) ----

__device__ __forceinline__ int64_t cuda_block_of(
    int64_t linear, int64_t dim, const int64_t* sizes, const int64_t* blocks,
    const int64_t* block_stride, const int64_t* size_stride) {
    int64_t b = 0;
    for (int64_t d = 0; d < dim; ++d) {
        const int64_t coord = (linear / size_stride[d]) % sizes[d];
        b += (coord / blocks[d]) * block_stride[d];
    }
    return b;
}

template <typename Storage, bool HasZero>
__global__ void quantize_affine_group_kernel(
    const float* input, Storage* output, int64_t numel, int64_t dim,
    const int64_t* sizes, const int64_t* blocks, const int64_t* block_stride,
    const int64_t* size_stride, const float* scales, const float* zeros,
    int64_t quant_min, int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t b =
        cuda_block_of(i, dim, sizes, blocks, block_stride, size_stride);
    const float inv = scales[b] == 0.f ? 1.f : 1.f / scales[b];
    const double position = HasZero
        ? nearbyint(input[i] * inv) + zeros[b]
        : nearbyint(input[i] * inv);
    const int64_t wide = position >= -0x1p63 && position < 0x1p63
        ? static_cast<int64_t>(position)
        : INT64_MIN;
    output[i] = static_cast<Storage>(
        min(quant_max, max(quant_min, wide)));
}

template <typename Storage, bool HasZero>
__global__ void dequantize_affine_group_kernel(
    const Storage* input, float* output, int64_t numel, int64_t dim,
    const int64_t* sizes, const int64_t* blocks, const int64_t* block_stride,
    const int64_t* size_stride, const float* scales, const float* zeros) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t b =
        cuda_block_of(i, dim, sizes, blocks, block_stride, size_stride);
    output[i] = (static_cast<float>(input[i]) -
                 (HasZero ? zeros[b] : 0.f)) * scales[b];
}

template <int Bits, bool HasZero>
__global__ void quantize_affine_subbyte_group_kernel(
    const float* input, uint8_t* output, int64_t numel, int64_t dim,
    const int64_t* sizes, const int64_t* blocks, const int64_t* block_stride,
    const int64_t* size_stride, const float* scales, const float* zeros,
    int64_t quant_min, int64_t quant_max) {
    constexpr int pack = 8 / Bits;
    const int64_t bytes = (numel + pack - 1) / pack;
    for (int64_t byte = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         byte < bytes; byte += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        uint8_t value = 0;
        for (int lane = 0; lane < pack && byte * pack + lane < numel; ++lane) {
            const int64_t i = byte * pack + lane;
            const int64_t b =
                cuda_block_of(i, dim, sizes, blocks, block_stride, size_stride);
            const float inv = scales[b] == 0.f ? 1.f : 1.f / scales[b];
            const float position = HasZero
                ? __fadd_rn(nearbyintf(input[i] * inv), zeros[b])
                : nearbyintf(input[i] * inv);
            const int64_t wide = position >= -0x1p63f && position < 0x1p63f
                ? static_cast<int64_t>(position)
                : INT64_MIN;
            const int32_t q = static_cast<int32_t>(static_cast<uint32_t>(wide));
            value |= static_cast<uint8_t>(
                min(static_cast<int32_t>(quant_max),
                    max(static_cast<int32_t>(quant_min), q)))
                << (lane * Bits);
        }
        output[byte] = value;
    }
}

template <int Bits, bool HasZero>
__global__ void dequantize_affine_subbyte_group_kernel(
    const uint8_t* input, float* output, int64_t numel, int64_t dim,
    const int64_t* sizes, const int64_t* blocks, const int64_t* block_stride,
    const int64_t* size_stride, const float* scales, const float* zeros) {
    constexpr int pack = 8 / Bits;
    constexpr int mask = (1 << Bits) - 1;
    const int64_t bytes = (numel + pack - 1) / pack;
    for (int64_t byte = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         byte < bytes; byte += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        for (int lane = 0; lane < pack && byte * pack + lane < numel; ++lane) {
            const int64_t i = byte * pack + lane;
            const int64_t b =
                cuda_block_of(i, dim, sizes, blocks, block_stride, size_stride);
            output[i] = (static_cast<float>((input[byte] >> (lane * Bits)) & mask) -
                         (HasZero ? zeros[b] : 0.f)) * scales[b];
        }
    }
}

struct AffineGroupParamsCuda {
    int64_t dim;
    std::vector<int64_t> sizes;
    std::vector<int64_t> blocks;
    std::vector<int64_t> blocks_per_dim;
    std::vector<int64_t> block_stride;
    std::vector<int64_t> size_stride;
    int64_t num_blocks;
};

AffineGroupParamsCuda make_affine_group_params_cuda(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& block_size) {
    AffineGroupParamsCuda p;
    p.dim = static_cast<int64_t>(sizes.size());
    p.sizes = sizes;
    p.blocks.resize(p.dim);
    p.blocks_per_dim.resize(p.dim);
    p.block_stride.resize(p.dim);
    p.size_stride.resize(p.dim);
    int64_t stride = 1;
    for (int64_t d = p.dim - 1; d >= 0; --d) {
        p.size_stride[d] = stride;
        stride *= sizes[d];
    }
    int64_t blocks_after = 1;
    p.num_blocks = 1;
    for (int64_t d = p.dim - 1; d >= 0; --d) {
        const int64_t b = std::min<int64_t>(block_size[d], sizes[d]);
        p.blocks[d] = b > 0 ? b : 1;
        p.blocks_per_dim[d] = (sizes[d] + p.blocks[d] - 1) / p.blocks[d];
        p.block_stride[d] = blocks_after;
        blocks_after *= p.blocks_per_dim[d];
        p.num_blocks *= p.blocks_per_dim[d];
    }
    return p;
}

template <bool Quantize>
Tensor quantize_affine_cuda_impl(const Tensor& self,
                                 const std::vector<int64_t>& block_size,
                                 const Tensor& scales,
                                 const std::optional<Tensor>& zero_points,
                                 DType dtype, int64_t quant_min,
                                 int64_t quant_max) {
    const int64_t dim = self.dim();
    if (static_cast<int64_t>(block_size.size()) != dim) {
        TP_THROW(ValueError,
                 "quantize_affine(): block_size rank must match the input rank");
    }
    for (int64_t d = 0; d < dim; ++d) {
        if (block_size[d] <= 0) {
            TP_THROW(ValueError,
                     "quantize_affine(): block_size entries must be positive");
        }
    }
    const Tensor input = self.contiguous();
    const AffineGroupParamsCuda p =
        make_affine_group_params_cuda(input.shape(), block_size);
    if (scales.numel() != p.num_blocks && scales.numel() != 1) {
        TP_THROW(ValueError,
                 "quantize_affine(): scales must have one entry per block");
    }
    if (zero_points.has_value() && zero_points->numel() > 0 &&
        zero_points->numel() != p.num_blocks && zero_points->numel() != 1) {
        TP_THROW(ValueError,
                 "quantize_affine(): zero_points must have one entry per "
                 "block, or be a scalar");
    }
    if (scales.device() != self.device() ||
        (zero_points.has_value() && zero_points->numel() > 0 &&
         zero_points->device() != self.device())) {
        TP_THROW(RuntimeError,
                 "quantize_affine(): scales and zero_points must be on the "
                 "input device");
    }
    const auto bounds = quantized_bounds_cuda(dtype);
    const int64_t qmin = quant_min < quant_max ? quant_min : bounds.first;
    const int64_t qmax = quant_max > quant_min ? quant_max : bounds.second;
    if (qmin < bounds.first || qmax > bounds.second) {
        TP_THROW(ValueError,
                 "quantize_affine(): quantization range does not fit the "
                 "output storage");
    }

    Tensor out = Tensor::empty(input.shape(), Quantize ? dtype : DType::Float32,
                               self.device());
    if (input.numel() == 0) return out;

    // Scales/zero points may live on the device; pull them to the host to
    // build the per-block buffers, then upload the buffers once for the
    // kernels (which need device-resident scale/zero arrays).
    const Tensor sc = scales.to(Device(DeviceType::CPU), DType::Float32)
                          .contiguous();
    std::vector<float> scale_buf(p.num_blocks);
    if (sc.numel() == 1) {
        std::fill(scale_buf.begin(), scale_buf.end(), *sc.data_ptr<float>());
    } else {
        std::memcpy(scale_buf.data(), sc.data_ptr<float>(),
                     sizeof(float) * p.num_blocks);
    }

    std::vector<float> zero_buf;
    const float* zeros = nullptr;
    bool has_zero = false;
    if (zero_points.has_value() && zero_points->numel() > 0) {
        has_zero = true;
        const Tensor z = zero_points->to(Device(DeviceType::CPU), DType::Float32)
                             .contiguous();
        zero_buf.resize(p.num_blocks);
        if (z.numel() == 1) {
            std::fill(zero_buf.begin(), zero_buf.end(), *z.data_ptr<float>());
        } else {
            std::memcpy(zero_buf.data(), z.data_ptr<float>(),
                         sizeof(float) * p.num_blocks);
        }
        zeros = zero_buf.data();
    }

    // Upload the block-grid metadata once per call; the grids are tiny
    // (one entry per input dimension).  A scalar input has rank 0: the
    // kernels never dereference the metadata, but the device buffers still
    // need a valid allocation.
    const int64_t rank = p.dim;
    const int64_t rank_alloc = std::max<int64_t>(rank, 1);
    int64_t* d_sizes = nullptr;
    int64_t* d_blocks = nullptr;
    int64_t* d_block_stride = nullptr;
    int64_t* d_size_stride = nullptr;
    checkCuda(cudaMalloc(&d_sizes, rank_alloc * sizeof(int64_t)),
              "quantize_affine cudaMalloc sizes");
    checkCuda(cudaMalloc(&d_blocks, rank_alloc * sizeof(int64_t)),
              "quantize_affine cudaMalloc blocks");
    checkCuda(cudaMalloc(&d_block_stride, rank_alloc * sizeof(int64_t)),
              "quantize_affine cudaMalloc block_stride");
    checkCuda(cudaMalloc(&d_size_stride, rank_alloc * sizeof(int64_t)),
              "quantize_affine cudaMalloc size_stride");
    if (rank > 0) {
        checkCuda(cudaMemcpy(d_sizes, p.sizes.data(), rank * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "quantize_affine cudaMemcpy sizes");
        checkCuda(cudaMemcpy(d_blocks, p.blocks.data(), rank * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "quantize_affine cudaMemcpy blocks");
        checkCuda(cudaMemcpy(d_block_stride, p.block_stride.data(),
                             rank * sizeof(int64_t), cudaMemcpyHostToDevice),
                  "quantize_affine cudaMemcpy block_stride");
        checkCuda(cudaMemcpy(d_size_stride, p.size_stride.data(),
                             rank * sizeof(int64_t), cudaMemcpyHostToDevice),
                  "quantize_affine cudaMemcpy size_stride");
    }

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int64_t numel = input.numel();
    const int bits = dtype == DType::QUInt4x2 ? 4
                    : dtype == DType::QUInt2x4 ? 2
                    : dtype == DType::QUInt1x8 ? 1 : 0;
    const bool subbyte = bits != 0;
    const int threads = 128;
    const int64_t elems_per_thread = subbyte ? (8 / bits) : 1;
    const int64_t work = (numel + elems_per_thread - 1) / elems_per_thread;
    const int blocks = static_cast<int>(
        std::min<int64_t>((work + threads - 1) / threads, 65535));

    const float* in = Quantize ? input.data_ptr<float>() : nullptr;
    float* d_scales = nullptr;
    float* d_zeros = nullptr;
    const int64_t num_blocks_alloc = std::max<int64_t>(p.num_blocks, 1);
    checkCuda(cudaMalloc(&d_scales, num_blocks_alloc * sizeof(float)),
              "quantize_affine cudaMalloc scales");
    checkCuda(cudaMemcpy(d_scales, scale_buf.data(),
                         p.num_blocks * sizeof(float), cudaMemcpyHostToDevice),
              "quantize_affine cudaMemcpy scales");
    if (has_zero) {
        checkCuda(cudaMalloc(&d_zeros, num_blocks_alloc * sizeof(float)),
                  "quantize_affine cudaMalloc zeros");
        checkCuda(cudaMemcpy(d_zeros, zero_buf.data(),
                            p.num_blocks * sizeof(float),
                            cudaMemcpyHostToDevice),
                  "quantize_affine cudaMemcpy zeros");
    }

    if (subbyte) {
        if (Quantize) {
            auto* outp = static_cast<uint8_t*>(out.data_ptr());
            if (bits == 4) {
                if (has_zero)
                    quantize_affine_subbyte_group_kernel<4, true>
                        <<<blocks, threads, 0, stream>>>(
                            in, outp, numel, rank, d_sizes, d_blocks,
                            d_block_stride, d_size_stride, d_scales, d_zeros, qmin, qmax);
                else
                    quantize_affine_subbyte_group_kernel<4, false>
                        <<<blocks, threads, 0, stream>>>(
                            in, outp, numel, rank, d_sizes, d_blocks,
                            d_block_stride, d_size_stride, d_scales, d_zeros, qmin, qmax);
            } else if (bits == 2) {
                if (has_zero)
                    quantize_affine_subbyte_group_kernel<2, true>
                        <<<blocks, threads, 0, stream>>>(
                            in, outp, numel, rank, d_sizes, d_blocks,
                            d_block_stride, d_size_stride, d_scales, d_zeros, qmin, qmax);
                else
                    quantize_affine_subbyte_group_kernel<2, false>
                        <<<blocks, threads, 0, stream>>>(
                            in, outp, numel, rank, d_sizes, d_blocks,
                            d_block_stride, d_size_stride, d_scales, d_zeros, qmin, qmax);
            } else if (has_zero)
                quantize_affine_subbyte_group_kernel<1, true>
                    <<<blocks, threads, 0, stream>>>(
                        in, outp, numel, rank, d_sizes, d_blocks,
                        d_block_stride, d_size_stride, d_scales, d_zeros, qmin, qmax);
            else
                quantize_affine_subbyte_group_kernel<1, false>
                    <<<blocks, threads, 0, stream>>>(
                        in, outp, numel, rank, d_sizes, d_blocks,
                        d_block_stride, d_size_stride, d_scales, d_zeros, qmin, qmax);
        } else {
            auto* outp = out.data_ptr<float>();
            if (bits == 4) {
                if (has_zero)
                    dequantize_affine_subbyte_group_kernel<4, true>
                        <<<blocks, threads, 0, stream>>>(
                            static_cast<const uint8_t*>(input.data_ptr()),
                            outp, numel, rank, d_sizes, d_blocks,
                            d_block_stride, d_size_stride, d_scales, d_zeros);
                else
                    dequantize_affine_subbyte_group_kernel<4, false>
                        <<<blocks, threads, 0, stream>>>(
                            static_cast<const uint8_t*>(input.data_ptr()),
                            outp, numel, rank, d_sizes, d_blocks,
                            d_block_stride, d_size_stride, d_scales, d_zeros);
            } else if (bits == 2) {
                if (has_zero)
                    dequantize_affine_subbyte_group_kernel<2, true>
                        <<<blocks, threads, 0, stream>>>(
                            static_cast<const uint8_t*>(input.data_ptr()), outp,
                            numel, rank, d_sizes, d_blocks, d_block_stride,
                            d_size_stride, d_scales, d_zeros);
                else
                    dequantize_affine_subbyte_group_kernel<2, false>
                        <<<blocks, threads, 0, stream>>>(
                            static_cast<const uint8_t*>(input.data_ptr()), outp,
                            numel, rank, d_sizes, d_blocks, d_block_stride,
                            d_size_stride, d_scales, d_zeros);
            } else if (has_zero)
                dequantize_affine_subbyte_group_kernel<1, true>
                    <<<blocks, threads, 0, stream>>>(
                        static_cast<const uint8_t*>(input.data_ptr()), outp,
                        numel, rank, d_sizes, d_blocks, d_block_stride,
                        d_size_stride, d_scales, d_zeros);
            else
                dequantize_affine_subbyte_group_kernel<1, false>
                    <<<blocks, threads, 0, stream>>>(
                        static_cast<const uint8_t*>(input.data_ptr()), outp,
                        numel, rank, d_sizes, d_blocks, d_block_stride,
                        d_size_stride, d_scales, d_zeros);
        }
    } else {
        if (Quantize) {
            switch (dtype) {
                case DType::QInt8:
                    if (has_zero)
                        quantize_affine_group_kernel<int8_t, true>
                            <<<blocks, threads, 0, stream>>>(
                                in, out.data_ptr<int8_t>(), numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros, qmin, qmax);
                    else
                        quantize_affine_group_kernel<int8_t, false>
                            <<<blocks, threads, 0, stream>>>(
                                in, out.data_ptr<int8_t>(), numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros, qmin, qmax);
                    break;
                case DType::QUInt8:
                    if (has_zero)
                        quantize_affine_group_kernel<uint8_t, true>
                            <<<blocks, threads, 0, stream>>>(
                                in, out.data_ptr<uint8_t>(), numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros, qmin, qmax);
                    else
                        quantize_affine_group_kernel<uint8_t, false>
                            <<<blocks, threads, 0, stream>>>(
                                in, out.data_ptr<uint8_t>(), numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros, qmin, qmax);
                    break;
                case DType::QInt32:
                    if (has_zero)
                        quantize_affine_group_kernel<int32_t, true>
                            <<<blocks, threads, 0, stream>>>(
                                in, out.data_ptr<int32_t>(), numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros, qmin, qmax);
                    else
                        quantize_affine_group_kernel<int32_t, false>
                            <<<blocks, threads, 0, stream>>>(
                                in, out.data_ptr<int32_t>(), numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros, qmin, qmax);
                    break;
                default:
                    break;
            }
        } else {
            auto* outp = out.data_ptr<float>();
            switch (dtype) {
                case DType::QInt8:
                    if (has_zero)
                        dequantize_affine_group_kernel<int8_t, true>
                            <<<blocks, threads, 0, stream>>>(
                                input.data_ptr<int8_t>(), outp, numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros);
                    else
                        dequantize_affine_group_kernel<int8_t, false>
                            <<<blocks, threads, 0, stream>>>(
                                input.data_ptr<int8_t>(), outp, numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros);
                    break;
                case DType::QUInt8:
                    if (has_zero)
                        dequantize_affine_group_kernel<uint8_t, true>
                            <<<blocks, threads, 0, stream>>>(
                                input.data_ptr<uint8_t>(), outp, numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros);
                    else
                        dequantize_affine_group_kernel<uint8_t, false>
                            <<<blocks, threads, 0, stream>>>(
                                input.data_ptr<uint8_t>(), outp, numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros);
                    break;
                case DType::QInt32:
                    if (has_zero)
                        dequantize_affine_group_kernel<int32_t, true>
                            <<<blocks, threads, 0, stream>>>(
                                input.data_ptr<int32_t>(), outp, numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros);
                    else
                        dequantize_affine_group_kernel<int32_t, false>
                            <<<blocks, threads, 0, stream>>>(
                                input.data_ptr<int32_t>(), outp, numel, rank,
                                d_sizes, d_blocks, d_block_stride,
                                d_size_stride, d_scales, d_zeros);
                    break;
                default:
                    break;
            }
        }
    }
    checkCuda(cudaGetLastError(), "quantize_affine kernel");
    checkCuda(cudaFree(d_sizes), "quantize_affine cudaFree sizes");
    checkCuda(cudaFree(d_blocks), "quantize_affine cudaFree blocks");
    checkCuda(cudaFree(d_block_stride), "quantize_affine cudaFree block_stride");
    checkCuda(cudaFree(d_size_stride), "quantize_affine cudaFree size_stride");
    checkCuda(cudaFree(d_scales), "quantize_affine cudaFree scales");
    if (has_zero) {
        checkCuda(cudaFree(d_zeros), "quantize_affine cudaFree zeros");
    }
    return out;
}

Tensor quantize_affine_cuda(const Tensor& self,
                            const std::vector<int64_t>& block_size,
                            const Tensor& scales,
                            const std::optional<Tensor>& zero_points,
                            DType dtype, int64_t quant_min, int64_t quant_max) {
    if (!isQuantizedType(dtype)) {
        TP_THROW(TypeError, "quantize_affine(): expected a quantized dtype");
    }
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float16 &&
        self.dtype() != DType::BFloat16 && self.dtype() != DType::Float64) {
        TP_THROW(TypeError,
                 "quantize_affine(): expected a floating point tensor");
    }
    return quantize_affine_cuda_impl<true>(self, block_size, scales,
                                           zero_points, dtype, quant_min,
                                           quant_max);
}

Tensor dequantize_affine_cuda(const Tensor& self,
                              const std::vector<int64_t>& block_size,
                              const Tensor& scales,
                              const std::optional<Tensor>& zero_points) {
    const DType dtype = self.dtype();
    if (!isQuantizedType(dtype)) {
        TP_THROW(TypeError,
                 "dequantize_affine(): expected a quantized tensor");
    }
    return quantize_affine_cuda_impl<false>(self, block_size, scales,
                                            zero_points, dtype, 0, 0);
}

// ---- FP8 affine quantization (quantize_affine_float8 family) ----
//
// The scale tensor carries the block structure: a shape equal to the input
// means per-element scales, a scalar means per-tensor, a singleton broadcast
// dimension means per-channel/per-row, and a shape `input / block_size`
// means per-group.  The element-to-scale mapping reuses the block grid from
// the integer path, so `cuda_block_of` indexes the scale buffer directly.

namespace {

std::vector<int64_t> float8_scale_block_size(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& scale_sizes, const char* op) {
    std::vector<int64_t> block_size(sizes.size());
    for (size_t d = 0; d < sizes.size(); ++d) {
        if (scale_sizes[d] == 1) {
            block_size[d] = sizes[d];
        } else {
            if (sizes[d] % scale_sizes[d] != 0) {
                TP_THROW(ValueError, op,
                         ": input size must be divisible by the scale size "
                         "along every non-broadcast dimension");
            }
            block_size[d] = sizes[d] / scale_sizes[d];
        }
    }
    return block_size;
}

float float8_max_value(DType dtype) {
    switch (dtype) {
        case DType::Float8_e4m3fn:
            return 448.0f;
        case DType::Float8_e4m3fnuz:
            return 240.0f;
        case DType::Float8_e5m2:
            return 57344.0f;
        case DType::Float8_e5m2fnuz:
            return 57344.0f;
        default:
            TP_THROW(TypeError,
                     "float8 quantization requires a float8 dtype, got ",
                     toString(dtype));
    }
}

struct Float8LaunchParams {
    int64_t rank = 0;
    int64_t num_blocks = 0;
    int64_t* d_sizes = nullptr;
    int64_t* d_blocks = nullptr;
    int64_t* d_block_stride = nullptr;
    int64_t* d_size_stride = nullptr;
    float* d_scales = nullptr;
};

Float8LaunchParams upload_float8_affine_params(
    const AffineGroupParamsCuda& p, const std::vector<float>& scale_buf) {
    Float8LaunchParams lp;
    lp.rank = p.dim;
    lp.num_blocks = p.num_blocks;
    const int64_t rank_alloc = std::max<int64_t>(p.dim, 1);
    const int64_t num_blocks_alloc = std::max<int64_t>(p.num_blocks, 1);
    checkCuda(cudaMalloc(&lp.d_sizes, rank_alloc * sizeof(int64_t)),
            "quantize_affine_float8 cudaMalloc sizes");
    checkCuda(cudaMalloc(&lp.d_blocks, rank_alloc * sizeof(int64_t)),
            "quantize_affine_float8 cudaMalloc blocks");
    checkCuda(cudaMalloc(&lp.d_block_stride, rank_alloc * sizeof(int64_t)),
            "quantize_affine_float8 cudaMalloc block_stride");
    checkCuda(cudaMalloc(&lp.d_size_stride, rank_alloc * sizeof(int64_t)),
            "quantize_affine_float8 cudaMalloc size_stride");
    if (p.dim > 0) {
        checkCuda(cudaMemcpy(lp.d_sizes, p.sizes.data(),
                             p.dim * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "quantize_affine_float8 cudaMemcpy sizes");
        checkCuda(cudaMemcpy(lp.d_blocks, p.blocks.data(),
                             p.dim * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "quantize_affine_float8 cudaMemcpy blocks");
        checkCuda(cudaMemcpy(lp.d_block_stride, p.block_stride.data(),
                             p.dim * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "quantize_affine_float8 cudaMemcpy block_stride");
        checkCuda(cudaMemcpy(lp.d_size_stride, p.size_stride.data(),
                             p.dim * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "quantize_affine_float8 cudaMemcpy size_stride");
    }
    checkCuda(cudaMalloc(&lp.d_scales, num_blocks_alloc * sizeof(float)),
              "quantize_affine_float8 cudaMalloc scales");
    checkCuda(cudaMemcpy(lp.d_scales, scale_buf.data(),
                         p.num_blocks * sizeof(float),
                         cudaMemcpyHostToDevice),
              "quantize_affine_float8 cudaMemcpy scales");
    return lp;
}

void free_float8_affine_params(Float8LaunchParams& lp) {
    checkCuda(cudaFree(lp.d_sizes), "quantize_affine_float8 cudaFree sizes");
    checkCuda(cudaFree(lp.d_blocks), "quantize_affine_float8 cudaFree blocks");
    checkCuda(cudaFree(lp.d_block_stride),
              "quantize_affine_float8 cudaFree block_stride");
    checkCuda(cudaFree(lp.d_size_stride),
              "quantize_affine_float8 cudaFree size_stride");
    checkCuda(cudaFree(lp.d_scales),
              "quantize_affine_float8 cudaFree scales");
}

template <typename Float8T>
__global__ void quantize_affine_float8_group_kernel(
    const float* input, Float8T* output, int64_t numel, int64_t dim,
    const int64_t* sizes, const int64_t* blocks, const int64_t* block_stride,
    const int64_t* size_stride, const float* scales, float max_value) {
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t b =
        cuda_block_of(i, dim, sizes, blocks, block_stride, size_stride);
    const float scaled = input[i] / scales[b];
    const float clamped = fmaxf(-max_value, fminf(max_value, scaled));
    output[i] = Float8T(clamped);
}

template <typename Float8T>
__global__ void dequantize_affine_float8_group_kernel(
    const Float8T* input, float* output, int64_t numel, int64_t dim,
    const int64_t* sizes, const int64_t* blocks, const int64_t* block_stride,
    const int64_t* size_stride, const float* scales) {
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t b =
        cuda_block_of(i, dim, sizes, blocks, block_stride, size_stride);
    output[i] = static_cast<float>(input[i]) * scales[b];
}

__device__ __forceinline__ int64_t cuda_block_base(
    int64_t b, int64_t dim, const int64_t* sizes, const int64_t* blocks,
    const int64_t* block_stride, const int64_t* size_stride) {
    int64_t base = 0;
    for (int64_t d = 0; d < dim; ++d) {
        const int64_t per = (sizes[d] + blocks[d] - 1) / blocks[d];
        const int64_t bc = (b / block_stride[d]) % per;
        base += (bc * blocks[d]) * size_stride[d];
    }
    return base;
}

__global__ void choose_scale_float8_kernel(
    const float* input, float* output, int64_t num_blocks, int64_t dim,
    const int64_t* sizes, const int64_t* blocks, const int64_t* block_stride,
    const int64_t* size_stride, float lb, float ub, float max_value) {
    const int64_t b =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;
    float max_abs = 0.f;
    const int64_t base =
        cuda_block_base(b, dim, sizes, blocks, block_stride, size_stride);
    int64_t block_numel = 1;
    for (int64_t d = 0; d < dim; ++d) block_numel *= blocks[d];
    for (int64_t t = 0; t < block_numel; ++t) {
        int64_t local = t;
        int64_t idx = base;
        for (int64_t d = dim - 1; d >= 0; --d) {
            const int64_t td = local % blocks[d];
            local /= blocks[d];
            idx += td * size_stride[d];
        }
        max_abs = fmaxf(max_abs, fabsf(input[idx]));
    }
    max_abs = fmaxf(max_abs, lb);
    max_abs = fminf(max_abs, ub);
    output[b] = max_abs / max_value;
}

} // namespace

Tensor quantize_affine_float8_cuda(const Tensor& self, const Tensor& scale,
                                   DType dtype) {
    if (!isFloat8Type(dtype)) {
        TP_THROW(TypeError,
                 "quantize_affine_float8(): expected a float8 dtype, got ",
                 toString(dtype));
    }
    const QuantInputs prepared = prepare_quantize(self);
    const Tensor& input = prepared.input;
    const int64_t dim = input.dim();
    if (scale.dim() != dim && scale.numel() != 1) {
        TP_THROW(ValueError,
                 "quantize_affine_float8(): scale rank must match the input "
                 "rank unless it is a scalar");
    }
    if (scale.device() != self.device()) {
        TP_THROW(RuntimeError,
                 "quantize_affine_float8(): scale must be on the input "
                 "device");
    }
    const float max_value = float8_max_value(dtype);
    Tensor out = Tensor::empty(input.shape(), dtype, self.device());
    if (input.numel() == 0) return out;

    const Tensor sc =
        scale.to(Device(DeviceType::CPU), DType::Float32).contiguous();
    std::vector<int64_t> scale_sizes(dim, 1);
    if (sc.numel() != 1) {
        for (int64_t d = 0; d < dim; ++d) scale_sizes[d] = sc.size(d);
    }
    const std::vector<int64_t> block_size =
        float8_scale_block_size(input.shape(), scale_sizes,
                               "quantize_affine_float8");
    const AffineGroupParamsCuda p =
        make_affine_group_params_cuda(input.shape(), block_size);
    std::vector<float> scale_buf(p.num_blocks);
    if (sc.numel() == 1) {
        std::fill(scale_buf.begin(), scale_buf.end(), *sc.data_ptr<float>());
    } else {
        std::memcpy(scale_buf.data(), sc.data_ptr<float>(),
                    sizeof(float) * p.num_blocks);
    }
    Float8LaunchParams lp = upload_float8_affine_params(p, scale_buf);
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int64_t numel = input.numel();
    const int threads = 128;
    const int blocks = static_cast<int>(
        std::min<int64_t>((numel + threads - 1) / threads, 65535));
    const float* in = input.data_ptr<float>();
    switch (dtype) {
        case DType::Float8_e4m3fn:
            quantize_affine_float8_group_kernel<Float8_e4m3fn>
                <<<blocks, threads, 0, stream>>>(
                    in, out.data_ptr<Float8_e4m3fn>(), numel, lp.rank,
                    lp.d_sizes, lp.d_blocks, lp.d_block_stride,
                    lp.d_size_stride, lp.d_scales, max_value);
            break;
        case DType::Float8_e4m3fnuz:
            quantize_affine_float8_group_kernel<Float8_e4m3fnuz>
                <<<blocks, threads, 0, stream>>>(
                    in, out.data_ptr<Float8_e4m3fnuz>(), numel, lp.rank,
                    lp.d_sizes, lp.d_blocks, lp.d_block_stride,
                    lp.d_size_stride, lp.d_scales, max_value);
            break;
        case DType::Float8_e5m2:
            quantize_affine_float8_group_kernel<Float8_e5m2>
                <<<blocks, threads, 0, stream>>>(
                    in, out.data_ptr<Float8_e5m2>(), numel, lp.rank,
                    lp.d_sizes, lp.d_blocks, lp.d_block_stride,
                    lp.d_size_stride, lp.d_scales, max_value);
            break;
        case DType::Float8_e5m2fnuz:
            quantize_affine_float8_group_kernel<Float8_e5m2fnuz>
                <<<blocks, threads, 0, stream>>>(
                    in, out.data_ptr<Float8_e5m2fnuz>(), numel, lp.rank,
                    lp.d_sizes, lp.d_blocks, lp.d_block_stride,
                    lp.d_size_stride, lp.d_scales, max_value);
            break;
        default:
            break;
    }
    checkCuda(cudaGetLastError(), "quantize_affine_float8 kernel");
    free_float8_affine_params(lp);
    return out;
}

Tensor dequantize_affine_float8_cuda(const Tensor& self, const Tensor& scale,
                                     DType output_dtype) {
    if (!isFloat8Type(self.dtype())) {
        TP_THROW(TypeError,
                 "dequantize_affine_float8(): expected a float8 tensor, got ",
                 toString(self.dtype()));
    }
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    const int64_t dim = input.dim();
    if (scale.dim() != dim && scale.numel() != 1) {
        TP_THROW(ValueError,
                 "dequantize_affine_float8(): scale rank must match the "
                 "input rank unless it is a scalar");
    }
    if (scale.device() != self.device()) {
        TP_THROW(RuntimeError,
                 "dequantize_affine_float8(): scale must be on the input "
                 "device");
    }
    Tensor out = Tensor::empty(input.shape(), DType::Float32, self.device());
    if (input.numel() == 0) {
        if (output_dtype != DType::Float32) out = out.to(output_dtype);
        return out;
    }

    const Tensor sc =
        scale.to(Device(DeviceType::CPU), DType::Float32).contiguous();
    std::vector<int64_t> scale_sizes(dim, 1);
    if (sc.numel() != 1) {
        for (int64_t d = 0; d < dim; ++d) scale_sizes[d] = sc.size(d);
    }
    const std::vector<int64_t> block_size =
        float8_scale_block_size(input.shape(), scale_sizes,
                                "dequantize_affine_float8");
    const AffineGroupParamsCuda p =
        make_affine_group_params_cuda(input.shape(), block_size);
    std::vector<float> scale_buf(p.num_blocks);
    if (sc.numel() == 1) {
        std::fill(scale_buf.begin(), scale_buf.end(), *sc.data_ptr<float>());
    } else {
        std::memcpy(scale_buf.data(), sc.data_ptr<float>(),
                    sizeof(float) * p.num_blocks);
    }
    Float8LaunchParams lp = upload_float8_affine_params(p, scale_buf);
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int64_t numel = input.numel();
    const int threads = 128;
    const int blocks = static_cast<int>(
        std::min<int64_t>((numel + threads - 1) / threads, 65535));
    float* outp = out.data_ptr<float>();
    switch (self.dtype()) {
        case DType::Float8_e4m3fn:
            dequantize_affine_float8_group_kernel<Float8_e4m3fn>
                <<<blocks, threads, 0, stream>>>(
                    input.data_ptr<Float8_e4m3fn>(), outp, numel, lp.rank,
                    lp.d_sizes, lp.d_blocks, lp.d_block_stride,
                    lp.d_size_stride, lp.d_scales);
            break;
        case DType::Float8_e4m3fnuz:
            dequantize_affine_float8_group_kernel<Float8_e4m3fnuz>
                <<<blocks, threads, 0, stream>>>(
                    input.data_ptr<Float8_e4m3fnuz>(), outp, numel, lp.rank,
                    lp.d_sizes, lp.d_blocks, lp.d_block_stride,
                    lp.d_size_stride, lp.d_scales);
            break;
        case DType::Float8_e5m2:
            dequantize_affine_float8_group_kernel<Float8_e5m2>
                <<<blocks, threads, 0, stream>>>(
                    input.data_ptr<Float8_e5m2>(), outp, numel, lp.rank,
                    lp.d_sizes, lp.d_blocks, lp.d_block_stride,
                    lp.d_size_stride, lp.d_scales);
            break;
        case DType::Float8_e5m2fnuz:
            dequantize_affine_float8_group_kernel<Float8_e5m2fnuz>
                <<<blocks, threads, 0, stream>>>(
                    input.data_ptr<Float8_e5m2fnuz>(), outp, numel, lp.rank,
                    lp.d_sizes, lp.d_blocks, lp.d_block_stride,
                    lp.d_size_stride, lp.d_scales);
            break;
        default:
            break;
    }
    checkCuda(cudaGetLastError(), "dequantize_affine_float8 kernel");
    free_float8_affine_params(lp);
    if (output_dtype != DType::Float32) {
        out = out.to(output_dtype);
    }
    return out;
}

Tensor choose_scale_float8_cuda(const Tensor& self,
                             const std::vector<int64_t>& block_size,
                             DType float8_dtype,
                             std::optional<double> hp_value_lb,
                             std::optional<double> hp_value_ub) {
    if (!isFloat8Type(float8_dtype)) {
        TP_THROW(TypeError,
                 "choose_scale_float8(): expected a float8 dtype, got ",
                 toString(float8_dtype));
    }
    const QuantInputs prepared = prepare_quantize(self);
    const Tensor& input = prepared.input;
    const int64_t dim = input.dim();
    if (static_cast<int64_t>(block_size.size()) != dim) {
        TP_THROW(ValueError,
                 "choose_scale_float8(): block_size rank must match the "
                 "input rank");
    }
    for (int64_t d = 0; d < dim; ++d) {
        if (block_size[d] <= 0 || input.size(d) % block_size[d] != 0) {
            TP_THROW(ValueError,
                     "choose_scale_float8(): input size must be divisible by "
                     "block_size along every dimension");
        }
    }
    const float max_value = float8_max_value(float8_dtype);
    const AffineGroupParamsCuda p =
        make_affine_group_params_cuda(input.shape(), block_size);
    Tensor out = Tensor::empty(p.blocks_per_dim, DType::Float32, self.device());
    if (input.numel() == 0) return out;

    const float lb = hp_value_lb.has_value()
        ? static_cast<float>(*hp_value_lb)
        : -std::numeric_limits<float>::infinity();
    const float ub = hp_value_ub.has_value()
        ? static_cast<float>(*hp_value_ub)
        : std::numeric_limits<float>::infinity();

    const int64_t rank_alloc = std::max<int64_t>(p.dim, 1);
    const int64_t num_blocks_alloc = std::max<int64_t>(p.num_blocks, 1);
    int64_t* d_sizes = nullptr;
    int64_t* d_blocks = nullptr;
    int64_t* d_block_stride = nullptr;
    int64_t* d_size_stride = nullptr;
    checkCuda(cudaMalloc(&d_sizes, rank_alloc * sizeof(int64_t)),
              "choose_scale_float8 cudaMalloc sizes");
    checkCuda(cudaMalloc(&d_blocks, rank_alloc * sizeof(int64_t)),
              "choose_scale_float8 cudaMalloc blocks");
    checkCuda(cudaMalloc(&d_block_stride, rank_alloc * sizeof(int64_t)),
              "choose_scale_float8 cudaMalloc block_stride");
    checkCuda(cudaMalloc(&d_size_stride, rank_alloc * sizeof(int64_t)),
              "choose_scale_float8 cudaMalloc size_stride");
    if (p.dim > 0) {
        checkCuda(cudaMemcpy(d_sizes, p.sizes.data(),
                             p.dim * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "choose_scale_float8 cudaMemcpy sizes");
        checkCuda(cudaMemcpy(d_blocks, p.blocks.data(),
                             p.dim * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "choose_scale_float8 cudaMemcpy blocks");
        checkCuda(cudaMemcpy(d_block_stride, p.block_stride.data(),
                             p.dim * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "choose_scale_float8 cudaMemcpy block_stride");
        checkCuda(cudaMemcpy(d_size_stride, p.size_stride.data(),
                             p.dim * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "choose_scale_float8 cudaMemcpy size_stride");
    }

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>(
        std::min<int64_t>((p.num_blocks + threads - 1) / threads, 65535));
    choose_scale_float8_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), out.data_ptr<float>(), p.num_blocks, p.dim,
        d_sizes, d_blocks, d_block_stride, d_size_stride, lb, ub, max_value);
    checkCuda(cudaGetLastError(), "choose_scale_float8 kernel");
    checkCuda(cudaFree(d_sizes), "choose_scale_float8 cudaFree sizes");
    checkCuda(cudaFree(d_blocks), "choose_scale_float8 cudaFree blocks");
    checkCuda(cudaFree(d_block_stride),
              "choose_scale_float8 cudaFree block_stride");
    checkCuda(cudaFree(d_size_stride),
              "choose_scale_float8 cudaFree size_stride");
    return out;
}


TENSORPLAY_LIBRARY_IMPL(CUDA, AffineQuantKernels) {
    m.impl("_empty_affine_quantized", empty_affine_quantized_cuda);
    m.impl("_empty_per_channel_affine_quantized",
           empty_per_channel_affine_quantized_cuda);
    m.impl("empty_quantized", empty_quantized_cuda);
    m.impl("quantize_per_tensor", quantize_per_tensor_cuda);
    m.impl("quantize_per_channel", quantize_per_channel_cuda);
    m.impl("quantize_affine", quantize_affine_cuda);
    m.impl("dequantize_affine", dequantize_affine_cuda);
    m.impl("quantize_affine_float8", quantize_affine_float8_cuda);
    m.impl("dequantize_affine_float8", dequantize_affine_float8_cuda);
    m.impl("choose_scale_float8", choose_scale_float8_cuda);
}

}
}
