#pragma once

#include "QuantTensorOperators.h"
#include "QuantNative.h"
#include "cpu/QuantizedElementwiseKernels.h"
#include "Parallel.h"
#include <algorithm>
#include <numeric>

namespace tensorplay::quantized {

Tensor storage_view(const Tensor& self) {
    require_quantized(self, "integer storage view");
    TP_CHECK(self.dtype() == DType::QInt8 || self.dtype() == DType::QUInt8 || self.dtype() == DType::QInt32,
             "integer storage view requires byte or word quantization");
    return Tensor(self.impl()->storage(), self.sizes().vec(), self.strides(),
                  underlying_storage_type(self.dtype()), self.storage_offset());
}

Tensor with_quantizer(const Tensor& codes, const Tensor& self) {
    Tensor out(codes.impl()->storage(), codes.sizes().vec(), codes.strides(),
               self.dtype(), codes.storage_offset());
    out.impl()->set_quantizer(quantizer_of(self));
    return out;
}

namespace { Scalar scalar_code(const Tensor& self, const Scalar& value, bool copy_rounding = false); }

Tensor activation(const Tensor& self, int operation, double a, double b) {
    double scale = q_scale(self);
    int64_t zp = q_zero_point(self);
    if (operation == 3 || operation == 4) {
        scale = self.dtype() == DType::QInt32 ? 0x1p-32 : 0x1p-8;
        zp = self.dtype() == DType::QInt8 ? -128 : 0;
    } else if (operation == 5) {
        scale = self.dtype() == DType::QInt32 ? 0x1p-31 : 0x1p-7;
        zp = self.dtype() == DType::QUInt8 ? 128 : 0;
    } else if (operation == 8 || operation == 9) {
        scale = static_cast<float>(scale);
    }
    if (operation == 10) return Tensor::quantized_relu(self);
    const quant::Activation fn{operation, static_cast<float>(a), static_cast<float>(b)};
    if (self.device().is_cpu()) return quant::activation_cpu(self, scale, zp, fn);
#ifdef USE_CUDA
    if (self.device().is_cuda()) return quant::activation_cuda(self, scale, zp, fn);
#endif
    TP_THROW(NotImplementedError, "quantized activation is unavailable on this device");
}

namespace {
Scalar scalar_code(const Tensor& self, const Scalar& value, bool copy_rounding) {
    const float v = static_cast<float>(value.toDouble());
    const float inverse = 1.f / static_cast<float>(q_scale(self));
    const int64_t zp = q_zero_point(self);
    if (copy_rounding && std::isnan(v)) {
        return Scalar(static_cast<int64_t>(self.dtype() == DType::QInt8 ? INT8_MAX :
            self.dtype() == DType::QUInt8 ? UINT8_MAX : INT32_MIN));
    }
    switch (self.dtype()) {
        case DType::QInt8: return Scalar(static_cast<int64_t>(quant::encode<int8_t>(v, inverse, zp)));
        case DType::QUInt8: return Scalar(static_cast<int64_t>(quant::encode<uint8_t>(v, inverse, zp)));
        case DType::QInt32: {
            if (copy_rounding) {
                const float position = std::nearbyint(v * inverse) + static_cast<float>(zp);
                return Scalar(static_cast<int64_t>(std::clamp(static_cast<double>(position),
                    static_cast<double>(INT32_MIN), static_cast<double>(INT32_MAX))));
            }
            return Scalar(static_cast<int64_t>(quant::encode<int32_t>(v, inverse, zp)));
        }
        default: TP_THROW(TypeError, "scalar encoding requires byte or word quantization");
    }
}
}

Tensor& activation_out(const Tensor& self, Tensor& out, int operation, double a, double b) {
    TP_CHECK(self.dtype() == out.dtype() && self.device() == out.device(),
             "Activation input and output must share dtype and device");
    Tensor result;
    const quant::Activation fn{operation, static_cast<float>(a), static_cast<float>(b)};
    if (self.device().is_cpu()) result = quant::activation_cpu(self, q_scale(out), q_zero_point(out), fn);
#ifdef USE_CUDA
    else if (self.device().is_cuda()) result = quant::activation_cuda(self, q_scale(out), q_zero_point(out), fn);
#endif
    else TP_THROW(NotImplementedError, "Quantized activation is unavailable on this device");
    if (out.sizes().vec() != result.sizes().vec()) out.resize_(result.sizes().vec());
    out.copy_(result);
    return out;
}

Tensor& relu_inplace(Tensor& self) {
    const int64_t zp = q_zero_point(self);
    Tensor codes = storage_view(self);
    if (self.device().is_cpu() && self.is_contiguous()) {
        int64_t maximum = self.dtype() == DType::QUInt8 ? 255 : self.dtype() == DType::QInt8 ? 127 : INT32_MAX;
        auto* data = static_cast<uint8_t*>(self.data_ptr());
        parallel::parallel_for(0, self.numel(), 32768, [&](int64_t first, int64_t last) {
            cpu::quantized_clip_stub(DeviceType::CPU, data + first * self.itemsize(),
                data + first * self.itemsize(), last - first, self.dtype(), zp, maximum);
        });
    } else codes.clamp_min_(Scalar(zp));
    return self;
}

Tensor& fill(Tensor& self, const Scalar& value) {
    if (quantizer_of(self)->qscheme() != kPerTensorAffine ||
        (self.dtype() != DType::QInt8 && self.dtype() != DType::QUInt8 && self.dtype() != DType::QInt32)) {
        self.copy_(Tensor::full(self.sizes().vec(), value, DType::Float32, self.device()));
        return self;
    }
    Tensor codes = storage_view(self);
    codes.fill_(scalar_code(self, value, true));
    return self;
}

Tensor& masked_fill(Tensor& self, const Tensor& mask, const Scalar& value) {
    TP_CHECK(quantizer_of(self)->qscheme() == kPerTensorAffine,
             "masked fill requires uniform affine quantization");
    Tensor codes = storage_view(self);
    codes.masked_fill_(mask, scalar_code(self, value));
    return self;
}

Tensor extrema(const Tensor& self, bool maximum) {
    q_scale(self);
    const Tensor codes = storage_view(self);
    return with_quantizer(maximum ? codes.max() : codes.min(), self);
}

std::tuple<Tensor, Tensor> extrema_dim(const Tensor& self, int64_t dim, bool keepdim, bool maximum) {
    q_scale(self);
    const Tensor codes = storage_view(self);
    auto result = maximum ? codes.max(dim, keepdim) : codes.min(dim, keepdim);
    return {with_quantizer(std::get<0>(result), self), std::get<1>(result)};
}

Tensor moment(const Tensor& self, const std::vector<int64_t>& dimensions, bool keepdim,
              bool standard_deviation, double correction, DType dtype) {
    const double scale = q_scale(self);
    const int64_t zp = q_zero_point(self);
    TP_CHECK(dtype == DType::Undefined || dtype == self.dtype(),
             "quantized moments require the input quantized dtype");
    auto dims = dimensions;
    if (dims.empty()) {
        dims.resize(self.dim());
        std::iota(dims.begin(), dims.end(), int64_t{0});
    }
    const int64_t rank = std::max<int64_t>(self.dim(), 1);
    for (auto& d : dims) {
        if (d < 0) d += rank;
        TP_CHECK(d >= 0 && d < rank, "reduction dimension is out of range");
    }
    std::sort(dims.begin(), dims.end());
    TP_CHECK(std::adjacent_find(dims.begin(), dims.end()) == dims.end(),
             "reduction dimensions must be unique");
    int64_t count = 1;
    for (auto d : dims) if (self.dim()) count *= self.size(d);
    const bool inner = dims.empty() || (dims.back() == self.dim() - 1 &&
        dims.back() - dims.front() + 1 == static_cast<int64_t>(dims.size()));
    if (self.impl()->is_contiguous_in(MemoryFormat::Contiguous) && inner && count > 0 &&
        (!standard_deviation || count != 1 || correction == 0.)) {
        if (self.device().is_cpu()) return quant::moment_inner_cpu(self, dims, keepdim, standard_deviation, correction);
#ifdef USE_CUDA
        if (self.device().is_cuda()) return quant::moment_inner_cuda(self, dims, keepdim, standard_deviation, correction);
#endif
    }
    if (!standard_deviation) {
        const auto format = self.dim() == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::ChannelsLast3d;
        if (self.device().is_cpu() && self.itemsize() == 1 && (self.dim() == 4 || self.dim() == 5) &&
            dims.size() == static_cast<size_t>(self.dim() - 2) && dims.front() == 2 &&
            dims.back() == self.dim() - 1 && count > 0 && self.size(0) * self.size(1) > 1 &&
            self.impl()->is_contiguous_in(format)) {
            const int64_t batch = self.size(0), channels = self.size(1);
            std::vector<int64_t> shape{batch, channels};
            if (keepdim) shape.resize(self.dim(), 1);
            Tensor result = Tensor::empty(shape, self.dtype(), self.device());
            result.impl()->set_quantizer(quantizer_of(self));
            const auto* input = static_cast<const uint8_t*>(self.data_ptr());
            auto* output = static_cast<uint8_t*>(result.data_ptr());
            auto run = [&](int64_t b, int64_t begin, int64_t end) {
                cpu::quantized_spatial_mean_stub(DeviceType::CPU, input + b * count * channels,
                    output + b * channels, count, channels, begin, end, self.dtype(), static_cast<float>(scale), zp);
            };
            const int threads = parallel::get_num_threads();
            if (self.numel() < 32768 || threads == 1 || parallel::in_parallel_region()) {
                for (int64_t b = 0; b < batch; ++b) run(b, 0, channels);
            } else if (batch >= threads || (channels < threads && batch > channels)) {
                parallel::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
                    for (int64_t b = begin; b < end; ++b) run(b, 0, channels);
                });
            } else {
                parallel::parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
                    begin -= begin % 32;
                    if (end != channels) end -= end % 32;
                    if (begin != end) for (int64_t b = 0; b < batch; ++b) run(b, begin, end);
                });
            }
            return result;
        }
        const Tensor real = self.dequantize();
        return Tensor::quantize_per_tensor(real.mean(dims, keepdim, dtype), scale, zp, self.dtype());
    }
    if (self.device().is_cpu()) return quant::std_strided_cpu(self, dims, keepdim, correction);
#ifdef USE_CUDA
    if (self.device().is_cuda()) return quant::std_strided_cuda(self, dims, keepdim, correction);
#endif
    TP_THROW(NotImplementedError, "Quantized standard deviation is unavailable on this device");
}

Tensor& index_put(Tensor& self, const std::vector<std::optional<Tensor>>& indices,
                  const Tensor& values, bool accumulate, bool unsafe) {
    TP_CHECK(!isQuantizedType(values.dtype()), "indexed values must be unquantized");
    TP_CHECK(!accumulate, "quantized indexed assignment does not support accumulation");
    const double scale = q_scale(self);
    const int64_t zp = q_zero_point(self);
    Tensor target = storage_view(self);
    Tensor input = values.to(DType::Float32);
    if (input.device() != self.device() && input.dim() == 0) input = input.to(self.device());
    Tensor value_codes = Tensor::quantize_per_tensor(input, scale, zp, self.dtype()).int_repr();
    Tensor::_index_put_impl_(target, indices, value_codes, false, unsafe);
    return self;
}

Tensor& cat_out(const std::vector<Tensor>& tensors, int64_t dim, Tensor& out) {
    TP_CHECK(!tensors.empty(), "concatenation requires a nonempty tensor list");
    for (const auto& tensor : tensors) {
        TP_CHECK(tensor.dtype() == out.dtype() && tensor.device() == out.device(),
                 "concatenation inputs and output must share dtype and device");
    }
    Tensor result;
    if (out.device().is_cpu()) result = quant::cat_cpu(tensors, dim, q_scale(out), q_zero_point(out), false);
#ifdef USE_CUDA
    else if (out.device().is_cuda()) result = quant::cat_cuda(tensors, dim, q_scale(out), q_zero_point(out), false);
#endif
    else TP_THROW(NotImplementedError, "quantized concatenation is unavailable on this device");
    if (out.sizes().vec() != result.sizes().vec()) out.resize_(result.sizes().vec());
    out.copy_(result);
    return out;
}

Tensor prelu(const Tensor& self, const Tensor& weight) {
    q_scale(self);
    return Tensor::quantized_prelu(self, weight, q_scale(self), q_zero_point(self));
}

Tensor grid_sample(const Tensor& self, const Tensor& grid, int64_t interpolation,
                   int64_t padding, bool align_corners) {
    if (self.device().is_cpu()) return quant::grid_sample_cpu(self, grid, interpolation, padding, align_corners);
#ifdef USE_CUDA
    if (self.device().is_cuda()) return quant::grid_sample_cuda(self, grid, interpolation, padding, align_corners);
#endif
    TP_THROW(NotImplementedError, "Quantized grid sampling is unavailable on this device");
}

} // namespace tensorplay::quantized
