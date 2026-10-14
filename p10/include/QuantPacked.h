#pragma once
#include "Tensor.h"
#include "Quantizer.h"
#include "Exception.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>
#include <type_traits>

namespace tensorplay::cuda {
Tensor quantized_relu_cuda(const Tensor& input);
}

namespace tensorplay::quant {

struct PackedDeviceGeometry {
    int64_t batch = 1, groups = 1, channels = 0, outputs = 0;
    int64_t rows = 0, spatial = 1, input_spatial = 1, kernel_elements = 1;
    int64_t K = 0, padded_K = 0, rank = 0;
    int64_t in[3]{1, 1, 1}, out[3]{1, 1, 1}, kernel[3]{1, 1, 1};
    int64_t stride[3]{1, 1, 1}, padding[3]{0, 0, 0}, dilation[3]{1, 1, 1};
    bool linear = true;
    int64_t activation = 0;
    double negative_slope = 0.;
    const void* residual = nullptr;
    double residual_scale = 1.;
    int64_t residual_zero_point = 0;
    bool residual_unsigned = false;
};

#if defined(__CUDACC__)
__host__ __device__
#endif
inline double packed_activation(double value, int64_t index, const PackedDeviceGeometry& g) {
    if (g.residual) {
        const int64_t code = g.residual_unsigned
            ? static_cast<const uint8_t*>(g.residual)[index]
            : static_cast<const int8_t*>(g.residual)[index];
        value += (code - g.residual_zero_point) * g.residual_scale;
    }
    if (g.activation == 1) return value >= 0. ? value : value * g.negative_slope;
    if (g.activation == 2) return ::tanh(value);
    return value;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
inline double packed_real_value(int64_t accumulator,double input_scale,float weight_scale,
        float bias,bool output_float) {
    if (output_float)
        return ::fmaf(static_cast<float>(accumulator),static_cast<float>(input_scale)*weight_scale,bias);
    return static_cast<double>(accumulator)*input_scale*weight_scale+bias;
}

struct PackedGeometry : PackedDeviceGeometry {
    std::vector<int64_t> shape;
};

inline PackedGeometry packed_geometry(
    const Tensor& input, const Tensor& packed, const Tensor& offsets,
    const Tensor& scales, const Tensor& zero_points, const std::optional<Tensor>& bias,
    const std::vector<int64_t>& weight_sizes, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    int64_t groups, double output_scale, int64_t output_zero_point, bool output_float) {
    TP_CHECK(input.dtype() == DType::QInt8 || input.dtype() == DType::QUInt8,
             "Packed compute requires signed or unsigned 8-bit quantized input");
    TP_CHECK(weight_sizes.size() >= 2 && weight_sizes.size() <= 5,
             "Packed weight rank must be between two and five");
    TP_CHECK(packed.dtype() == DType::Int8 && packed.dim() == 2 && packed.is_contiguous(),
             "Packed weights must be a contiguous 8-bit matrix");
    PackedGeometry g;
    g.linear = weight_sizes.size() == 2;
    g.rank = weight_sizes.size() - 2;
    g.outputs = weight_sizes[0];
    g.K = weight_sizes[1];
    TP_CHECK(g.outputs >= 0 && g.K >= 0 && groups > 0, "Invalid packed weight dimensions or groups");
    if (g.linear) {
        TP_CHECK(input.dim() >= 2 && input.size(input.dim() - 1) == g.K && groups == 1,
                 "Packed linear input features do not match the weights");
        g.rows = 1;
        for (int64_t d = 0; d < input.dim() - 1; ++d) g.rows *= input.size(d);
        g.shape = input.shape();
        g.shape.back() = g.outputs;
    } else {
        TP_CHECK(input.dim() == g.rank + 2 && stride.size() == g.rank &&
                 padding.size() == g.rank && dilation.size() == g.rank,
                 "Packed convolution input and parameter ranks do not match");
        g.batch = input.size(0); g.channels = input.size(1); g.groups = groups;
        TP_CHECK(g.channels == weight_sizes[1] * groups && g.outputs % groups == 0,
                 "Packed convolution channels are incompatible with groups");
        g.shape = {g.batch, g.outputs};
        for (int64_t d = 0; d < g.rank; ++d) {
            TP_CHECK(weight_sizes[d + 2] > 0 && stride[d] > 0 && padding[d] >= 0 && dilation[d] > 0,
                     "Invalid packed convolution kernel, stride, padding or dilation");
            g.in[d] = input.size(d + 2); g.kernel[d] = weight_sizes[d + 2];
            g.stride[d] = stride[d]; g.padding[d] = padding[d]; g.dilation[d] = dilation[d];
            const int64_t extent = g.in[d] + 2 * padding[d] - dilation[d] * (g.kernel[d] - 1) - 1;
            TP_CHECK(extent >= 0, "Packed convolution kernel exceeds the padded input");
            g.out[d] = extent / stride[d] + 1;
            g.spatial *= g.out[d]; g.input_spatial *= g.in[d];
            g.kernel_elements *= g.kernel[d]; g.K *= g.kernel[d];
            g.shape.push_back(g.out[d]);
        }
        g.rows = g.batch * g.spatial;
    }
    g.padded_K = ((g.K + 31) / 32) * 32;
    TP_CHECK(packed.size(0) == g.outputs && packed.size(1) == g.padded_K,
             "Packed weight shape does not match its logical dimensions");
    for (const auto* value : {&offsets, &scales, &zero_points}) {
        TP_CHECK(value->device() == input.device() && value->dim() == 1 &&
                 value->numel() == g.outputs && value->is_contiguous(),
                 "Packed parameters must be contiguous output-channel vectors on the input device");
    }
    TP_CHECK(packed.device() == input.device() && offsets.dtype() == DType::Int64 &&
             scales.dtype() == DType::Float32 && zero_points.dtype() == DType::Int64,
             "Invalid packed parameter device or dtype");
    if (bias) TP_CHECK(bias->device() == input.device() && bias->dtype() == DType::Float32 &&
                       bias->dim() == 1 && bias->numel() == g.outputs && bias->is_contiguous(),
                       "Packed bias must be a contiguous float output-channel vector on the input device");
    if (!output_float) {
        const int64_t lo = input.dtype() == DType::QUInt8 ? 0 : -128;
        TP_CHECK(std::isfinite(output_scale) && output_scale > 0 &&
                 output_zero_point >= lo && output_zero_point <= lo + 255,
                 "Invalid packed output scale or zero point");
    }
    quantized::q_scale(input);
    return g;
}

inline std::pair<int64_t, int64_t> packed_weight_shape(const Tensor& weight) {
    TP_CHECK(weight.dtype() == DType::QInt8 && weight.dim() >= 2 && weight.dim() <= 5,
             "Packing requires signed 8-bit quantized weights of rank two through five");
    int64_t K = 1;
    for (int64_t d = 1; d < weight.dim(); ++d) K *= weight.size(d);
    return {K, ((K + 31) / 32) * 32};
}

} // namespace tensorplay::quant



namespace tensorplay::cpu {
std::tuple<Tensor,Tensor> _quantized_weight_prepack_cpu(const Tensor&);
Tensor _quantized_packed_run_cpu(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,
    const std::optional<Tensor>&,const std::vector<int64_t>&,double,int64_t,const std::vector<int64_t>&,
    const std::vector<int64_t>&,const std::vector<int64_t>&,int64_t,bool,bool,int64_t,double,const std::optional<Tensor>&);
}

#ifdef USE_CUDA

namespace tensorplay::cuda {
std::tuple<Tensor,Tensor> _quantized_weight_prepack_cuda(const Tensor&);
Tensor _quantized_packed_run_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,
    const std::optional<Tensor>&,const std::vector<int64_t>&,double,int64_t,const std::vector<int64_t>&,
    const std::vector<int64_t>&,const std::vector<int64_t>&,int64_t,bool,bool,int64_t,double,const std::optional<Tensor>&);
}

#endif
