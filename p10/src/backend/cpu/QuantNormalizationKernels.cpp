#include "QuantKernels.h"
#include "QuantNormalization.h"
#include "cpu/QuantizedElementwiseKernels.h"
#include "Parallel.h"

#include <type_traits>

namespace tensorplay::cpu {
namespace {

template <typename Storage>
void normalize_codes(const Tensor& input, const Tensor& weight,
                     const Tensor& bias, const quant::NormalizationGeometry& geometry,
                     double eps, Tensor& output) {
    const Storage* X = input.data_ptr<Storage>();
    Storage* Y = output.data_ptr<Storage>();
    const float* gamma = weight.defined() ? weight.data_ptr<float>() : nullptr;
    const float* beta = bias.defined() ? bias.data_ptr<float>() : nullptr;
    const float x_scale = quantized::q_scale(input);
    const int64_t x_zero_point = quantized::q_zero_point(input);
    const float inv_y_scale = 1.0f / static_cast<float>(quantized::q_scale(output));
    const int64_t y_zero_point = quantized::q_zero_point(output);
    const int64_t N = geometry.elements;
    const int64_t channels_per_group = geometry.channels / geometry.channel_groups;
    const int64_t spatial = N / channels_per_group;

    parallel::parallel_for(0, geometry.groups, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            int64_t sum = 0;
            // Wide storage uses a floating square sum to avoid integer overflow.
            using SquareSum = std::conditional_t<sizeof(Storage) == 4, double, int64_t>;
            SquareSum square_sum = 0;
            for (int64_t k = 0; k < N; ++k) {
                const int64_t code = X[quant::normalization_offset(i, k, geometry)];
                sum += code;
                square_sum += static_cast<SquareSum>(code) * static_cast<SquareSum>(code);
            }
            const float mean_code = static_cast<float>(sum) / N;
            const float mean_div_scale = mean_code - x_zero_point;
            const float var_code = std::max(static_cast<float>(square_sum) / N
                                           - mean_code * mean_code, 0.0f);
            const float scale_div_std = x_scale / std::sqrt(var_code * x_scale * x_scale + eps);

            for (int64_t k = 0; k < N; ++k) {
                const int64_t index = geometry.affine_per_channel
                    ? (i % geometry.channel_groups) * channels_per_group + k / spatial : k;
                const float g = gamma ? gamma[index] : 1.0f;
                const float b = beta ? beta[index] : 0.0f;
                const int64_t offset = quant::normalization_offset(i, k, geometry);
                const float code = static_cast<float>(X[offset]) - x_zero_point;
                const float value = geometry.affine_per_channel
                    ? (code - mean_div_scale) * (scale_div_std * g) + b
                    : ((code - mean_div_scale) * scale_div_std) * g + b;
                Y[offset] = quant::normalization_requantize<Storage>(value, inv_y_scale, y_zero_point);
            }
        }
    });
}

Tensor normalize(const Tensor& input, const std::optional<Tensor>& weight,
                 const std::optional<Tensor>& bias,
                 const quant::NormalizationGeometry& geometry, double eps,
                 double output_scale, int64_t output_zero_point) {
    Tensor out = quant::normalization_output(input, geometry, output_scale, output_zero_point);
    if (out.numel() == 0) return out;
    const Tensor in = input.contiguous(static_cast<int64_t>(geometry.channels_last
        ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous));
    const Tensor gamma = weight.has_value() ? weight->contiguous() : Tensor();
    const Tensor beta = bias.has_value() ? bias->contiguous() : Tensor();
    if (!geometry.channels_last) {
        // Every group is one dense run, so the whole normalization can be handed
        // to the kernel that carries the vectorized byte sums.
        quantized_normalize_stub(DeviceType::CPU, in.data_ptr(),
            gamma.defined() ? gamma.data_ptr<float>() : nullptr,
            beta.defined() ? beta.data_ptr<float>() : nullptr, out.data_ptr(),
            input.dtype(), geometry.elements, geometry.groups, geometry.affine_per_channel,
            geometry.channels, geometry.channel_groups,
            static_cast<float>(quantized::q_scale(in)), quantized::q_zero_point(in),
            static_cast<float>(quantized::q_scale(out)), quantized::q_zero_point(out), eps);
        return out;
    }
    switch (input.dtype()) {
        case DType::QInt8: normalize_codes<int8_t>(in, gamma, beta, geometry, eps, out); break;
        case DType::QUInt8: normalize_codes<uint8_t>(in, gamma, beta, geometry, eps, out); break;
        case DType::QInt32: normalize_codes<int32_t>(in, gamma, beta, geometry, eps, out); break;
        default: TP_THROW(TypeError, "quantized normalization: unsupported storage type");
    }
    return out;
}

} // namespace

Tensor quantized_layer_norm_cpu(
    const Tensor& input, const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& bias,
    double eps, double output_scale, int64_t output_zero_point) {
    return normalize(input, weight, bias,
                     quant::layer_norm_geometry(input, normalized_shape, weight, bias),
                     eps, output_scale, output_zero_point);
}

Tensor quantized_group_norm_cpu(
    const Tensor& input, int64_t num_groups, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias, double eps, double output_scale,
    int64_t output_zero_point) {
    return normalize(input, weight, bias,
                     quant::group_norm_geometry(input, num_groups, weight, bias),
                     eps, output_scale, output_zero_point);
}

Tensor quantized_instance_norm_cpu(
    const Tensor& input, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias, double eps, double output_scale,
    int64_t output_zero_point) {
    return normalize(input, weight, bias,
                     quant::instance_norm_geometry(input, weight, bias),
                     eps, output_scale, output_zero_point);
}

TENSORPLAY_LIBRARY_IMPL(CPU, QuantNormalizationKernels) {
    m.impl("quantized_layer_norm", quantized_layer_norm_cpu);
    m.impl("quantized_group_norm", quantized_group_norm_cpu);
    m.impl("quantized_instance_norm", quantized_instance_norm_cpu);
}

} // namespace tensorplay::cpu
