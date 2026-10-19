#include "QuantKernels.h"
#include "QuantNormalization.h"
#include "cpu/QuantizedElementwiseKernels.h"

namespace tensorplay::cpu {
namespace {

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
    quantized_normalize_stub(DeviceType::CPU, in.data_ptr(),
        gamma.defined() ? gamma.data_ptr<float>() : nullptr,
        beta.defined() ? beta.data_ptr<float>() : nullptr, out.data_ptr(),
        input.dtype(), geometry.elements, geometry.groups, geometry.affine_per_channel,
        geometry.channels_last, geometry.channels, geometry.channel_groups,
        static_cast<float>(quantized::q_scale(in)), quantized::q_zero_point(in),
        static_cast<float>(quantized::q_scale(out)), quantized::q_zero_point(out), eps);
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
