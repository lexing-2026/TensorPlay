#include "QuantKernels.h"
#include "QuantNormalization.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>
#include <type_traits>

namespace tensorplay::cuda {
namespace {

constexpr int kNormThreads = 256;

template <typename Storage>
__global__ void normalize_codes(
    const Storage* X, Storage* Y, const float* gamma, const float* beta,
    quant::NormalizationGeometry geometry, float x_scale, int64_t x_zero_point,
    float inv_y_scale, int64_t y_zero_point, double eps) {
    using SquareSum = std::conditional_t<sizeof(Storage) == 4, double, long long>;
    __shared__ long long sums[kNormThreads];
    __shared__ SquareSum squares[kNormThreads];
    __shared__ float mean_div_scale;
    __shared__ float scale_div_std;
    const int tid = threadIdx.x;
    const int64_t N = geometry.elements;
    const int64_t channels_per_group = geometry.channels / geometry.channel_groups;
    const int64_t spatial = N / channels_per_group;

    // A bounded grid also supports more groups than one launch's grid limit.
    for (int64_t i = blockIdx.x; i < geometry.groups; i += gridDim.x) {
        long long sum = 0;
        SquareSum square_sum = 0;
        for (int64_t k = tid; k < N; k += blockDim.x) {
            const long long code = X[quant::normalization_offset(i, k, geometry)];
            sum += code;
            square_sum += static_cast<SquareSum>(code) * static_cast<SquareSum>(code);
        }
        sums[tid] = sum;
        squares[tid] = square_sum;
        __syncthreads();
        for (int offset = kNormThreads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                sums[tid] += sums[tid + offset];
                squares[tid] += squares[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const float mean_code = static_cast<float>(sums[0]) / N;
            const float var_code = fmaxf(static_cast<float>(squares[0]) / N
                                        - mean_code * mean_code, 0.0f);
            mean_div_scale = mean_code - x_zero_point;
            scale_div_std = x_scale / ::sqrt(var_code * x_scale * x_scale + eps);
        }
        __syncthreads();
        for (int64_t k = tid; k < N; k += blockDim.x) {
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
        __syncthreads();
    }
}

template <typename Storage>
void launch(const Tensor& input, const Tensor& weight, const Tensor& bias,
            const quant::NormalizationGeometry& geometry, double eps, Tensor& out) {
    const int blocks = static_cast<int>(std::min<int64_t>(geometry.groups, 65535));
    normalize_codes<Storage><<<blocks, kNormThreads, 0, getCurrentCUDAStream().stream()>>>(
        input.data_ptr<Storage>(), out.data_ptr<Storage>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr, geometry,
        static_cast<float>(quantized::q_scale(input)), quantized::q_zero_point(input),
        1.0f / static_cast<float>(quantized::q_scale(out)), quantized::q_zero_point(out), eps);
    checkCuda(cudaGetLastError(), "CUDA quantized normalization kernel");
}

Tensor normalize(const Tensor& input, const std::optional<Tensor>& weight,
                 const std::optional<Tensor>& bias,
                 const quant::NormalizationGeometry& geometry, double eps,
                 double output_scale, int64_t output_zero_point) {
    CUDAGuard guard(input.device());
    Tensor out = quant::normalization_output(input, geometry, output_scale, output_zero_point);
    if (out.numel() == 0) return out;
    const Tensor in = input.contiguous(static_cast<int64_t>(geometry.channels_last
        ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous));
    const Tensor gamma = weight.has_value() ? weight->contiguous() : Tensor();
    const Tensor beta = bias.has_value() ? bias->contiguous() : Tensor();
    switch (input.dtype()) {
        case DType::QInt8: launch<int8_t>(in, gamma, beta, geometry, eps, out); break;
        case DType::QUInt8: launch<uint8_t>(in, gamma, beta, geometry, eps, out); break;
        case DType::QInt32: launch<int32_t>(in, gamma, beta, geometry, eps, out); break;
        default: TP_THROW(TypeError, "quantized normalization: unsupported storage type");
    }
    return out;
}

} // namespace

Tensor quantized_layer_norm_cuda(
    const Tensor& input, const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& bias,
    double eps, double output_scale, int64_t output_zero_point) {
    return normalize(input, weight, bias,
                     quant::layer_norm_geometry(input, normalized_shape, weight, bias),
                     eps, output_scale, output_zero_point);
}

Tensor quantized_group_norm_cuda(
    const Tensor& input, int64_t num_groups, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias, double eps, double output_scale,
    int64_t output_zero_point) {
    return normalize(input, weight, bias,
                     quant::group_norm_geometry(input, num_groups, weight, bias),
                     eps, output_scale, output_zero_point);
}

Tensor quantized_instance_norm_cuda(
    const Tensor& input, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias, double eps, double output_scale,
    int64_t output_zero_point) {
    return normalize(input, weight, bias,
                     quant::instance_norm_geometry(input, weight, bias),
                     eps, output_scale, output_zero_point);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantNormalizationKernels) {
    m.impl("quantized_layer_norm", quantized_layer_norm_cuda);
    m.impl("quantized_group_norm", quantized_group_norm_cuda);
    m.impl("quantized_instance_norm", quantized_instance_norm_cuda);
}

} // namespace tensorplay::cuda
