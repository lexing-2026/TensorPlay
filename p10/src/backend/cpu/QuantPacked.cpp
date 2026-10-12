#include "QuantPacked.h"
#include "Dispatcher.h"
#include "Parallel.h"
#include <cstring>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace tensorplay::cpu {
namespace {

template<class X>
int64_t dot_scalar(const X* x, const int8_t* w, int64_t K) {
    int64_t sum = 0;
    for (int64_t k = 0; k < K; ++k) sum += static_cast<int64_t>(x[k]) * w[k];
    return sum;
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
template<class X>
__attribute__((target("avx2")))
int64_t dot_avx2(const X* x, const int8_t* w, int64_t K) {
    int64_t sum = 0, k = 0;
    alignas(32) int32_t lanes[8];
    while (k + 16 <= K) {
        const int64_t stop = std::min(k + 4096, K - K % 16);
        __m256i accum = _mm256_setzero_si256();
        for (; k < stop; k += 16) {
            const __m128i xb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + k));
            const __m128i wb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k));
            const __m256i xv = std::is_same_v<X, uint8_t> ? _mm256_cvtepu8_epi16(xb) : _mm256_cvtepi8_epi16(xb);
            const __m256i wv = _mm256_cvtepi8_epi16(wb);
            accum = _mm256_add_epi32(accum, _mm256_madd_epi16(xv, wv));
        }
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), accum);
        for (int i = 0; i < 8; ++i) sum += lanes[i];
    }
    return sum + dot_scalar(x + k, w + k, K - k);
}
#endif

template<class X>
int64_t dot(const X* x, const int8_t* w, int64_t K) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool avx2 = __builtin_cpu_supports("avx2");
    if (avx2) return dot_avx2(x, w, K);
#endif
    return dot_scalar(x, w, K);
}

template<class X>
void run(const Tensor& input, const Tensor& packed, const Tensor& offsets,
         const Tensor& scales, const Tensor& zps, const std::optional<Tensor>& bias,
         const quant::PackedGeometry& g, Tensor& out, double ys, int64_t yz,
         bool relu, bool output_float) {
    const X* xp = input.data_ptr<X>();
    const int8_t* wp = packed.data_ptr<int8_t>();
    const int64_t* col = offsets.data_ptr<int64_t>();
    const float* sc = scales.data_ptr<float>();
    const int64_t* wz = zps.data_ptr<int64_t>();
    const float* bp = bias ? bias->data_ptr<float>() : nullptr;
    const int64_t xz = quantized::q_zero_point(input);
    const double xs = quantized::q_scale(input);
    const int64_t n_per_group = g.outputs / g.groups;
    const int64_t workers = std::max<int64_t>(1, parallel::get_num_threads());
    const int64_t tile_rows = g.linear
        ? std::max<int64_t>(1, std::min<int64_t>(32, (g.rows + workers - 1) / workers))
        : 32;
    const int64_t tiles = (g.rows + tile_rows - 1) / tile_rows;
    parallel::parallel_for(0, tiles * g.groups, 1, [&](int64_t begin, int64_t end) {
        std::vector<X> columns(g.linear ? 0 : tile_rows * g.K);
        for (int64_t task = begin; task < end; ++task) {
            const int64_t group = task / tiles;
            const int64_t start = (task % tiles) * tile_rows;
            const int64_t stop = std::min(start + tile_rows, g.rows);
            for (int64_t row = start; row < stop; ++row) {
                X* column = g.linear ? nullptr : columns.data() + (row - start) * g.K;
                if (!g.linear) {
                    const int64_t batch = row / g.spatial;
                    for (int64_t k = 0; k < g.K; ++k) {
                        int64_t kp = k % g.kernel_elements, pos = row % g.spatial;
                        int64_t index = 0, multiplier = 1;
                        bool inside = true;
                        for (int64_t d = g.rank - 1; d >= 0; --d) {
                            const int64_t coordinate = (pos % g.out[d]) * g.stride[d] - g.padding[d] + (kp % g.kernel[d]) * g.dilation[d];
                            pos /= g.out[d]; kp /= g.kernel[d];
                            inside = inside && coordinate >= 0 && coordinate < g.in[d];
                            index += coordinate * multiplier; multiplier *= g.in[d];
                        }
                        const int64_t channel = group * (g.channels / g.groups) + k / g.kernel_elements;
                        column[k] = inside ? xp[(batch * g.channels + channel) * g.input_spatial + index] : static_cast<X>(xz);
                    }
                }
                const X* xr = g.linear ? xp + row * g.K : column;
                int64_t row_sum = 0;
                for (int64_t k = 0; k < g.K; ++k) row_sum += xr[k];
                for (int64_t n = group * n_per_group; n < (group + 1) * n_per_group; ++n) {
                    const int64_t acc = dot(xr, wp + n * g.padded_K, g.K) - xz * col[n] - wz[n] * row_sum + g.K * xz * wz[n];
                    double value = static_cast<double>(acc) * xs * sc[n] + (bp ? bp[n] : 0.);
                    if (relu) value = std::max(value, 0.);
                    const int64_t index = g.linear ? row * g.outputs + n : ((row / g.spatial) * g.outputs + n) * g.spatial + row % g.spatial;
                    if (output_float) out.data_ptr<float>()[index] = static_cast<float>(value);
                    else {
                        constexpr int64_t lo = std::is_same_v<X, uint8_t> ? 0 : -128;
                        const double code = std::nearbyint(value / ys) + yz;
                        out.data_ptr<X>()[index] = static_cast<X>(std::clamp(code, static_cast<double>(lo), static_cast<double>(lo + 255)));
                    }
                }
            }
        }
    });
}
} // namespace

std::tuple<Tensor, Tensor> _quantized_weight_prepack_cpu(const Tensor& weight) {
    const auto [K, padded_K] = quant::packed_weight_shape(weight);
    const Tensor w = weight.contiguous();
    Tensor packed = Tensor::zeros({weight.size(0), padded_K}, DType::Int8, weight.device());
    Tensor offsets = Tensor::zeros({weight.size(0)}, DType::Int64, weight.device());
    parallel::parallel_for(0, weight.size(0), 16, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            if (K) std::memcpy(packed.data_ptr<int8_t>() + n * padded_K, w.data_ptr<int8_t>() + n * K, K);
            int64_t sum = 0;
            for (int64_t k = 0; k < K; ++k) sum += w.data_ptr<int8_t>()[n * K + k];
            offsets.data_ptr<int64_t>()[n] = sum;
        }
    });
    return {packed, offsets};
}

Tensor _quantized_packed_run_cpu(
    const Tensor& input, const Tensor& packed, const Tensor& offsets,
    const Tensor& scales, const Tensor& zero_points, const std::optional<Tensor>& bias,
    const std::vector<int64_t>& weight_sizes, double output_scale, int64_t output_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups, bool relu, bool output_float) {
    const auto g = quant::packed_geometry(input, packed, offsets, scales, zero_points, bias,
        weight_sizes, stride, padding, dilation, groups, output_scale, output_zero_point, output_float);
    const Tensor x = input.contiguous();
    const DType storage = output_float ? DType::Float32 : (input.dtype() == DType::QUInt8 ? DType::UInt8 : DType::Int8);
    Tensor out = Tensor::empty(g.shape, storage, input.device());
    if (input.dtype() == DType::QUInt8) run<uint8_t>(x, packed, offsets, scales, zero_points, bias, g, out, output_scale, output_zero_point, relu, output_float);
    else run<int8_t>(x, packed, offsets, scales, zero_points, bias, g, out, output_scale, output_zero_point, relu, output_float);
    if (output_float) return out;
    return quantized::make_qtensor(out, make_per_tensor_affine_quantizer(output_scale, output_zero_point, input.dtype()), input.dtype());
}

Tensor quantized_batch_norm_relu_cpu(
    const Tensor& input, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias,
    const Tensor& mean, const Tensor& var, double eps, double output_scale, int64_t output_zero_point) {
    quant::check_bn_relu(input, weight, bias, mean, var, eps, output_scale, output_zero_point);
    const Tensor x = input.contiguous(), m = mean.contiguous(), v = var.contiguous();
    const Tensor w = weight ? weight->contiguous() : Tensor(), b = bias ? bias->contiguous() : Tensor();
    const bool unsigned_input = input.dtype() == DType::QUInt8;
    Tensor out = Tensor::empty(input.shape(), unsigned_input ? DType::UInt8 : DType::Int8, input.device());
    const double xs = quantized::q_scale(input);
    const int64_t xz = quantized::q_zero_point(input), C = input.size(1);
    int64_t spatial = 1;
    for (int64_t d = 2; d < input.dim(); ++d) spatial *= input.size(d);
    std::vector<float> alpha(C), beta(C);
    for (int64_t c = 0; c < C; ++c) {
        const float inv = 1.f / std::sqrt(v.data_ptr<float>()[c] + static_cast<float>(eps));
        const float gamma = weight ? w.data_ptr<float>()[c] : 1.f;
        alpha[c] = inv * gamma * (xs / output_scale);
        beta[c] = ((bias ? b.data_ptr<float>()[c] : 0.f) - m.data_ptr<float>()[c] * inv * gamma) / output_scale;
    }
    parallel::parallel_for(0, input.numel(), 32768, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t c = (i / spatial) % C;
            const int code = unsigned_input ? x.data_ptr<uint8_t>()[i] : x.data_ptr<int8_t>()[i];
            const float value = std::fma(alpha[c], static_cast<float>(code - xz), beta[c]);
            const double q = std::clamp(std::nearbyint(static_cast<double>(value)) + output_zero_point,
                                        static_cast<double>(output_zero_point), unsigned_input ? 255. : 127.);
            if (unsigned_input) out.data_ptr<uint8_t>()[i] = static_cast<uint8_t>(q);
            else out.data_ptr<int8_t>()[i] = static_cast<int8_t>(q);
        }
    });
    return quantized::make_qtensor(out, make_per_tensor_affine_quantizer(output_scale, output_zero_point, input.dtype()), input.dtype());
}

TENSORPLAY_LIBRARY_IMPL(CPU, QuantPacked) {
    m.impl("quantized_batch_norm_relu", quantized_batch_norm_relu_cpu);
    m.impl("_quantized_weight_prepack", _quantized_weight_prepack_cpu);
    m.impl("_quantized_packed_run", _quantized_packed_run_cpu);
}
} // namespace tensorplay::cpu
