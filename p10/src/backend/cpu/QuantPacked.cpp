#include "QuantPacked.h"
#include "QuantNative.h"
#include "Dispatcher.h"
#include "Parallel.h"
#include "QuantElementwise.h"
#include "cpu/QuantizedElementwiseKernels.h"
#include <array>
#include <cfenv>
#include "OneDNNContext.h"
#include <map>
#include <cstring>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace tensorplay::quant {
namespace {
struct ActivationTable {
    bool valid = false;
    Activation fn{0};
    float scale = 0, inverse_scale = 0;
    int64_t zero_point = 0, output_zero_point = 0;
    DType dtype = DType::QInt8;
    int rounding = 0;
    std::array<uint8_t, 256> codes{};
    std::array<float, 256> wide_codes{};
};

const ActivationTable& activation_table(const Tensor& input,
        float inverse_scale, int64_t output_zero_point, Activation fn) {
    // The finite input code domain admits exact evaluation once per grid.
    // Each calling thread owns a bounded cache; parallel readers share a
    // completed immutable table for the duration of one invocation.
    thread_local std::array<ActivationTable, 16> cache;
    thread_local size_t next = 0;
    const float scale = static_cast<float>(quantized::q_scale(input));
    const int64_t zp = quantized::q_zero_point(input);
    const int rounding = std::fegetround();
    for (auto& entry : cache) {
        if (entry.valid && entry.fn == fn && entry.scale == scale &&
            entry.inverse_scale == inverse_scale && entry.zero_point == zp &&
            entry.output_zero_point == output_zero_point && entry.dtype == input.dtype() &&
            entry.rounding == rounding) return entry;
    }
    auto& entry = cache[next++ % cache.size()];
    entry.valid = true; entry.fn = fn; entry.scale = scale;
    entry.inverse_scale = inverse_scale; entry.zero_point = zp;
    entry.output_zero_point = output_zero_point; entry.dtype = input.dtype();
    entry.rounding = rounding;
    for (int code = 0; code < 256; ++code) {
        const int value = input.dtype() == DType::QInt8 ? static_cast<int8_t>(code) : code;
        const float y = fn(static_cast<float>(value - zp) * scale);
        entry.codes[code] = input.dtype() == DType::QInt8
            ? static_cast<uint8_t>(encode<int8_t>(y, inverse_scale, output_zero_point))
            : encode<uint8_t>(y, inverse_scale, output_zero_point);
        entry.wide_codes[code] = entry.codes[code];
    }
    return entry;
}
}

Tensor activation_cpu(const Tensor& input, double scale, int64_t zp, Activation fn) {
    auto grid = make_per_tensor_affine_quantizer(scale, zp, input.dtype());
    const float input_scale = static_cast<float>(quantized::q_scale(input));
    const int64_t input_zp = quantized::q_zero_point(input);
    const float inverse_scale = 1.f / static_cast<float>(scale);
    const Tensor x = input.contiguous();
    Tensor out = Tensor::empty(input.shape(), input.dtype(), input.device());
    if (input.dtype() == DType::QInt8 || input.dtype() == DType::QUInt8) {
        const auto& cached = activation_table(input, inverse_scale, zp, fn);
        const auto* src = static_cast<const uint8_t*>(x.data_ptr());
        auto* dst = static_cast<uint8_t*>(out.data_ptr());
        parallel::parallel_for(0, input.numel(), 32768, [&](int64_t begin, int64_t end) {
            cpu::quantized_lookup_stub(DeviceType::CPU, src + begin, dst + begin,
                                        end - begin, cached.wide_codes.data());
        });
    } else if (input.dtype() == DType::QInt32) {
        const auto* src = x.data_ptr<int32_t>();
        auto* dst = out.data_ptr<int32_t>();
        parallel::parallel_for(0, input.numel(), 32768, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i)
                dst[i] = encode<int32_t>(fn(static_cast<float>(static_cast<int64_t>(src[i]) - input_zp) * input_scale), inverse_scale, zp);
        });
    } else TP_THROW(TypeError, "Activation requires quantized integer input");
    out.impl()->set_quantizer(std::move(grid));
    return out;
}

Tensor rectify_cpu(const Tensor& input, bool clamp_six) {
    const double scale = quantized::q_scale(input);
    const int64_t zp = quantized::q_zero_point(input);
    const Tensor x = input.contiguous();
    Tensor out = Tensor::empty(input.shape(), input.dtype(), input.device());
    auto apply = [&]<class T>() {
        const T hi = clamp_six ? encode<T>(6.f, 1.f / static_cast<float>(scale), zp) : std::numeric_limits<T>::max();
        const auto* src = x.data_ptr<T>(); auto* dst = out.data_ptr<T>();
        parallel::parallel_for(0, input.numel(), 32768, [&](int64_t begin, int64_t end) {
            cpu::quantized_clip_stub(DeviceType::CPU, src + begin, dst + begin,
                                      end - begin, input.dtype(), zp, hi);
        });
    };
    switch (input.dtype()) {
        case DType::QInt8: apply.template operator()<int8_t>(); break;
        case DType::QUInt8: apply.template operator()<uint8_t>(); break;
        case DType::QInt32: apply.template operator()<int32_t>(); break;
        default: TP_THROW(TypeError, "Rectification requires quantized integer input");
    }
    out.impl()->set_quantizer(input.impl()->quantizer());
    return out;
}
Tensor relu_cpu(const Tensor& input) { return rectify_cpu(input, false); }
Tensor relu6_cpu(const Tensor& input) { return rectify_cpu(input, true); }
} // namespace tensorplay::quant

namespace tensorplay::cpu {
DEFINE_DISPATCH(quantized_int4_pack_stub);
DEFINE_DISPATCH(quantized_int4_mm_stub);
DEFINE_DISPATCH(quantized_batch_norm_stub);
DEFINE_DISPATCH(quantized_bilinear_stub);
DEFINE_DISPATCH(quantized_clip_stub);
DEFINE_DISPATCH(quantized_lookup_stub);
DEFINE_DISPATCH(quantized_binary_stub);
DEFINE_DISPATCH(quantized_embedding_stub);
DEFINE_DISPATCH(quantized_fp16_linear_stub);
namespace {

template<class X>
int64_t dot_scalar(const X* x, const int8_t* w, int64_t K) {
    int64_t sum = 0;
    for (int64_t k = 0; k < K; ++k) sum += static_cast<int64_t>(x[k]) * w[k];
    return sum;
}

#ifdef USE_ONEDNN
struct IntegerProduct {
    dnnl::memory::desc src, weight, dst;
    dnnl::matmul product;
    dnnl::memory::desc scratch_desc;
    std::vector<uint8_t> scratch;
    IntegerProduct(const std::array<int64_t, 6>& key) {
        using DT = dnnl::memory::data_type;
        src = dnnl::memory::desc({key[0], key[2]}, key[4] ? DT::u8 : DT::s8, {key[2], 1});
        weight = dnnl::memory::desc({key[2], key[1]}, DT::s8, {1, key[3]});
        dst = dnnl::memory::desc({key[0], key[1]}, DT::s32, {key[1], 1});
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        const auto desc = dnnl::matmul::primitive_desc(OneDNNContext::get_engine(), src, weight, dst, attr);
        product = dnnl::matmul(desc);
        scratch_desc = desc.scratchpad_desc();
        scratch.resize(scratch_desc.get_size());
    }
};

template<class X>
bool linear_integer_product(const X* x, const int8_t* w, const int64_t* offsets,
        const float* scales, const int64_t* zps, const float* bias,
        const quant::PackedGeometry& g, Tensor& out, double xs, int64_t xz,
        double ys, int64_t yz, bool relu, bool output_float) {
    // The raw product fits signed 32-bit storage for this K bound. Affine
    // corrections widen to 64 bits, including arbitrary weight zero points.
    if (!OneDNNContext::is_enabled() || !g.linear || g.K == 0 || g.K > 65536 ||
        g.rows < 8 || g.rows * g.outputs * g.K < 1048576) return false;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (!__builtin_cpu_supports("avxvnni") && !__builtin_cpu_supports("avx512vnni")) return false;
#endif
    thread_local std::map<std::array<int64_t, 6>, std::unique_ptr<IntegerProduct>> cache;
    const std::array<int64_t, 6> key{g.rows, g.outputs, g.K, g.padded_K,
        std::is_same_v<X, uint8_t>, parallel::get_num_threads()};
    auto it = cache.find(key);
    if (it == cache.end()) {
        if (cache.size() >= 32) cache.clear();
        try { it = cache.emplace(key, std::make_unique<IntegerProduct>(key)).first; }
        catch (const dnnl::error&) { return false; }
    }
    auto& kernel = *it->second;
    Tensor acc = Tensor::empty({g.rows, g.outputs}, DType::Int32, out.device());
    auto& engine = OneDNNContext::get_engine();
    auto& stream = OneDNNContext::get_stream();
    kernel.product.execute(stream, {
        {DNNL_ARG_SRC, dnnl::memory(kernel.src, engine, const_cast<X*>(x))},
        {DNNL_ARG_WEIGHTS, dnnl::memory(kernel.weight, engine, const_cast<int8_t*>(w))},
        {DNNL_ARG_DST, dnnl::memory(kernel.dst, engine, acc.data_ptr())},
        {DNNL_ARG_SCRATCHPAD, dnnl::memory(kernel.scratch_desc, engine, kernel.scratch.data())}});
    stream.wait();
    const auto* raw = acc.data_ptr<int32_t>();
    parallel::parallel_for(0, g.rows, 8, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            int64_t sum = 0;
            for (int64_t k = 0; k < g.K; ++k) sum += x[row * g.K + k];
            for (int64_t n = 0; n < g.outputs; ++n) {
                const int64_t index = row * g.outputs + n;
                const int64_t value_i = static_cast<int64_t>(raw[index]) - xz * offsets[n] - zps[n] * sum + g.K * xz * zps[n];
                double value = static_cast<double>(value_i) * xs * scales[n] + (bias ? bias[n] : 0.);
                value = quant::packed_activation(value, index, g);
                if (relu) value = std::max(value, 0.);
                if (output_float) out.data_ptr<float>()[index] = static_cast<float>(value);
                else {
                    constexpr int64_t lo = std::is_same_v<X, uint8_t> ? 0 : -128;
                    out.data_ptr<X>()[index] = static_cast<X>(std::clamp(std::nearbyint(value / ys) + yz, static_cast<double>(lo), static_cast<double>(lo + 255)));
                }
            }
        }
    });
    return true;
}
#endif

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

template<class X, int Rows>
__attribute__((target("avx2,avxvnni")))
void block_vnni(const X* x, const int8_t* w, int64_t K, int64_t padded_K,
                int64_t columns, int64_t (&totals)[Rows][4]) {
    int64_t k = 0;
    for (int r = 0; r < Rows; ++r) for (int c = 0; c < 4; ++c) totals[r][c] = 0;
    // Each lane accumulates at most 4096 products of unsigned and signed
    // bytes before widening, including the extrema of both storage ranges.
    while (k + 32 <= K) {
        const int64_t stop = std::min(k + 16384, K - K % 32);
        __m256i accum[Rows][4];
        for (int r = 0; r < Rows; ++r) for (int c = 0; c < 4; ++c) accum[r][c] = _mm256_setzero_si256();
        for (; k < stop; k += 32) {
            __m256i weights[4];
            #pragma GCC unroll 4
            for (int c = 0; c < 4; ++c)
                weights[c] = c < columns ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + c * padded_K + k)) : _mm256_setzero_si256();
            for (int r = 0; r < Rows; ++r) {
                __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + r * K + k));
                if constexpr (std::is_same_v<X, int8_t>) values = _mm256_xor_si256(values, _mm256_set1_epi8(-128));
                #pragma GCC unroll 4
                for (int c = 0; c < 4; ++c)
                    accum[r][c] = _mm256_dpbusd_epi32(accum[r][c], values, weights[c]);
            }
        }
        alignas(32) int32_t lanes[8];
        for (int r = 0; r < Rows; ++r) for (int c = 0; c < columns; ++c) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), accum[r][c]);
            for (int lane = 0; lane < 8; ++lane) totals[r][c] += lanes[lane];
        }
    }
    for (; k < K; ++k) for (int r = 0; r < Rows; ++r) for (int c = 0; c < columns; ++c) {
        const int value = static_cast<int>(x[r * K + k]) + (std::is_same_v<X, int8_t> ? 128 : 0);
        totals[r][c] += static_cast<int64_t>(value) * w[c * padded_K + k];
    }
}

template<class X>
void linear_vnni(const X* x, const int8_t* w, const int64_t* offsets,
        const float* scales, const int64_t* zps, const float* bias,
        const quant::PackedGeometry& g, Tensor& out, double xs, int64_t xz,
        double ys, int64_t yz, bool relu, bool output_float) {
    const int64_t row_tiles = (g.rows + 1) / 2, column_tiles = (g.outputs + 63) / 64;
    auto store = [&](int64_t row, int64_t n, int64_t dot, int64_t sum) {
        const int64_t shift = std::is_same_v<X, int8_t> ? 128 : 0;
        const int64_t acc = dot - (xz + shift) * offsets[n] - zps[n] * sum + g.K * xz * zps[n];
        double value = static_cast<double>(acc) * xs * scales[n] + (bias ? bias[n] : 0.);
        const int64_t index = row * g.outputs + n;
        value = quant::packed_activation(value, index, g);
        if (relu) value = std::max(value, 0.);
        if (output_float) out.data_ptr<float>()[index] = static_cast<float>(value);
        else {
            constexpr int64_t lo = std::is_same_v<X, uint8_t> ? 0 : -128;
            out.data_ptr<X>()[index] = static_cast<X>(std::clamp(std::nearbyint(value / ys) + yz, static_cast<double>(lo), static_cast<double>(lo + 255)));
        }
    };
    const int64_t tasks = row_tiles * column_tiles;
    const int64_t grain = g.rows * g.outputs * g.K < 1048576 ? tasks : 1;
    parallel::parallel_for(0, tasks, grain, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t row = (task / column_tiles) * 2;
            const int64_t n0 = (task % column_tiles) * 64, n1 = std::min(n0 + 64, g.outputs);
            const int64_t rows = std::min<int64_t>(2, g.rows - row);
            int64_t sums[4]{};
            for (int r = 0; r < rows; ++r) for (int64_t k = 0; k < g.K; ++k) sums[r] += x[(row + r) * g.K + k];
            for (int64_t n = n0; n < n1; n += 4) {
                const int64_t columns = std::min<int64_t>(4, n1 - n);
                if (rows == 2) {
                    int64_t totals[2][4];
                    block_vnni<X,2>(x + row * g.K, w + n * g.padded_K, g.K, g.padded_K, columns, totals);
                    for (int r = 0; r < 2; ++r) for (int c = 0; c < columns; ++c) store(row + r, n + c, totals[r][c], sums[r]);
                } else {
                    for (int r = 0; r < rows; ++r) {
                        int64_t totals[1][4];
                        block_vnni<X,1>(x + (row + r) * g.K, w + n * g.padded_K, g.K, g.padded_K, columns, totals);
                        for (int c = 0; c < columns; ++c) store(row + r, n + c, totals[0][c], sums[r]);
                    }
                }
            }
        }
    });
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
#ifdef USE_ONEDNN
    if (linear_integer_product(xp, wp, col, sc, wz, bp, g, out, xs, xz, ys, yz, relu, output_float)) return;
#endif
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool vnni = __builtin_cpu_supports("avxvnni");
    if (g.linear && vnni) {
        linear_vnni(xp, wp, col, sc, wz, bp, g, out, xs, xz, ys, yz, relu, output_float);
        return;
    }
#endif
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
                    const int64_t index = g.linear ? row * g.outputs + n : ((row / g.spatial) * g.outputs + n) * g.spatial + row % g.spatial;
                    value = quant::packed_activation(value, index, g);
                    if (relu) value = std::max(value, 0.);
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
    const std::vector<int64_t>& dilation, int64_t groups, bool relu, bool output_float,
    int64_t activation, double negative_slope, const std::optional<Tensor>& residual) {
    auto g = quant::packed_geometry(input, packed, offsets, scales, zero_points, bias,
        weight_sizes, stride, padding, dilation, groups, output_scale, output_zero_point, output_float);
    TP_CHECK(activation >= 0 && activation <= 2 && std::isfinite(negative_slope), "Invalid packed activation parameters");
    TP_CHECK(activation == 0 || g.linear, "Packed activation requires linear weights");
    g.activation = activation; g.negative_slope = negative_slope;
    Tensor residual_codes;
    if (residual) {
        TP_CHECK(!g.linear && g.rank == 2 && !output_float && residual->shape() == g.shape &&
            residual->device() == input.device() && residual->dtype() == input.dtype(),
            "Convolution residual must match the output shape, device and quantized dtype");
        residual_codes = residual->contiguous();
        g.residual = residual_codes.data_ptr();
        g.residual_scale = quantized::q_scale(*residual);
        g.residual_zero_point = quantized::q_zero_point(*residual);
        g.residual_unsigned = residual->dtype() == DType::QUInt8;
    }
    const Tensor x = input.contiguous();
    const DType storage = output_float ? DType::Float32 : (input.dtype() == DType::QUInt8 ? DType::UInt8 : DType::Int8);
    Tensor out = Tensor::empty(g.shape, storage, input.device());
    if (input.dtype() == DType::QUInt8) run<uint8_t>(x, packed, offsets, scales, zero_points, bias, g, out, output_scale, output_zero_point, relu, output_float);
    else run<int8_t>(x, packed, offsets, scales, zero_points, bias, g, out, output_scale, output_zero_point, relu, output_float);
    if (output_float) return out;
    Tensor result = quantized::make_qtensor(out, make_per_tensor_affine_quantizer(output_scale, output_zero_point, input.dtype()), input.dtype());
    if (residual && result.numel()) {
        Tensor destination = residual->contiguous(static_cast<int64_t>(MemoryFormat::ChannelsLast));
        destination.impl()->set_quantizer(result.impl()->quantizer());
        destination.copy_(result);
        return destination;
    }
    return result;
}

Tensor quantized_batch_norm_relu_cpu(
    const Tensor& input, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias,
    const Tensor& mean, const Tensor& var, double eps, double output_scale, int64_t output_zero_point) {
    return quant::batch_norm_cpu(input, weight, bias, mean, var,
        eps, output_scale, output_zero_point, true);
}

TENSORPLAY_LIBRARY_IMPL(CPU, QuantPacked) {
    m.impl("quantized_batch_norm_relu", quantized_batch_norm_relu_cpu);
    m.impl("_quantized_weight_prepack", _quantized_weight_prepack_cpu);
    m.impl("_quantized_packed_run", _quantized_packed_run_cpu);
}
} // namespace tensorplay::cpu
