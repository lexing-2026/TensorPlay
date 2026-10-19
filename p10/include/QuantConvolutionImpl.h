#pragma once

namespace tensorplay::quant::TP_QN_DETAIL {
struct ConvGeometry {
    int64_t in[3]{1,1,1}, out[3]{1,1,1}, kernel[3]{1,1,1}, stride[3]{1,1,1}, pad[3]{}, dilation[3]{1,1,1};
    int64_t input_spatial = 1, output_spatial = 1, kernel_spatial = 1, channels, outputs, groups, xz, wz, yz;
    float xs, ws, inverse; bool transposed, relu, output_float, channel_grid;
};

template<class X, class Y>
struct ConvCodes {
    const X* x; const int8_t* w; const float *scales, *bias; const int64_t* zero_points; Y* y; ConvGeometry g;
    const int32_t* taps;
    TP_QN_DEVICE void operator()(int64_t i) const {
        const int64_t out_pos = i % g.output_spatial, oc = i / g.output_spatial % g.outputs;
        const int64_t batch = i / (g.outputs * g.output_spatial), group = oc / (g.outputs / g.groups);
        const int64_t local_oc = oc % (g.outputs / g.groups), inputs_per_group = g.channels / g.groups;
        const int64_t wz = g.channel_grid ? zero_points[oc] : g.wz;
        int64_t sum = 0;
        for (int64_t kp = 0; kp < g.kernel_spatial; ++kp) {
            int64_t pos;
            if (taps) {
                const int32_t entry = taps[out_pos * g.kernel_spatial + kp];
                if (entry < 0) continue;
                pos = entry;
            } else {
                int64_t p = out_pos, k = kp, multiplier = 1; bool inside = true; pos = 0;
                for (int d = 2; d >= 0; --d) {
                    const int64_t out_coord = p % g.out[d], kernel_coord = k % g.kernel[d]; p /= g.out[d]; k /= g.kernel[d];
                    int64_t coordinate;
                    if (g.transposed) {
                        const int64_t numerator = out_coord + g.pad[d] - kernel_coord * g.dilation[d];
                        inside = inside && numerator % g.stride[d] == 0;
                        coordinate = numerator / g.stride[d];
                    } else coordinate = out_coord * g.stride[d] - g.pad[d] + kernel_coord * g.dilation[d];
                    inside = inside && coordinate >= 0 && coordinate < g.in[d];
                    pos += coordinate * multiplier; multiplier *= g.in[d];
                }
                if (!inside) continue;
            }
            for (int64_t c = 0; c < inputs_per_group; ++c) {
                const int64_t ic = group * inputs_per_group + c;
                const int64_t weight_index = g.transposed
                    ? (ic * (g.outputs / g.groups) + local_oc) * g.kernel_spatial + kp
                    : (oc * inputs_per_group + c) * g.kernel_spatial + kp;
                sum += (static_cast<int64_t>(x[(batch * g.channels + ic) * g.input_spatial + pos]) - g.xz)
                     * (static_cast<int64_t>(w[weight_index]) - wz);
            }
        }
        float value = static_cast<float>(sum) * g.xs * (g.channel_grid ? scales[oc] : g.ws);
        if (bias) value += bias[oc];
        if (g.relu) value = ::fmaxf(value, 0.f);
        if constexpr (std::is_same_v<Y,float>) y[i] = value;
        else y[i] = encode<Y>(value, g.inverse, g.yz);
    }
};

#ifndef TP_QUANT_CUDA
// A tap answers to one input address fixed by the output location and the tap,
// never by the batch, the group or the output channel, so a table row serves
// every element that lands on that location. The build is only worth its own
// coordinate work where several of them do, and is dropped for shapes whose
// location and tap counts would outgrow a sane scratch size.
constexpr int64_t kConvGatherLimit = 1 << 20;

// One thread-pool chunk should hold roughly this many table entries.
constexpr int64_t kConvTapChunkEntries = 4096;

// An axis contributes an offset set only by its own output coordinate and its
// own kernel position, so resolving all of those once leaves each tap a sum of
// three loads instead of a fresh coordinate derivation. A coordinate outside the
// input keeps -1, a value no in-range offset can take because every one of them
// is non-negative, and a tap lies inside exactly when none of its three axes do.
std::vector<int32_t> build_conv_taps(const ConvGeometry& g, int64_t reuse) {
    std::vector<int32_t> table;
    if (reuse <= 1 || g.output_spatial * g.kernel_spatial > kConvGatherLimit ||
        g.input_spatial > (1LL << 31) - 1) return table;
    std::vector<int32_t> axis[3];
    int64_t multiplier = 1;
    for (int d = 2; d >= 0; --d) {
        axis[d].resize(g.out[d] * g.kernel[d]);
        for (int64_t coordinate = 0; coordinate < g.out[d]; ++coordinate)
            for (int64_t place = 0; place < g.kernel[d]; ++place) {
                int64_t address;
                if (g.transposed) {
                    const int64_t numerator = coordinate + g.pad[d] - place * g.dilation[d];
                    address = numerator % g.stride[d] == 0 ? numerator / g.stride[d] : -1;
                } else address = coordinate * g.stride[d] - g.pad[d] + place * g.dilation[d];
                axis[d][coordinate * g.kernel[d] + place] =
                    address >= 0 && address < g.in[d] ? static_cast<int32_t>(address * multiplier) : -1;
            }
        multiplier *= g.in[d];
    }
    table.resize(g.output_spatial * g.kernel_spatial);
    // A chunk spans locations, each holding one entry per tap, so a small layout
    // stays on one thread instead of splitting into rows of a handful of entries.
    parallel::parallel_for(0, g.output_spatial,
        std::max<int64_t>(1, kConvTapChunkEntries / g.kernel_spatial), [&](int64_t first, int64_t last) {
        for (int64_t out_pos = first; out_pos < last; ++out_pos) {
            int64_t p = out_pos;
            const int32_t* row[3];
            for (int d = 2; d >= 0; --d) {
                const int64_t coordinate = p % g.out[d]; p /= g.out[d];
                row[d] = axis[d].data() + coordinate * g.kernel[d];
            }
            int32_t* write = table.data() + out_pos * g.kernel_spatial;
            for (int64_t a = 0; a < g.kernel[0]; ++a) {
                const int32_t outer = row[0][a];
                for (int64_t b = 0; b < g.kernel[1]; ++b) {
                    const int32_t middle = row[1][b];
                    const int32_t pair = (outer | middle) < 0 ? -1 : outer + middle;
                    for (int64_t c = 0; c < g.kernel[2]; ++c) {
                        const int32_t inner = row[2][c];
                        *write++ = (pair | inner) < 0 ? -1 : pair + inner;
                    }
                }
            }
        }
    });
    return table;
}
#endif

Tensor convolution(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias,
        double scale, int64_t zp, const std::vector<int64_t>& strides,
        const std::vector<int64_t>& pads, const std::vector<int64_t>& dilations,
        int64_t groups, bool transposed, const std::vector<int64_t>& output_pads, bool relu, bool output_float) {
    Guard guard(input); validate(input);
    const int rank = weight.dim() - 2;
    TP_CHECK(rank >= 1 && rank <= 3 && input.dim() == rank + 2 && weight.dtype() == DType::QInt8
        && input.device() == weight.device(), "Integer convolution requires compatible spatial input and signed weights");
    TP_CHECK(groups > 0 && input.size(1) % groups == 0, "Input channels must be divisible by positive groups");
    ConvGeometry g{}; g.channels = input.size(1); g.groups = groups;
    g.outputs = transposed ? weight.size(1) * groups : weight.size(0);
    TP_CHECK(g.outputs > 0 && g.outputs % groups == 0 && g.channels > 0, "Convolution channel dimensions must be positive and divisible by groups");
    TP_CHECK((transposed && weight.size(0) == g.channels) || (!transposed && weight.size(1) * groups == g.channels), "Convolution weight channels do not match the input");
    g.transposed = transposed; g.relu = relu; g.output_float = output_float;
    g.xs = static_cast<float>(quantized::q_scale(input)); g.xz = quantized::q_zero_point(input);
    g.inverse = 1.f / static_cast<float>(scale); g.yz = zp;
    Tensor scales, zps;
    const auto qscheme = weight.impl()->quantizer()->qscheme();
    if (qscheme == kPerTensorAffine) { g.ws = quantized::q_scale(weight); g.wz = quantized::q_zero_point(weight); }
    else {
        TP_CHECK(!transposed && qscheme == kPerChannelAffine && quantized::q_per_channel_axis(weight) == 0,
            "Convolution channel grids require axis zero; transposed weights require a tensor grid");
        g.channel_grid = true;
        scales = quantized::q_per_channel_scales(weight).to(DType::Float32).contiguous();
        zps = quantized::q_per_channel_zero_points(weight).to(DType::Int64).contiguous();
    }
    const auto stride = parameters(strides, rank, 1), pad = parameters(pads, rank, 0);
    const auto dilation = parameters(dilations, rank, 1), output_pad = parameters(output_pads, rank, 0);
    auto shape = static_cast<std::vector<int64_t>>(input.shape()); shape[1] = g.outputs;
    for (int d = 0; d < rank; ++d) {
        const int j = 3 - rank + d;
        g.in[j] = input.size(d + 2); g.kernel[j] = weight.size(d + 2);
        TP_CHECK(g.in[j] > 0 && g.kernel[j] > 0 && stride[d] > 0 && dilation[d] > 0 && pad[d] >= 0 && output_pad[d] >= 0,
                 "Convolution geometry requires positive spatial dimensions, stride and dilation, and nonnegative padding");
        TP_CHECK(!transposed || output_pad[d] < stride[d] || output_pad[d] < dilation[d], "Output padding must be smaller than stride or dilation");
        TP_CHECK(transposed || output_pad[d] == 0, "Ordinary convolution does not accept output padding");
        g.stride[j] = stride[d]; g.pad[j] = pad[d]; g.dilation[j] = dilation[d];
        g.out[j] = transposed ? (g.in[j] - 1) * stride[d] - 2 * pad[d] + dilation[d] * (g.kernel[j] - 1) + output_pad[d] + 1
            : (g.in[j] + 2 * pad[d] - dilation[d] * (g.kernel[j] - 1) - 1) / stride[d] + 1;
        TP_CHECK(g.out[j] > 0 && (transposed || g.in[j] + 2 * pad[d] >= dilation[d] * (g.kernel[j] - 1) + 1), "Convolution kernel exceeds the padded spatial input");
        shape[d+2] = g.out[j]; g.input_spatial *= g.in[j]; g.output_spatial *= g.out[j]; g.kernel_spatial *= g.kernel[j];
    }
    Tensor x = input.contiguous(), w = weight.contiguous(), b;
    if (bias) {
        TP_CHECK(bias->dim() == 1 && bias->numel() == g.outputs && bias->device() == input.device(), "Convolution bias must contain one value per output channel");
        b = bias->to(DType::Float32).contiguous();
    }
    Tensor y = output_float ? Tensor::empty(shape, DType::Float32, input.device()) : output(input, shape, scale, zp);
    // One output element folds every tap of the window over its slice of input
    // channels, so the chunk has to shrink by that many visits or the shape
    // never reaches the thread pool.
    const int64_t visits = g.kernel_spatial * (g.channels / g.groups);
#ifdef TP_QUANT_CUDA
    const int32_t* tap_table = nullptr;
#else
    const std::vector<int32_t> taps = build_conv_taps(g, y.numel() / g.output_spatial);
    const int32_t* tap_table = taps.empty() ? nullptr : taps.data();
#endif
    codes(input.dtype(), [&]<class T>() {
        if (output_float) apply(y.numel(), ConvCodes<T,float>{x.data_ptr<T>(),w.data_ptr<int8_t>(),
            g.channel_grid ? scales.data_ptr<float>() : nullptr, bias ? b.data_ptr<float>() : nullptr,
            g.channel_grid ? zps.data_ptr<int64_t>() : nullptr,y.data_ptr<float>(),g,tap_table},
            chunk_grain(visits));
        else apply(y.numel(), ConvCodes<T,T>{x.data_ptr<T>(),w.data_ptr<int8_t>(),
            g.channel_grid ? scales.data_ptr<float>() : nullptr,bias ? b.data_ptr<float>() : nullptr,
            g.channel_grid ? zps.data_ptr<int64_t>() : nullptr,y.data_ptr<T>(),g,tap_table},
            chunk_grain(visits));
    });
    return y;
}
}
