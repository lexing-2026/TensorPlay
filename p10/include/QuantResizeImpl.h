#pragma once
#include "MemoryFormat.h"

namespace tensorplay::quant::TP_QN_DETAIL {
struct ResizeGeometry {
    int rank = 0;
    int64_t channels = 0, pixels = 1, in[3]{1,1,1}, out[3]{1,1,1};
    int64_t source_stride[5]{}, target_stride[5]{};
    float scale[3]{};
    bool exact = false, bilinear = false, align = false;
    int64_t zp = 0;
    TP_QN_DEVICE void position(int64_t i, int64_t& base, int64_t& target,
            int64_t* lower, float* fraction) const {
        const int64_t channel = i / pixels % channels;
        const int64_t batch = i / (pixels * channels);
        base = batch * source_stride[0] + channel * source_stride[1];
        target = batch * target_stride[0] + channel * target_stride[1];
        int64_t pixel = i % pixels;
        for (int d = rank - 1; d >= 0; --d) {
            const int64_t coord = pixel % out[d]; pixel /= out[d];
            target += coord * target_stride[d+2];
            if (bilinear) {
                const float p = align ? scale[d] * coord
                    : ::fmaxf(scale[d] * (static_cast<float>(coord) + .5f) - .5f, 0.f);
                lower[d] = static_cast<int64_t>(p);
                fraction[d] = p - lower[d];
            } else lower[d] = static_cast<int64_t>(::floorf(scale[d] *
                (static_cast<float>(coord) + (exact ? .5f : 0.f))));
            if (lower[d] >= in[d]) lower[d] = in[d] - 1;
            base += lower[d] * source_stride[d+2];
        }
    }
};

template<class T>
struct ResizeCodes {
    const T* x; T* y; ResizeGeometry g;
    TP_QN_DEVICE void operator()(int64_t i) const {
        int64_t base, target, lower[3]{}; float fraction[3]{};
        g.position(i, base, target, lower, fraction);
        if (!g.bilinear) { y[target] = x[base]; return; }
        const int64_t dh = lower[0] < g.in[0] - 1 ? g.source_stride[2] : 0;
        const int64_t dw = lower[1] < g.in[1] - 1 ? g.source_stride[3] : 0;
        const float h = fraction[0], w = fraction[1];
        const float value = (1.f-h) * ((1.f-w) * x[base] + w * x[base+dw])
            + h * ((1.f-w) * x[base+dh] + w * x[base+dh+dw]);
        y[target] = encode<T>(value - static_cast<float>(g.zp), 1.f, g.zp);
    }
};

Tensor resize(const Tensor& input, const std::vector<int64_t>& sizes,
        const std::vector<std::optional<double>>& scales, bool exact, bool bilinear, bool align) {
    Guard guard(input); validate(input);
    const int rank = sizes.size();
    TP_CHECK((rank == 2 || rank == 3) && input.dim() == rank + 2 && scales.size() == sizes.size(),
        "Quantized resize requires a four or five dimensional spatial tensor");
    TP_CHECK(!bilinear || rank == 2, "Bilinear resize requires two spatial dimensions");
    TP_CHECK(input.size(1) > 0 && (rank != 3 || input.size(0) > 0), "Resize requires nonempty channels and three dimensional batches");
    ResizeGeometry g; g.rank = rank; g.channels = input.size(1);
    g.exact = exact; g.bilinear = bilinear; g.align = align;
    g.zp = quantized::q_zero_point(input);
    auto shape = static_cast<std::vector<int64_t>>(input.shape());
    bool identical = true;
    for (int d = 0; d < rank; ++d) {
        TP_CHECK(input.size(d+2) > 0 && sizes[d] > 0, "Resize spatial dimensions must be positive");
        shape[d+2] = sizes[d]; g.in[d] = input.size(d+2); g.out[d] = sizes[d]; g.pixels *= sizes[d];
        identical = identical && g.in[d] == g.out[d];
        g.scale[d] = bilinear && align ? (sizes[d] > 1 ? static_cast<float>(
            static_cast<double>(g.in[d]-1) / (sizes[d]-1)) : 0.f)
            : scales[d] && *scales[d] > 0 ? static_cast<float>(1. / *scales[d])
            : static_cast<float>(static_cast<double>(g.in[d]) / sizes[d]);
    }
    if (identical) return input.clone();
    Tensor y = output(input, shape, quantized::q_scale(input), g.zp);
    const bool channels_last = input.is_contiguous(rank == 2 ? MemoryFormat::ChannelsLast : MemoryFormat::ChannelsLast3d);
    if (channels_last) {
        y = y.as_strided(shape, get_channels_last_strides(shape), 0);
        y.impl()->set_quantizer(input.impl()->quantizer());
    }
    for (int d = 0; d < input.dim(); ++d) {
        g.source_stride[d] = input.stride(d); g.target_stride[d] = y.stride(d);
    }
    codes(input.dtype(), [&]<class T>() {
        ResizeCodes<T> fn{input.data_ptr<T>(), y.data_ptr<T>(), g};
#ifndef TP_QUANT_CUDA
        if (bilinear && channels_last) {
            parallel::parallel_for(0, input.size(0) * g.pixels, 1024, [&](int64_t first, int64_t last) {
                for (int64_t pixel = first; pixel < last; ++pixel) {
                    const int64_t index = pixel / g.pixels * g.channels * g.pixels + pixel % g.pixels;
                    int64_t base, target, lower[3]{}; float fraction[3]{};
                    g.position(index, base, target, lower, fraction);
                    const int64_t dh = lower[0] < g.in[0]-1 ? g.source_stride[2] : 0;
                    const int64_t dw = lower[1] < g.in[1]-1 ? g.source_stride[3] : 0;
                    tensorplay::cpu::quantized_bilinear_stub(DeviceType::CPU, fn.x+base,
                        fn.x+base+dw, fn.x+base+dh, fn.x+base+dh+dw, fn.y+target,
                        g.channels, input.dtype(), fraction[0], fraction[1], g.zp);
                }
            });
        } else apply(y.numel(), fn);
#else
        apply(y.numel(), fn);
#endif
    });
    return y;
}
}
