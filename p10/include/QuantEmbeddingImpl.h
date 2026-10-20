#pragma once
#include "QuantRowwise.h"
#ifndef TP_QUANT_CUDA
#include "cpu/vec/vec.h"
#include <cfenv>
#include <cstring>
#endif

namespace tensorplay::quant::TP_QN_DETAIL {

// Weighted codes a bag walk has to carry before sharing it with the pool beats
// finishing it on the calling thread.
constexpr int64_t kEmbeddingChunkSteps = 131072;

TP_QN_DEVICE inline float quant_loss(const float* x, int64_t size, float lo, float hi, int bits) {
    lo = static_cast<float>(Half(lo));
    const float range = hi - lo, qmax = static_cast<float>((1 << bits) - 1);
    const float scale = range == 0 ? 1.f : static_cast<float>(Half(range / qmax));
    const float inverse = scale == 0 ? 1.f : 1.f / scale;
    float sum = 0;
    for (int64_t i = 0; i < size; ++i) {
        const float q = ::fmaxf(0.f, ::fminf(qmax, ::nearbyintf((x[i] - lo) * inverse)));
        const float diff = x[i] - (q * scale + lo); sum += diff * diff;
    }
    return ::sqrtf(sum);
}

struct RowPack {
    const float *x, *ranges; uint8_t* y; int64_t columns, width, bins; int bits; bool optimized; float ratio;
    TP_QN_DEVICE void operator()(int64_t row) const {
        const float* src = x + row * columns; uint8_t* dst = y + row * width;
        float lo, hi;
        if (ranges && !optimized) {
            lo = ranges[2 * row]; hi = ranges[2 * row + 1];
        } else {
#ifndef TP_QUANT_CUDA
            using V = vec::Vectorized<float>;
            constexpr int lanes = V::size();
            if (std::fegetround() == FE_TONEAREST && columns >= lanes) {
                V minimum(std::numeric_limits<float>::infinity());
                V maximum(-std::numeric_limits<float>::infinity());
                int64_t j = 0;
                for (; j + lanes <= columns; j += lanes) {
                    const V values = V::loadu(src + j);
                    minimum = vec::minimum(minimum, values);
                    maximum = vec::maximum(maximum, values);
                }
                alignas(64) float minima[lanes], maxima[lanes];
                minimum.store(minima); maximum.store(maxima);
                lo = minima[0]; hi = maxima[0];
                for (int lane = 1; lane < lanes; ++lane) {
                    lo = std::min(lo, minima[lane]); hi = std::max(hi, maxima[lane]);
                }
                for (; j < columns; ++j) { lo = std::min(lo, src[j]); hi = std::max(hi, src[j]); }
            } else
#endif
            {
                lo = src[0]; hi = src[0];
                for (int64_t j = 1; j < columns; ++j) { lo = ::fminf(lo, src[j]); hi = ::fmaxf(hi, src[j]); }
            }
        }
        if (optimized && bits != 8 && hi > lo) {
            const float step = (hi - lo) / static_cast<float>(bins);
            const float threshold = static_cast<float>(bins) * (1.f - ratio) * step;
            float best = quant_loss(src, columns, lo, hi, bits), current = best, left = lo, right = hi;
            while (left + threshold < right) {
                const float loss_left = quant_loss(src, columns, left + step, right, bits);
                const float loss_right = quant_loss(src, columns, left, right - step, bits);
                if (current < loss_left && current < loss_right && current < best) { best = current; lo = left; hi = right; }
                if (loss_left < loss_right) { if (left + step == left) break; left += step; current = loss_left; }
                else { if (right - step == right) break; right -= step; current = loss_right; }
            }
        }
        float scale, inverse;
        if (bits == 8) { scale = (hi - lo) / 255.f; inverse = 255.f / (hi - lo + 1.e-8f); }
        else {
            lo = static_cast<float>(Half(lo));
            scale = hi == lo ? 1.f : static_cast<float>(Half((hi - lo) / ((1 << bits) - 1)));
            inverse = scale == 0 ? 1.f : 1.f / scale;
            if (scale == 0 || (inverse == std::numeric_limits<float>::infinity() || inverse == -std::numeric_limits<float>::infinity())) { scale = inverse = 1.f; }
        }
        const int values_per_byte = 8 / bits;
#ifndef TP_QUANT_CUDA
        using V = vec::Vectorized<float>;
        constexpr int lanes = V::size();
        int64_t j = 0;
        if (std::fegetround() == FE_TONEAREST && lanes >= values_per_byte && lanes % values_per_byte == 0) {
            alignas(64) float quantized[lanes];
            for (; j + lanes <= columns; j += lanes) {
                V values = ((V::loadu(src + j) - V(lo)) * V(inverse)).round();
                values = vec::minimum(V(static_cast<float>((1 << bits) - 1)), vec::maximum(V(0.f), values));
                values.store(quantized);
                for (int lane = 0; lane < lanes; lane += values_per_byte) {
                    uint8_t packed = 0;
                    for (int v = 0; v < values_per_byte; ++v)
                        packed |= static_cast<uint8_t>(quantized[lane + v]) << (v * bits);
                    dst[(j + lane) / values_per_byte] = packed;
                }
            }
        }
#else
        int64_t j = 0;
#endif
        for (; j < columns; j += values_per_byte) {
            uint8_t packed = 0;
            for (int v = 0; v < values_per_byte; ++v) {
                const float code = ::nearbyintf((src[j + v] - lo) * inverse);
                const uint8_t q = static_cast<uint8_t>(::fmaxf(0.f, ::fminf((1 << bits) - 1, code)));
                packed |= q << (v * bits);
            }
            dst[j / values_per_byte] = packed;
        }
        if (bits == 8) { put_float(dst + width - 8, scale); put_float(dst + width - 4, lo); }
        else { put_half(dst + width - 4, scale); put_half(dst + width - 2, lo); }
    }
};

Tensor rowwise_pack(const Tensor& weight, int64_t bits, bool optimized, int64_t bins,
                    double ratio, const std::optional<Tensor>& min_max) {
    Guard guard(weight);
    TP_CHECK((bits == 8 && weight.dim() >= 2) || ((bits == 2 || bits == 4) && weight.dim() == 2), "Row packing requires a matrix; byte packing also supports leading batch dimensions");
    TP_CHECK(weight.dtype() == DType::Float32 || weight.dtype() == DType::Float16, "Row packing requires float or half weights");
    TP_CHECK(bins > 0 && ratio >= 0 && ratio < 1 && std::isfinite(ratio), "Invalid grid optimization parameters");
    const int64_t columns = weight.size(weight.dim() - 1);
    TP_CHECK(columns > 0 && columns % (8 / bits) == 0, "Row width must be positive and divisible by the number of packed codes per byte");
    Tensor x = weight.to(DType::Float32).contiguous(), ranges;
    const int64_t rows = weight.numel() / columns, width = columns / (8 / bits) + (bits == 8 ? 8 : 4);
    auto shape = static_cast<std::vector<int64_t>>(weight.shape()); shape.back() = width;
    if (min_max) {
        TP_CHECK(min_max->dtype() == weight.dtype() && min_max->device() == weight.device() && min_max->numel() == rows * 2 && min_max->size(min_max->dim() - 1) == 2, "Row ranges must contain one minimum and maximum per row with matching dtype and device");
        ranges = min_max->to(DType::Float32).contiguous();
    }
    Tensor y = Tensor::empty(shape, DType::UInt8, weight.device());
#ifdef TP_QUANT_CUDA
    const RowPack pack{x.data_ptr<float>(), min_max ? ranges.data_ptr<float>() : nullptr,
        y.data_ptr<uint8_t>(), columns, width, bins, static_cast<int>(bits), optimized, static_cast<float>(ratio)};
    apply(rows, pack);
#else
    tensorplay::cpu::quantized_rowwise_pack_stub(DeviceType::CPU,
        x.data_ptr<float>(), min_max ? ranges.data_ptr<float>() : nullptr,
        y.data_ptr<uint8_t>(), rows, columns, width, bins, static_cast<int>(bits),
        optimized, static_cast<float>(ratio));
#endif
    return y;
}

Tensor embedding_weight_pack(const Tensor& weight) {
    Guard guard(weight);
    TP_CHECK(weight.dim() == 2 && (weight.dtype() == DType::QUInt8 || weight.dtype() == DType::QUInt4x2) && weight.impl()->quantizer()
        && weight.impl()->quantizer()->qscheme() == kPerChannelAffineFloatQParams
        && quantized::q_per_channel_axis(weight) == 0, "Embedding weights require unsigned row codes with floating affine parameters on axis zero");
    Tensor x = weight.contiguous();
    Tensor scales = quantized::q_per_channel_scales(weight).to(DType::Float32).contiguous();
    Tensor zps = quantized::q_per_channel_zero_points(weight).to(DType::Float32).contiguous();
    const int bits=weight.dtype()==DType::QUInt4x2 ? 4 : 8;
    const int64_t columns = (weight.size(1)+(8/bits)-1)/(8/bits), width = columns + (bits==8 ? 8 : 4);
    Tensor y = Tensor::empty({weight.size(0), width}, DType::UInt8, weight.device());
    const uint8_t* src = x.data_ptr<uint8_t>(); uint8_t* dst = y.data_ptr<uint8_t>();
    const float *ss = scales.data_ptr<float>(), *zz = zps.data_ptr<float>();
#ifdef TP_QUANT_CUDA
    apply(weight.size(0), [=] TP_QN_DEVICE (int64_t row) {
        for (int64_t j = 0; j < columns; ++j) dst[row * width + j] = src[row * columns + j];
        if (bits==8) {
            put_float(dst + row * width + columns, ss[row]);
            put_float(dst + row * width + columns + 4, -zz[row] * ss[row]);
        } else {
            put_half(dst + row * width + columns, ss[row]);
            put_half(dst + row * width + columns + 2, -zz[row] * ss[row]);
        }
    });
#else
    const int64_t row_grain = std::max<int64_t>(
        1, weight.size(0) / (parallel::get_num_threads() * 4));
    parallel::parallel_for(0, weight.size(0), row_grain, [&](int64_t first, int64_t last) {
        for (int64_t row = first; row < last; ++row) {
            uint8_t* output = dst + row * width;
            std::memcpy(output, src + row * columns, static_cast<size_t>(columns));
            if (bits == 8) {
                put_float(output + columns, ss[row]);
                put_float(output + columns + 4, -zz[row] * ss[row]);
            } else {
                put_half(output + columns, ss[row]);
                put_half(output + columns + 2, -zz[row] * ss[row]);
            }
        }
    });
#endif
    return y;
}

int64_t row_width(const Tensor& packed, int64_t bits) {
    TP_CHECK((bits == 8 || bits == 4 || bits == 2) && packed.dim() >= 2 && packed.dtype() == DType::UInt8, "Packed rows require byte storage and a valid bit width");
    const int64_t columns = (packed.size(packed.dim() - 1) - (bits == 8 ? 8 : 4)) * (8 / bits);
    TP_CHECK(columns >= 0, "Packed row is shorter than its affine metadata");
    return columns;
}

Tensor rowwise_unpack(const Tensor& packed, int64_t bits) {
    Guard guard(packed);
    const int64_t columns = row_width(packed, bits), width = packed.size(packed.dim() - 1);
    Tensor w = packed.contiguous(); auto shape = static_cast<std::vector<int64_t>>(packed.shape()); shape.back() = columns;
    Tensor y = Tensor::empty(shape, DType::Float32, packed.device());
    const uint8_t* src = w.data_ptr<uint8_t>(); float* dst = y.data_ptr<float>();
    apply(y.numel(), [=] TP_QN_DEVICE (int64_t i) {
        const uint8_t* row = src + i / columns * width; float scale, bias; row_grid(row, width, bits, scale, bias);
        dst[i] = static_cast<float>(row_code(row, i % columns, bits)) * scale + bias;
    });
    return y;
}

struct EmbeddingLookup {
    const uint8_t* w; const int64_t *indices, *offsets, *mapping; const float* weights; float* y;
    int64_t columns, width, bags, index_count, fixed_length; int bits; bool embedding;
    TP_QN_DEVICE void operator()(int64_t i) const {
        const int64_t bag = i / columns, col = i % columns;
        const int64_t first = embedding ? bag : (offsets ? offsets[bag] : bag * fixed_length);
        const int64_t last = embedding ? bag + 1 : (offsets ? (bag + 1 < bags ? offsets[bag + 1] : index_count) : first + fixed_length);
        float sum = 0;
        for (int64_t j = first; j < last; ++j) {
            const int64_t row_index = mapping ? mapping[indices[j]] : indices[j];
            if (row_index == -1) continue;
            const uint8_t* row = w + row_index * width; float scale, bias; row_grid(row, width, bits, scale, bias);
            const float weight = weights ? weights[j] : 1.f;
            sum = ::fmaf(weight * scale, static_cast<float>(row_code(row, col, bits)), sum + weight * bias);
        }
        y[i] = sum;
    }
};

Tensor embedding_lookup(const Tensor& packed, const Tensor& indices, int64_t bits,
        const std::optional<Tensor>& offsets, const std::optional<Tensor>& per_sample,
        const std::optional<Tensor>& mapping, bool include_last, bool embedding, bool pruned) {
    Guard guard(packed);
    const int64_t columns = row_width(packed, bits), width = packed.size(1);
    TP_CHECK(packed.dim() == 2 && indices.device() == packed.device() && (indices.dtype() == DType::Int32 || indices.dtype() == DType::Int64), "Embedding indices require integer storage on the weight device");
    TP_CHECK(embedding || indices.dim() == 1 || indices.dim() == 2, "Embedding bag indices must be a vector or matrix");
    TP_CHECK(!pruned || mapping, "Pruned rows require an index mapping");
    Tensor w = packed.contiguous(), ind = indices.to(DType::Int64).contiguous(), off, ps, map;
    bool sparse = pruned;
    if (mapping && pruned) {
        TP_CHECK(mapping->device() == packed.device() && mapping->dim() == 1 && mapping->dtype() == DType::Int32, "Compressed mapping must be an integer vector on the weight device");
        map = mapping->to(DType::Int64).contiguous();
        if (map.numel() == 1 && map.to(Device(DeviceType::CPU)).item<int64_t>() == 0) sparse = false;
    }
    // Bounds are checked before issuing device work so malformed inputs raise
    // a recoverable exception without poisoning a device execution context.
    // The row domain and the count of indices hold for every index, so each is read
    // once: a scan that reaches back for the tensor it is guarding turns every index
    // into calls the walk cannot fold away, and the scan is as long as the indices.
    Tensor host_ind = ind.to(Device(DeviceType::CPU)), host_map;
    if (sparse) host_map = map.to(Device(DeviceType::CPU));
    const int64_t* ids = host_ind.data_ptr<int64_t>();
    const int64_t rows = packed.size(0), limit = sparse ? map.numel() : rows;
    const int64_t* mapped = sparse ? host_map.data_ptr<int64_t>() : nullptr;
    for (int64_t i = 0, id_count = ind.numel(); i < id_count; ++i) {
        TP_CHECK(ids[i] >= 0 && ids[i] < limit, "Embedding index exceeds its row domain");
        if (sparse) TP_CHECK(mapped[ids[i]] >= -1 && mapped[ids[i]] < rows, "Compressed mapping exceeds its stored row domain");
    }
    int64_t bags = indices.numel(), count = indices.numel(), fixed = 1;
    std::vector<int64_t> shape;
    if (embedding) { shape = static_cast<std::vector<int64_t>>(indices.shape()); shape.push_back(columns); }
    else if (indices.dim() == 2) {
        TP_CHECK(!offsets, "Matrix bag indices do not accept offsets");
        bags = indices.size(0); fixed = indices.size(1); shape = {bags, columns};
    } else {
        TP_CHECK(offsets && offsets->dim() == 1 && offsets->device() == packed.device()
            && (offsets->dtype() == DType::Int32 || offsets->dtype() == DType::Int64), "Vector bag indices require integer offsets on the weight device");
        off = offsets->to(DType::Int64).contiguous();
        TP_CHECK(!include_last || off.numel() >= 1, "Offsets must include a terminal position");
        bags = off.numel() - (include_last ? 1 : 0); shape = {bags, columns};
        Tensor host = off.to(Device(DeviceType::CPU)); const int64_t* values = host.data_ptr<int64_t>();
        const int64_t marks = off.numel();
        if (marks) TP_CHECK(values[0] == 0, "First bag offset must be zero");
        for (int64_t i = 0; i < marks; ++i) TP_CHECK(values[i] >= 0 && values[i] <= count && (i == 0 || values[i] >= values[i-1]), "Bag offsets must be ordered positions in the index vector");
        if (include_last) count = values[marks - 1];
    }
    if (per_sample) {
        TP_CHECK(per_sample->device() == packed.device() && per_sample->dtype() == DType::Float32 && per_sample->shape() == indices.shape(), "Per-sample weights must be float values with the index shape and device");
        ps = per_sample->contiguous();
    }
    Tensor y = Tensor::empty(shape, DType::Float32, packed.device());
    const int64_t* pm = sparse ? map.data_ptr<int64_t>() : nullptr;
    const int64_t* po = offsets && !embedding ? off.data_ptr<int64_t>() : nullptr;
    const float* pp = per_sample ? ps.data_ptr<float>() : nullptr;
#ifdef TP_QUANT_CUDA
    apply(y.numel(), EmbeddingLookup{w.data_ptr<uint8_t>(), ind.data_ptr<int64_t>(), po, pm, pp,
        y.data_ptr<float>(), columns, width, bags, count, fixed, static_cast<int>(bits), embedding});
#else
    // A walk of a bag decodes a code per column of it, and a code of less than a
    // byte takes several steps more to unfold than a byte does, so the work of a
    // bag is its codes weighted by the width they arrive in. Below the cost of a
    // handoff the walk stays whole on the calling thread; above it the walk splits
    // one bag per chunk, because fewer chunks than threads leave workers idle and
    // the bags are alike enough that no finer bound is needed.
    const int64_t per_bag = embedding ? 1
        : (indices.dim() == 2 ? fixed : (bags > 0 ? count / bags : 1));
    const int64_t steps = std::max<int64_t>(1, per_bag * columns) * (8 + bits) / bits;
    const int64_t grain = bags > kEmbeddingChunkSteps / steps ? 1 : std::max<int64_t>(1, bags);
    parallel::parallel_for(0, bags, grain, [&](int64_t first, int64_t last) {
        tensorplay::cpu::quantized_embedding_stub(DeviceType::CPU, w.data_ptr<uint8_t>(), ind.data_ptr<int64_t>(),
            po, pm, pp, y.data_ptr<float>(), first, last, bags, count, fixed, columns, width, bits, embedding);
    });
#endif
    return y;
}

Tensor embedding(const Tensor& w, const Tensor& ind, int64_t bits) {
    return embedding_lookup(w, ind, bits, {}, {}, {}, false, true, false);
}
Tensor embedding_bag(const Tensor& w, const Tensor& ind, int64_t bits, const std::optional<Tensor>& offsets,
        const std::optional<Tensor>& weights, const std::optional<Tensor>& mapping, bool include_last, bool pruned) {
    return embedding_lookup(w, ind, bits, offsets, weights, mapping, include_last, false, pruned);
}
}
