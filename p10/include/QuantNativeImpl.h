#pragma once
#include "QuantNative.h"
#include "Dispatcher.h"
#include "DispatchStub.h"
#include "Parallel.h"
#include "Half.h"
#include <algorithm>
#include <array>
#include <numeric>
#include <cstring>
#include <type_traits>
#ifdef TP_QUANT_CUDA
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define TP_QN_DEVICE __device__
#define TP_QN_DETAIL cuda_native
#define TP_QN_NAME(x) x##_cuda
#else
#include "cpu/QuantizedElementwiseKernels.h"
#include "cpu/vec/vec.h"
#define TP_QN_DEVICE
#define TP_QN_DETAIL cpu_native
#define TP_QN_NAME(x) x##_cpu
#endif

namespace tensorplay::quant {
namespace TP_QN_DETAIL {

#ifdef TP_QUANT_CUDA
template<class F>
__global__ void apply_kernel(int64_t size, F fn) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(gridDim.x) * blockDim.x) fn(i);
}
#endif

template<class F>
void apply(int64_t size, F fn) {
    if (!size) return;
#ifdef TP_QUANT_CUDA
    const int blocks = static_cast<int>(std::min<int64_t>((size + 255) / 256, 4096));
    apply_kernel<<<blocks, 256, 0, tensorplay::cuda::getCurrentCUDAStream().stream()>>>(size, fn);
    tensorplay::cuda::checkCuda(cudaGetLastError(), "Quantized execution");
#else
    parallel::parallel_for(0, size, 32768, [&](int64_t first, int64_t last) {
        for (int64_t i = first; i < last; ++i) fn(i);
    });
#endif
}

struct Guard {
#ifdef TP_QUANT_CUDA
    tensorplay::cuda::CUDAGuard value;
    explicit Guard(const Tensor& x) : value(x.device()) {}
#else
    explicit Guard(const Tensor&) {}
#endif
};

template<class F>
void codes(DType dtype, F&& fn) {
    switch (dtype) {
        case DType::QInt8: fn.template operator()<int8_t>(); break;
        case DType::QUInt8: fn.template operator()<uint8_t>(); break;
        case DType::QInt32: fn.template operator()<int32_t>(); break;
        default: TP_THROW(TypeError, "Expected affine quantized integer storage");
    }
}

void validate(const Tensor& x) {
    TP_CHECK(x.impl()->quantizer(), "Expected a quantized tensor");
    quantized::q_scale(x);
    quantized::q_zero_point(x);
}

Tensor output(const Tensor& x, const std::vector<int64_t>& shape, double scale, int64_t zp) {
    auto grid = make_per_tensor_affine_quantizer(scale, zp, x.dtype());
    Tensor y = Tensor::empty(shape, x.dtype(), x.device());
    y.impl()->set_quantizer(std::move(grid));
    return y;
}

struct Broadcast {
    int rank = 0;
    int64_t dims[64]{}, a[64]{}, b[64]{};
    TP_QN_DEVICE void indices(int64_t index, int64_t& ai, int64_t& bi) const {
        ai = bi = 0;
        for (int d = rank - 1; d >= 0; --d) {
            const int64_t c = index % dims[d]; index /= dims[d];
            ai += c * a[d]; bi += c * b[d];
        }
    }
};

std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    std::vector<int64_t> shape(std::max(a.size(), b.size()), 1);
    TP_CHECK(shape.size() <= 64, "Broadcast rank exceeds 64 dimensions");
    for (size_t i = 0; i < shape.size(); ++i) {
        const int64_t av = i < a.size() ? a[a.size() - 1 - i] : 1;
        const int64_t bv = i < b.size() ? b[b.size() - 1 - i] : 1;
        TP_CHECK(av == bv || av == 1 || bv == 1, "Operand dimensions cannot broadcast");
        shape[shape.size() - 1 - i] = av == 1 ? bv : av;
    }
    return shape;
}

Broadcast broadcast(const Tensor& a, const Tensor& b, const std::vector<int64_t>& shape) {
    Broadcast g; g.rank = static_cast<int>(shape.size());
    int64_t sa = 1, sb = 1;
    for (int d = g.rank - 1; d >= 0; --d) {
        g.dims[d] = shape[d];
        const int ad = d - (g.rank - a.dim()), bd = d - (g.rank - b.dim());
        if (ad >= 0) { g.a[d] = a.size(ad) == 1 ? 0 : a.stride(ad); sa *= a.size(ad); }
        if (bd >= 0) { g.b[d] = b.size(bd) == 1 ? 0 : b.stride(bd); sb *= b.size(bd); }
    }
    return g;
}

template<class T>
struct Binary {
    const T *a, *b; T* y; Broadcast g;
    float as, bs, inverse; int64_t az, bz, yz; int op; bool relu;
    TP_QN_DEVICE void operator()(int64_t i) const {
        int64_t ai, bi; g.indices(i, ai, bi);
        const float av = static_cast<float>(static_cast<int64_t>(a[ai]) - az) * as;
        const float bv = static_cast<float>(static_cast<int64_t>(b[bi]) - bz) * bs;
        float v;
        switch (op) {
            case 0: v = av + bv; break;
            case 1: v = av - bv; break;
            case 2: v = av * bv; break;
            default: v = av / bv; break;
        }
        if (relu) v = ::fmaxf(v, 0.f);
        y[i] = encode<T>(v, inverse, yz);
    }
};

Tensor binary(const Tensor& a, const Tensor& b, double as, int64_t az, double bs, int64_t bz,
              double ys, int64_t yz, int64_t op, bool relu) {
    Guard guard(a); validate(a); validate(b);
    TP_CHECK(a.dtype() == b.dtype() && a.device() == b.device(), "Operand dtype and device must match");
    TP_CHECK(op >= 0 && op <= 3, "Invalid binary operation");
    make_per_tensor_affine_quantizer(as, az, a.dtype());
    make_per_tensor_affine_quantizer(bs, bz, b.dtype());
    const auto shape = broadcast_shape(a.shape(), b.shape());
    Tensor x = a.contiguous(), z = b.contiguous(), y = output(a, shape, ys, yz);
#ifndef TP_QUANT_CUDA
    if (a.dtype() != DType::QInt32 && a.shape() == b.shape()) {
        parallel::parallel_for(0, y.numel(), 32768, [&](int64_t first, int64_t last) {
            tensorplay::cpu::quantized_binary_stub(DeviceType::CPU,
                static_cast<const uint8_t*>(x.data_ptr()) + first * a.itemsize(),
                static_cast<const uint8_t*>(z.data_ptr()) + first * b.itemsize(),
                static_cast<uint8_t*>(y.data_ptr()) + first * y.itemsize(), last - first,
                a.dtype(), as, az, bs, bz, ys, yz, static_cast<int>(op), relu);
        });
        return y;
    }
#endif
    const auto g = broadcast(x, z, shape);
    codes(a.dtype(), [&]<class T>() {
        apply(y.numel(), Binary<T>{x.data_ptr<T>(), z.data_ptr<T>(), y.data_ptr<T>(), g,
            static_cast<float>(as), static_cast<float>(bs), 1.f / static_cast<float>(ys), az, bz, yz, static_cast<int>(op), relu});
    });
    return y;
}

Tensor binary_op(const Tensor& a, const Tensor& b, double scale, int64_t zp, int64_t op, bool relu) {
    return binary(a, b, quantized::q_scale(a), quantized::q_zero_point(a),
                  quantized::q_scale(b), quantized::q_zero_point(b), scale, zp, op, relu);
}

template<class T>
struct ScalarCodes {
    const T* x; T* y; int mode; int64_t delta, xz, yz; float multiplier; bool relu;
    TP_QN_DEVICE T value(T code) const {
        int64_t q;
        if (mode == 0) q = code;
        else if (mode == 1) q = 0;
        else if (mode == 2) q = static_cast<int64_t>(std::numeric_limits<T>::max()) + std::numeric_limits<T>::lowest() - code;
        else {
            T result = encode<T>(static_cast<float>(static_cast<int64_t>(code) - xz + delta), multiplier, yz);
            return relu && static_cast<int64_t>(result) < yz ? static_cast<T>(yz) : result;
        }
        if (relu && q < yz) q = yz;
        return static_cast<T>(q);
    }
    TP_QN_DEVICE void operator()(int64_t i) const { y[i] = value(x[i]); }
};

Tensor scalar(const Tensor& input, double value, int64_t op, bool relu) {
    Guard guard(input); validate(input);
    TP_CHECK(std::isfinite(value), "Scalar must be finite");
    TP_CHECK(op == 0 || op == 2, "Scalar operation must be addition or multiplication");
    Tensor x = input.contiguous(), y;
    codes(input.dtype(), [&]<class T>() {
        const double s = quantized::q_scale(input);
        const int64_t z = quantized::q_zero_point(input), lo = std::numeric_limits<T>::lowest(), hi = std::numeric_limits<T>::max();
        double ys = s; int64_t yz = z, delta = 0; int mode = 0;
        if (op == 0) {
            const double rounded = std::nearbyint(value / s);
            TP_CHECK(rounded >= -9.22e18 && rounded <= 9.22e18, "Scalar code exceeds the integer range");
            delta = static_cast<int64_t>(rounded);
            const double shifted = static_cast<double>(z) - delta;
            if (shifted < lo) { ys = (hi - shifted) / (static_cast<double>(hi) - lo) * s; yz = lo; mode = 3; }
            else if (shifted > hi) { ys = (shifted - lo) / (static_cast<double>(hi) - lo) * s; yz = hi; mode = 3; }
            else yz = z - delta;
        } else if (value > 0) ys = value * s;
        else if (value == 0) { ys = 1; yz = 0; mode = 1; }
        else { ys = -value * s; yz = hi - (z - lo); mode = 2; }
        y = output(input, input.shape(), ys, yz);
        ScalarCodes<T> fn{x.data_ptr<T>(), y.data_ptr<T>(), mode, delta, z, yz, static_cast<float>(s / ys), relu};
#ifndef TP_QUANT_CUDA
        if (mode==0) {
            parallel::parallel_for(0,y.numel(),32768,[&](int64_t first,int64_t last) {
                if (relu) cpu::quantized_clip_stub(DeviceType::CPU,fn.x+first,fn.y+first,last-first,
                    input.dtype(),yz,hi);
                else std::memcpy(fn.y+first,fn.x+first,static_cast<size_t>(last-first)*sizeof(T));
            });
            return;
        }
        if constexpr (sizeof(T) == 1) {
            std::array<float, 256> table{};
            for (int q = 0; q < 256; ++q) table[q] = static_cast<uint8_t>(fn.value(static_cast<T>(q)));
            parallel::parallel_for(0, y.numel(), 32768, [&](int64_t first, int64_t last) {
                tensorplay::cpu::quantized_lookup_stub(DeviceType::CPU,
                    reinterpret_cast<const uint8_t*>(fn.x) + first,
                    reinterpret_cast<uint8_t*>(fn.y) + first, last - first, table.data());
            });
        } else apply(y.numel(), fn);
#else
        apply(y.numel(), fn);
#endif
    });
    return y;
}

Tensor& scalar_out(const Tensor& input, double value, int64_t op, Tensor& out, bool relu) {
    validate(out);
    TP_CHECK(out.dtype() == input.dtype() && out.device() == input.device()
        && out.shape() == input.shape(), "Scalar output shape, dtype and device must match the input");
    Tensor result = scalar(input, value, op, relu);
    out.impl()->set_quantizer(result.impl()->quantizer());
    out.copy_(result);
    return out;
}

template<class T>
struct CatCodes {
    const T* x; T* y; int64_t chunk, inner, width, offset, xz, yz; float scale, inverse; bool relu;
    TP_QN_DEVICE void operator()(int64_t i) const {
        const int64_t block = chunk * inner;
        const int64_t dest = (i / block * width + offset) * inner + i % block;
        float v = static_cast<float>(static_cast<int64_t>(x[i]) - xz) * scale;
        if (relu) v = ::fmaxf(v, 0.f);
        y[dest] = encode<T>(v, inverse, yz);
    }
};

template<class T>
struct CatRows {
    const T* x; T* y; int64_t chunk, width, offset, xz, yz; float scale, inverse; bool relu;
    TP_QN_DEVICE void operator()(int64_t row) const {
        const T* source = x + row * chunk;
        T* destination = y + row * width + offset;
        for (int64_t column = 0; column < chunk; ++column) {
            float value = static_cast<float>(static_cast<int64_t>(source[column]) - xz) * scale;
            if (relu) value = ::fmaxf(value, 0.f);
            destination[column] = encode<T>(value, inverse, yz);
        }
    }
};

Tensor cat(const std::vector<Tensor>& inputs, int64_t dim, std::optional<double> scale,
           std::optional<int64_t> zp, bool relu) {
    TP_CHECK(!inputs.empty(), "Concatenation requires at least one operand");
    const Tensor& first = inputs.front(); Guard guard(first); validate(first);
    TP_CHECK(first.dim() > 0, "Concatenation does not accept scalar operands");
    if (dim < 0) dim += first.dim();
    TP_CHECK(dim >= 0 && dim < first.dim(), "Concatenation dimension is invalid");
    auto shape = static_cast<std::vector<int64_t>>(first.shape()); shape[dim] = 0;
    for (const auto& x : inputs) {
        validate(x);
        TP_CHECK(x.dtype() == first.dtype() && x.device() == first.device() && x.dim() == first.dim(), "Concatenation operands must share dtype, device and rank");
        for (int d = 0; d < x.dim(); ++d) if (d != dim) TP_CHECK(x.size(d) == first.size(d), "Concatenation dimensions do not match");
        shape[dim] += x.size(dim);
    }
    const double ys = scale.value_or(quantized::q_scale(first));
    const int64_t yz = zp.value_or(quantized::q_zero_point(first));
    Tensor y = output(first, shape, ys, yz);
    int64_t outer = 1, inner = 1, offset = 0;
    for (int d = 0; d < dim; ++d) outer *= first.size(d);
    for (int d = dim + 1; d < first.dim(); ++d) inner *= first.size(d);
    for (const auto& src : inputs) {
        Tensor x = src.contiguous();
        codes(src.dtype(), [&]<class T>() {
            if (inner == 1) {
#ifdef TP_QUANT_CUDA
                apply(outer, CatRows<T>{x.data_ptr<T>(), y.data_ptr<T>(), src.size(dim),
                    shape[dim], offset, quantized::q_zero_point(src), yz,
                    static_cast<float>(quantized::q_scale(src)), 1.f / static_cast<float>(ys), relu});
#else
                if constexpr (std::is_same_v<T, int8_t>) {
                    const bool same_qparams = quantized::q_scale(src) == ys &&
                        quantized::q_zero_point(src) == yz;
                    tensorplay::cpu::quantized_cat_rows_stub(
                        DeviceType::CPU, x.data_ptr<int8_t>(), y.data_ptr<int8_t>(),
                        outer, src.size(dim), shape[dim], offset,
                        quantized::q_scale(src), quantized::q_zero_point(src),
                        ys, yz, same_qparams, relu);
                } else {
                    apply(outer, CatRows<T>{x.data_ptr<T>(), y.data_ptr<T>(), src.size(dim),
                        shape[dim], offset, quantized::q_zero_point(src), yz,
                        static_cast<float>(quantized::q_scale(src)), 1.f / static_cast<float>(ys), relu});
                }
#endif
            } else {
                apply(x.numel(), CatCodes<T>{x.data_ptr<T>(), y.data_ptr<T>(), src.size(dim), inner,
                    shape[dim], offset, quantized::q_zero_point(src), yz,
                    static_cast<float>(quantized::q_scale(src)), 1.f / static_cast<float>(ys), relu});
            }
        });
        offset += src.size(dim);
    }
    return y;
}

template<class T>
struct PoolCodes {
    const T* x; T* y; PoolGeometry g;
    TP_QN_DEVICE void operator()(int64_t i) const {
        const int64_t outer = g.channels_last ? i / (g.channels * g.out_size) * g.channels + i % g.channels : i / g.out_size;
        int64_t start[3]{}, end[3]{1,1,1}, count = 1, valid = 1;
        int64_t p = g.channels_last ? i / g.channels % g.out_size : i % g.out_size;
        for (int d = g.rank - 1; d >= 0; --d) {
            const int64_t o = p % g.out[d]; p /= g.out[d];
            if (g.mode == 1) {
                start[d] = o * g.in[d] / g.out[d];
                end[d] = ((o + 1) * g.in[d] + g.out[d] - 1) / g.out[d];
            } else {
                start[d] = o * g.stride[d] - g.pad[d];
                end[d] = start[d] + (g.kernel[d] - 1) * g.dilation[d] + 1;
                const int64_t padded_end = end[d] < g.in[d] + g.pad[d] ? end[d] : g.in[d] + g.pad[d];
                count *= padded_end - start[d];
            }
            if (g.mode != 2) {
                if (start[d] < 0) start[d] = 0;
                if (end[d] > g.in[d]) end[d] = g.in[d];
                valid *= end[d] - start[d];
            } else {
                while (start[d] < 0) start[d] += g.dilation[d];
                if (end[d] > g.in[d]) end[d] = g.in[d];
            }
        }
        const T* row = x + outer / g.channels * g.batch_stride + outer % g.channels * g.channel_stride;
        int64_t raw_sum = 0, maximum = std::numeric_limits<T>::lowest();
        uint32_t wrapped_sum = 0;
        const bool raw_adaptive = g.mode == 1 && !g.channels_last;
        for (int64_t a = start[0]; a < end[0]; a += g.dilation[0])
            for (int64_t b = start[1]; b < end[1]; b += g.dilation[1])
                for (int64_t c = start[2]; c < end[2]; c += g.dilation[2]) {
                    const int64_t code = row[a * g.spatial_strides[0] + b * g.spatial_strides[1] + c * g.spatial_strides[2]];
                    if (g.mode == 2) {
                        if (code > maximum) maximum = code;
                    } else if (raw_adaptive) raw_sum += code;
                    else wrapped_sum += static_cast<uint32_t>(code) - static_cast<uint32_t>(g.zp);
                }
        if (g.mode == 2) y[i] = static_cast<T>(maximum);
        else if (raw_adaptive) {
            float inverse = 1.f;
            for (int d = 0; d < 3; ++d) inverse /= static_cast<float>(end[d] - start[d]);
            const float code = ::nearbyintf(static_cast<float>(raw_sum) * inverse);
            if constexpr (sizeof(T) == 4)
                y[i] = code >= 0x1p31f || code < -0x1p31f ? std::numeric_limits<T>::lowest() : static_cast<T>(code);
            else y[i] = static_cast<T>(code);
        } else {
            const int64_t divisor = g.divisor ? g.divisor : (g.mode == 1 || !g.include_pad ? valid : count);
            const int32_t sum = static_cast<int32_t>(wrapped_sum);
#ifdef TP_QUANT_CUDA
            const double code = divisor ? ::nearbyintf(__fadd_rn(
                __fmul_rn(static_cast<float>(sum), 1.f / divisor), static_cast<float>(g.zp))) : g.zp;
#else
            const double code = divisor ? ::nearbyintf(static_cast<float>(sum) * (1.f / divisor) + static_cast<float>(g.zp)) : g.zp;
#endif
            y[i] = static_cast<T>(::fmin(static_cast<double>(std::numeric_limits<T>::max()), ::fmax(static_cast<double>(std::numeric_limits<T>::lowest()), code)));
        }
    }
};

std::vector<int64_t> parameters(const std::vector<int64_t>& v, int rank, int64_t fallback) {
    if (v.empty()) return std::vector<int64_t>(rank, fallback);
    if (v.size() == 1) return std::vector<int64_t>(rank, v[0]);
    TP_CHECK(v.size() == static_cast<size_t>(rank), "Spatial parameter rank does not match");
    return v;
}

Tensor pool(const Tensor& input, const std::vector<int64_t>& kernel,
            const std::vector<int64_t>& strides, const std::vector<int64_t>& pads,
            const std::vector<int64_t>& dilations, bool ceil, bool include_pad,
            std::optional<int64_t> divisor, int mode) {
    Guard guard(input); validate(input);
    const int rank = static_cast<int>(kernel.size());
    TP_CHECK(rank >= 1 && rank <= 3 && (input.dim() == rank + 1 || input.dim() == rank + 2), "Pooling requires one to three spatial dimensions");
    TP_CHECK(mode >= 0 && mode <= 2, "Invalid pooling operation");
    TP_CHECK(!divisor || *divisor != 0, "Pooling divisor must be nonzero");
    const auto stride = strides.empty() ? kernel : parameters(strides, rank, 1);
    const auto pad = parameters(pads, rank, 0), dilation = parameters(dilations, rank, 1);
    PoolGeometry g{}; g.rank = rank; g.mode = mode; g.zp = quantized::q_zero_point(input);
    g.divisor = divisor.value_or(0); g.include_pad = include_pad;
    auto shape = static_cast<std::vector<int64_t>>(input.shape());
    for (int d = 0; d < rank; ++d) {
        const int j = 3 - rank + d;
        g.in[j] = input.size(input.dim() - rank + d);
        g.spatial_strides[j] = input.stride(input.dim() - rank + d);
        TP_CHECK(g.in[j] > 0, "Pooling input spatial dimensions must be positive");
        if (mode == 1) {
            TP_CHECK(kernel[d] >= 0, "Adaptive output dimensions must be nonnegative");
            g.out[j] = kernel[d];
        } else {
            TP_CHECK(kernel[d] > 0 && stride[d] > 0 && dilation[d] > 0 && pad[d] >= 0 && pad[d] <= kernel[d] / 2, "Invalid pooling window parameters");
            TP_CHECK(mode == 2 || dilation[d] == 1, "Average pooling does not support dilated windows");
            g.kernel[j] = kernel[d]; g.stride[j] = stride[d]; g.pad[j] = pad[d]; g.dilation[j] = dilation[d];
            const int64_t numerator = g.in[j] + 2 * pad[d] - dilation[d] * (kernel[d] - 1) - 1 + (ceil ? stride[d] - 1 : 0);
            g.out[j] = static_cast<int64_t>(std::floor(static_cast<double>(numerator) / stride[d])) + 1;
            if (ceil && (g.out[j] - 1) * stride[d] >= g.in[j] + pad[d]) --g.out[j];
            TP_CHECK(g.out[j] > 0, "Pooling produces an empty spatial dimension");
        }
        shape[input.dim() - rank + d] = g.out[j];
        g.in_size *= g.in[j]; g.out_size *= g.out[j];
    }
    // Coordinates occupy the last spatial slots, keeping every dimensional
    // variant on the same three-axis iteration without synthetic padding.
    g.rank = 3;
    auto format = activation_format(input);
    if (mode == 0 && rank == 3 && input.dim() == 5)
        format = MemoryFormat::ChannelsLast3d;
    else if (format != MemoryFormat::Contiguous && !input.impl()->is_contiguous_in(format))
        format = MemoryFormat::Contiguous;
    g.channels_last = format != MemoryFormat::Contiguous;
    const int channel_dim = input.dim() - rank - 1;
    g.channels = input.size(channel_dim);
    g.channel_stride = input.stride(channel_dim);
    g.batch_stride = channel_dim ? input.stride(0) : 0;
    Tensor source = input;
#ifndef TP_QUANT_CUDA
    if (mode == 0 && rank == 3 && input.dim() == 5 && input.itemsize() == 1) {
        if (input.impl()->is_contiguous_in(MemoryFormat::Contiguous) &&
            !input.impl()->is_contiguous_in(MemoryFormat::ChannelsLast3d) && g.channels * g.in_size >= 3600) {
            const auto source_shape = input.sizes().vec();
            Tensor storage = Tensor::empty(source_shape, input.dtype(), input.device());
            source = Tensor(storage.impl()->storage(), source_shape, get_channels_last_strides(source_shape), input.dtype(), 0);
            source.impl()->set_quantizer(input.impl()->quantizer());
            const DType storage_type = underlying_storage_type(input.dtype());
            for (int64_t b = 0; b < input.size(0); ++b) {
                Tensor from(input.impl()->storage(), {g.in_size, g.channels}, {1, g.in_size},
                    storage_type, input.storage_offset() + b * g.batch_stride);
                Tensor to(source.impl()->storage(), {g.in_size, g.channels}, {g.channels, 1},
                    storage_type, b * g.in_size * g.channels);
                to.copy_(from);
            }
        } else source = input.contiguous(static_cast<int64_t>(MemoryFormat::ChannelsLast3d));
        g.channel_stride = source.stride(channel_dim);
        g.batch_stride = source.stride(0);
        for (int d = 0; d < rank; ++d) g.spatial_strides[d] = source.stride(source.dim() - rank + d);
    }
#endif
    Tensor y = output(input, shape, quantized::q_scale(input), g.zp);
    if (g.channels_last) {
        Tensor formatted(y.impl()->storage(), shape, get_channels_last_strides(shape), input.dtype(), 0);
        formatted.impl()->set_quantizer(y.impl()->quantizer()); y = formatted;
    }
#ifndef TP_QUANT_CUDA
    if (g.channels_last && g.channel_stride == 1 && g.mode != 2 && g.channels > 0 &&
        input.itemsize() == 1 && g.in_size <= INT32_MAX / 255) {
        const int64_t window_work = g.mode == 1 ? std::max<int64_t>(1, g.in_size / std::max<int64_t>(g.out_size, 1))
            : g.kernel[0] * g.kernel[1] * g.kernel[2];
        parallel::parallel_for(0, y.numel() / g.channels, std::max<int64_t>(1, 32768 / g.channels / window_work),
            [&](int64_t first, int64_t last) {
                cpu::quantized_pool_stub(DeviceType::CPU, source.data_ptr(), y.data_ptr(),
                    first, last, input.dtype(), g, nullptr);
            });
    } else if (g.mode != 2 && g.out_size && g.channels) {
        std::vector<PoolWindow> windows(g.out_size);
        for (int64_t location = 0; location < g.out_size; ++location) {
            auto& window = windows[location];
            int64_t p = location, count = 1, valid = 1;
            float adaptive_inverse = 1.f;
            for (int d = 2; d >= 0; --d) {
                const int64_t o = p % g.out[d]; p /= g.out[d];
                if (g.mode == 1) {
                    window.start[d] = o * g.in[d] / g.out[d];
                    window.end[d] = ((o + 1) * g.in[d] + g.out[d] - 1) / g.out[d];
                } else {
                    window.start[d] = o * g.stride[d] - g.pad[d];
                    window.end[d] = window.start[d] + g.kernel[d];
                    count *= std::min(window.end[d], g.in[d] + g.pad[d]) - window.start[d];
                    window.start[d] = std::max<int64_t>(window.start[d], 0);
                    window.end[d] = std::min(window.end[d], g.in[d]);
                }
                valid *= window.end[d] - window.start[d];
            }
            for (int d = 0; d < 3; ++d)
                adaptive_inverse /= static_cast<float>(window.end[d] - window.start[d]);
            window.initial = uint32_t(0) - static_cast<uint32_t>(g.zp) * static_cast<uint32_t>(valid);
            const int64_t divisor = g.divisor ? g.divisor : (g.mode == 1 || !g.include_pad ? valid : count);
            window.inverse = g.mode == 1 && !g.channels_last ? adaptive_inverse : (divisor ? 1.f / divisor : 0.f);
        }
        parallel::parallel_for(0, y.numel() / g.out_size, 1, [&](int64_t first, int64_t last) {
            cpu::quantized_pool_stub(DeviceType::CPU, source.data_ptr(), y.data_ptr(), first, last,
                input.dtype(), g, windows.data());
        });
    } else
#endif
        codes(input.dtype(), [&]<class T>() { apply(y.numel(), PoolCodes<T>{input.data_ptr<T>(), y.data_ptr<T>(), g}); });
    return y;
}

Tensor avg_pool(const Tensor& input, const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        bool ceil, bool include_pad, std::optional<int64_t> divisor) {
    return pool(input, kernel, stride, padding, {}, ceil, include_pad, divisor, 0);
}
Tensor adaptive_pool(const Tensor& input, const std::vector<int64_t>& size) {
    return pool(input, size, {}, {}, {}, false, false, {}, 1);
}

template<class T>
struct SoftmaxCodes {
    const T* x; T* y; int64_t width, inner, zp; float scale, inverse;
    TP_QN_DEVICE void operator()(int64_t row) const {
        const int64_t base = row / inner * width * inner + row % inner;
        int64_t maximum = std::numeric_limits<T>::lowest();
        for (int64_t c = 0; c < width; ++c) if (x[base + c * inner] > maximum) maximum = x[base + c * inner];
        float sum = 0;
        for (int64_t c = 0; c < width; ++c) sum += ::expf(static_cast<float>(static_cast<int64_t>(x[base + c * inner]) - maximum) * scale);
        for (int64_t c = 0; c < width; ++c) {
            const float v = ::expf(static_cast<float>(static_cast<int64_t>(x[base + c * inner]) - maximum) * scale) / sum;
            y[base + c * inner] = encode<T>(v, inverse, zp);
        }
    }
};

Tensor softmax(const Tensor& input, int64_t dim, double scale, int64_t zp) {
    Guard guard(input); validate(input);
    const int rank = std::max<int64_t>(input.dim(), 1);
    if (dim < 0) dim += rank;
    TP_CHECK(dim >= 0 && dim < rank, "Softmax dimension is invalid");
    Tensor x = input.contiguous(), y = output(input, input.shape(), scale, zp);
    const int64_t width = input.dim() ? input.size(dim) : 1;
    int64_t inner = 1;
    for (int d = dim + 1; d < input.dim(); ++d) inner *= input.size(d);
    codes(input.dtype(), [&]<class T>() {
        if (width) {
#ifdef TP_QUANT_CUDA
            const auto fn = SoftmaxCodes<T>{x.data_ptr<T>(), y.data_ptr<T>(), width, inner, zp,
                static_cast<float>(quantized::q_scale(input)), 1.f / static_cast<float>(scale)};
            apply(input.numel() / width, fn);
#else
            cpu::quantized_softmax_stub(DeviceType::CPU, x.data_ptr(), y.data_ptr(), input.dtype(),
                width, inner, input.numel() / width, zp,
                static_cast<float>(quantized::q_scale(input)), 1.f / static_cast<float>(scale));
#endif
        }
    });
    return y;
}

template<class T>
struct MatmulCodes {
    const T *a, *b; T* y; Broadcast batch; int64_t m, n, k, az, bz, yz; float factor;
    TP_QN_DEVICE void operator()(int64_t i) const {
        const int64_t col = i % n, row = i / n % m;
        int64_t ai, bi; batch.indices(i / (m * n), ai, bi);
        double sum = 0;
        for (int64_t j = 0; j < k; ++j)
            sum += static_cast<double>(static_cast<int64_t>(a[ai * m * k + row * k + j]) - az)
                 * static_cast<double>(static_cast<int64_t>(b[bi * k * n + j * n + col]) - bz);
        y[i] = encode<T>(static_cast<float>(sum) * factor, 1.f, yz);
    }
};

Tensor matmul(const Tensor& a, const Tensor& b, double scale, int64_t zp) {
    Guard guard(a); validate(a); validate(b);
    TP_CHECK(a.dtype() == b.dtype() && a.device() == b.device(), "Matrix operands must share dtype and device");
    TP_CHECK(a.dim() >= 1 && b.dim() >= 1, "Matrix operands must have at least one dimension");
    const int64_t m = a.dim() == 1 ? 1 : a.size(a.dim() - 2), k = a.size(a.dim() - 1), n = b.dim() == 1 ? 1 : b.size(b.dim() - 1);
    TP_CHECK(k == b.size(b.dim() == 1 ? 0 : b.dim() - 2), "Matrix inner dimensions do not match");
    auto ashape=static_cast<std::vector<int64_t>>(a.shape());
    auto bshape=static_cast<std::vector<int64_t>>(b.shape());
    ashape.resize(a.dim()-std::min<int64_t>(a.dim(),2));
    bshape.resize(b.dim()-std::min<int64_t>(b.dim(),2));
    auto shape = broadcast_shape(ashape, bshape);
    Broadcast batch{}; batch.rank = static_cast<int>(shape.size());
    int64_t sa = 1, sb = 1;
    for (int d = batch.rank - 1; d >= 0; --d) {
        batch.dims[d] = shape[d];
        const int ad = d - (batch.rank - ashape.size()), bd = d - (batch.rank - bshape.size());
        if (ad >= 0) { batch.a[d] = ashape[ad] == 1 ? 0 : sa; sa *= ashape[ad]; }
        if (bd >= 0) { batch.b[d] = bshape[bd] == 1 ? 0 : sb; sb *= bshape[bd]; }
    }
    if (a.dim() > 1) shape.push_back(m);
    if (b.dim() > 1) shape.push_back(n);
    Tensor x = a.contiguous(), z = b.contiguous(), y = output(a, shape, scale, zp);
    const float factor = static_cast<float>(quantized::q_scale(a) * quantized::q_scale(b) / scale);
    codes(a.dtype(), [&]<class T>() { apply(y.numel(), MatmulCodes<T>{x.data_ptr<T>(), z.data_ptr<T>(), y.data_ptr<T>(), batch,
        m, n, k, quantized::q_zero_point(a), quantized::q_zero_point(b), zp, factor}); });
    return y;
}

template<class T>
struct PreluCodes {
    const T *x, *w; T* y; Broadcast geometry; int64_t xz, wz, yz, one; float multiplier;
    TP_QN_DEVICE void operator()(int64_t i) const {
        int64_t xi, wi; geometry.indices(i, xi, wi);
        const int64_t a = static_cast<int64_t>(x[xi]) - xz;
        const int64_t b = static_cast<int64_t>(w[wi]) - wz;
        const int32_t value = static_cast<int32_t>((a > 0 ? a : 0) * one + (a < 0 ? a : 0) * b);
        y[xi] = encode<T>(static_cast<float>(value), multiplier, yz);
    }
};

template<class T>
struct FloatPreluCodes {
    const T* x; const float* w; uint8_t* y;
    int64_t inner, channels, input_zp, output_zp;
    float scale, inverse; bool scalar_weight;
    TP_QN_DEVICE void operator()(int64_t i) const {
        const float real = static_cast<float>(static_cast<int64_t>(x[i]) - input_zp) * scale;
        const float value = real > 0.f ? real : real * w[scalar_weight ? 0 : i / inner % channels];
#ifdef TP_QUANT_CUDA
        const float code = ::nearbyintf(__fadd_rn(__fmul_rn(value, inverse), static_cast<float>(output_zp)));
#else
        const float code = ::nearbyintf(value * inverse + static_cast<float>(output_zp));
#endif
        y[i] = static_cast<uint8_t>(::fmaxf(0.f, ::fminf(255.f, code)));
    }
};

Tensor prelu(const Tensor& input, const Tensor& weight, double scale, int64_t zp) {
    Guard guard(input); validate(input);
    if (input.dim() < 1 || input.dim() > 5) {
        TP_CHECK(weight.dtype() == DType::Float32 && input.device() == weight.device(),
                 "Floating slopes must have Float32 dtype and share the input device");
        TP_CHECK(weight.dim() <= 1, "Floating slopes must be scalar or one dimensional");
        const int64_t channels = input.dim() > 1 ? input.size(1) : 1;
        TP_CHECK(weight.numel() == 1 || (input.dim() > 0 && weight.numel() == channels),
                 "Slope count must be one or match the input channels");
        Tensor x = input.contiguous(), w = weight.contiguous();
        Tensor y = Tensor::empty(input.sizes().vec(), DType::QUInt8, input.device());
        y.impl()->set_quantizer(make_per_tensor_affine_quantizer(scale, zp, DType::QUInt8));
        int64_t inner = 1;
        for (int d = 2; d < input.dim(); ++d) inner *= input.size(d);
        const float input_scale = quantized::q_scale(input), inverse = 1.f / static_cast<float>(scale);
        const int64_t input_zp = quantized::q_zero_point(input);
        codes(input.dtype(), [&]<class T>() {
#ifndef TP_QUANT_CUDA
            if (weight.numel() == 1) {
                parallel::parallel_for(0, y.numel(), 32768, [&](int64_t first, int64_t last) {
                    cpu::quantized_float_prelu_stub(DeviceType::CPU, x.data_ptr<T>() + first,
                        w.data_ptr<float>(), y.data_ptr<uint8_t>() + first, last - first, true,
                        input.dtype(), input_zp, input_scale, inverse, zp);
                });
            } else if (inner) {
                parallel::parallel_for(0, y.numel() / inner, 1, [&](int64_t first, int64_t last) {
                    for (int64_t row = first; row < last; ++row)
                        cpu::quantized_float_prelu_stub(DeviceType::CPU, x.data_ptr<T>() + row * inner,
                            w.data_ptr<float>() + row % channels, y.data_ptr<uint8_t>() + row * inner,
                            inner, true, input.dtype(), input_zp, input_scale, inverse, zp);
                });
            }
#else
            apply(y.numel(), FloatPreluCodes<T>{x.data_ptr<T>(), w.data_ptr<float>(), y.data_ptr<uint8_t>(),
                inner, channels, input_zp, zp, input_scale, inverse, weight.numel() == 1});
#endif
        });
        return y;
    }
    validate(weight);
    TP_CHECK(input.dim() > 0, "Parametric rectification requires nonscalar input");
    TP_CHECK(input.dtype() == weight.dtype() && input.device() == weight.device(), "Input and slope must share dtype and device");
    Tensor slope = weight;
    if (input.dim() != weight.dim()) {
        std::vector<int64_t> shape(input.dim(), 1);
        if (input.dim() > 1) shape[1] = weight.numel();
        slope = weight.reshape(shape);
    }
    TP_CHECK(broadcast_shape(input.shape(), slope.shape()) == input.sizes().vec(),
             "Slope dimensions must broadcast to the input shape");
    const auto format = activation_format(input);
    Tensor x = input.contiguous(static_cast<int64_t>(format));
    Tensor w = slope.contiguous(static_cast<int64_t>(format));
    Tensor y = activation_output(input, format);
    y.impl()->set_quantizer(make_per_tensor_affine_quantizer(scale, zp, input.dtype()));
    const float ws = quantized::q_scale(weight), multiplier = static_cast<float>(quantized::q_scale(input)) * ws * (1.f / static_cast<float>(scale));
    const int64_t wz = quantized::q_zero_point(weight), xz = quantized::q_zero_point(input);
    const auto geometry = broadcast(x, w, input.sizes().vec());
    codes(input.dtype(), [&]<class T>() {
        const int64_t one = static_cast<int64_t>(encode<T>(1.f, 1.f / ws, wz)) - wz;
#ifndef TP_QUANT_CUDA
        if (w.numel() == 1 || w.sizes().vec() == input.sizes().vec()) {
            parallel::parallel_for(0, y.numel(), 32768, [&](int64_t first, int64_t last) {
                cpu::quantized_prelu_stub(DeviceType::CPU, x.data_ptr<T>() + first,
                    w.data_ptr<T>() + (w.numel() == 1 ? 0 : first), y.data_ptr<T>() + first,
                    last - first, w.numel() == 1, input.dtype(), xz, wz, zp, one, multiplier);
            });
            return;
        }
        bool channel_weight = input.dim() > 1 && w.size(1) == input.size(1);
        for (int d = 0; d < w.dim(); ++d) if (d != 1 && w.size(d) != 1) channel_weight = false;
        if (channel_weight) {
            if (format != MemoryFormat::Contiguous) {
                const int64_t channels = input.size(1);
                if (channels) parallel::parallel_for(0, y.numel() / channels, std::max<int64_t>(1, 32768 / channels), [&](int64_t first, int64_t last) {
                    for (int64_t row = first; row < last; ++row)
                        cpu::quantized_prelu_stub(DeviceType::CPU, x.data_ptr<T>() + row * channels,
                            w.data_ptr<T>(), y.data_ptr<T>() + row * channels,
                            channels, false, input.dtype(), xz, wz, zp, one, multiplier);
                });
                return;
            }
            int64_t inner = 1;
            for (int d = 2; d < input.dim(); ++d) inner *= input.size(d);
            if (inner) parallel::parallel_for(0, y.numel() / inner, 1, [&](int64_t first, int64_t last) {
                for (int64_t row = first; row < last; ++row)
                    cpu::quantized_prelu_stub(DeviceType::CPU, x.data_ptr<T>() + row * inner,
                        w.data_ptr<T>() + row % input.size(1), y.data_ptr<T>() + row * inner,
                        inner, true, input.dtype(), xz, wz, zp, one, multiplier);
            });
            return;
        }
#endif
        apply(y.numel(), PreluCodes<T>{x.data_ptr<T>(), w.data_ptr<T>(), y.data_ptr<T>(), geometry, xz, wz, zp, one, multiplier});
    });
    return y;
}

template<class T>
struct MomentInnerCodes {
    const T* x; T* y; int64_t count, zp; double scale, correction; bool deviation; int square_lanes;
    TP_QN_DEVICE void operator()(int64_t i) const {
        int64_t sum = 0;
        double square_sum = 0.;
        float partial[16]{};
        for (int64_t j = 0; j < count; ++j) {
            const int64_t code = x[i * count + j];
            sum += code;
            if constexpr (sizeof(T) == 4) {
                const float value = static_cast<float>(code);
                if (j < count - count % square_lanes) partial[j % square_lanes] += value * value;
            } else square_sum += static_cast<double>(code) * code;
        }
        if constexpr (sizeof(T) == 4) {
            float total = 0.f;
            for (int lane = 0; lane < square_lanes; ++lane) total += partial[lane];
            for (int64_t j = count - count % square_lanes; j < count; ++j) {
                const float value = static_cast<float>(x[i * count + j]); total += value * value;
            }
            square_sum = total;
        }
        if (!deviation) {
            y[i] = static_cast<T>(::nearbyintf(static_cast<float>(sum) / count));
            return;
        }
        const double raw_den = static_cast<double>(count) - correction;
        const double den = raw_den < 0. ? 0. : raw_den;
        const double mean = static_cast<double>(sum) / count;
        const double unbiased = static_cast<double>(sum) / den;
        const double raw_variance = square_sum / den - 2. * mean * unbiased + mean * mean * count / den;
        const double variance = raw_variance < 0. ? 0. : raw_variance;
        const float value = static_cast<float>(::sqrt(variance) * scale);
        y[i] = value != value ? (sizeof(T) == 4 ? std::numeric_limits<T>::lowest() : T(0)) :
            encode<T>(value, 1.f / static_cast<float>(scale), zp);
    }
};

Tensor moment_inner(const Tensor& input, const std::vector<int64_t>& dims,
                    bool keepdim, bool deviation, double correction) {
    Guard guard(input); validate(input);
    auto shape = input.sizes().vec();
    int64_t count = 1;
    for (auto d : dims) { count *= input.size(d); shape[d] = 1; }
    if (!keepdim) for (auto d = dims.rbegin(); d != dims.rend(); ++d) shape.erase(shape.begin() + *d);
    Tensor y = output(input, shape, quantized::q_scale(input), quantized::q_zero_point(input));
#ifndef TP_QUANT_CUDA
    const auto* source = static_cast<const uint8_t*>(input.data_ptr());
    auto* destination = static_cast<uint8_t*>(y.data_ptr());
    parallel::parallel_for(0, y.numel(), std::max<int64_t>(1, 32768 / std::max<int64_t>(count, 1)), [&](int64_t first, int64_t last) {
        cpu::quantized_moment_stub(DeviceType::CPU, source + first * count * input.itemsize(),
            destination + first * y.itemsize(), count, last - first, input.dtype(),
            quantized::q_scale(input), quantized::q_zero_point(input), deviation, correction);
    });
#else
    int square_lanes = 1;
#if defined(__x86_64__) || defined(_M_X64)
    const auto capability = cpu::get_cpu_capability();
    if (capability == cpu::CPUCapability::AVX512) square_lanes = 16;
    else if (capability == cpu::CPUCapability::AVX2) square_lanes = 8;
#endif
    codes(input.dtype(), [&]<class T>() {
        apply(y.numel(), MomentInnerCodes<T>{input.data_ptr<T>(), y.data_ptr<T>(), count,
            quantized::q_zero_point(input), quantized::q_scale(input), correction, deviation, square_lanes});
    });
#endif
    return y;
}

struct MomentStrides {
    int reduced_rank = 0, output_rank = 0;
    int64_t reduced_sizes[64]{}, reduced_strides[64]{}, output_sizes[64]{}, output_strides[64]{};
    TP_QN_DEVICE int64_t offset(int64_t group, int64_t element) const {
        int64_t result = 0;
        for (int d = output_rank - 1; d >= 0; --d) {
            result += group % output_sizes[d] * output_strides[d]; group /= output_sizes[d];
        }
        for (int d = reduced_rank - 1; d >= 0; --d) {
            result += element % reduced_sizes[d] * reduced_strides[d]; element /= reduced_sizes[d];
        }
        return result;
    }
};

template<class T>
struct StdStridedCodes {
    const T* x; T* y; MomentStrides strides; int64_t count, zp; double scale, correction;
    TP_QN_DEVICE void operator()(int64_t group) const {
        double mean = 0., m2 = 0.;
        for (int64_t j = 0; j < count; ++j) {
            const float value = static_cast<float>(static_cast<int64_t>(x[strides.offset(group, j)]) - zp) * static_cast<float>(scale);
            const double delta = static_cast<double>(value) - mean;
            mean += delta / (j + 1);
            m2 += delta * (static_cast<double>(value) - mean);
        }
        const double raw_denominator = static_cast<double>(count) - correction;
        const double denominator = raw_denominator < 0. ? 0. : raw_denominator;
        const float value = static_cast<float>(::sqrt(m2 / denominator));
        y[group] = value != value ? (sizeof(T) == 4 ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max()) :
            encode<T>(value, 1.f / static_cast<float>(scale), zp);
    }
};

Tensor std_strided(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim, double correction) {
    Guard guard(input); validate(input);
    TP_CHECK(input.dim() <= 64, "Reduction rank exceeds 64 dimensions");
    MomentStrides strides{};
    auto shape = input.sizes().vec();
    int64_t count = 1;
    for (int d = 0; d < input.dim(); ++d) {
        if (std::find(dims.begin(), dims.end(), d) != dims.end()) {
            strides.reduced_sizes[strides.reduced_rank] = input.size(d);
            strides.reduced_strides[strides.reduced_rank++] = input.stride(d);
            count *= input.size(d); shape[d] = 1;
        } else {
            strides.output_sizes[strides.output_rank] = input.size(d);
            strides.output_strides[strides.output_rank++] = input.stride(d);
        }
    }
    if (!keepdim) for (auto d = dims.rbegin(); d != dims.rend(); ++d)
        if (input.dim()) shape.erase(shape.begin() + *d);
    const double scale = quantized::q_scale(input);
    const int64_t zp = quantized::q_zero_point(input);
    Tensor y = output(input, shape, scale, zp);
    codes(input.dtype(), [&]<class T>() {
        apply(y.numel(), StdStridedCodes<T>{input.data_ptr<T>(), y.data_ptr<T>(), strides, count, zp, scale, correction});
    });
    return y;
}

struct GridSampleCodes {
    const uint8_t* x; const float* grid; uint8_t* y;
    int64_t channels, height, width, out_height, out_width, xs[4], gs[4], zp;
    int padding; bool align;
    TP_QN_DEVICE float coordinate(float v, int64_t size) const {
        v = align ? ((v + 1.f) / 2.f) * (size - 1) : ((v + 1.f) * size - 1.f) / 2.f;
        if (padding == 2) {
            const float low = align ? 0.f : -.5f;
            const float span = align ? size - 1.f : static_cast<float>(size);
            if (span == 0.f) v = 0.f;
            else {
                const float distance = ::fabsf(v - low);
                const float extra = ::fmodf(distance, span);
                const float flips = ::floorf(distance / span);
                v = ::fmodf(flips, 2.f) == 0.f ? extra + low : span - extra + low;
            }
        }
        if (padding) v = ::fminf(size - 1.f, ::fmaxf(v, 0.f));
        #if defined(__CUDA_ARCH__)
        if (!::isfinite(static_cast<double>(v)) || v < -2147483648.f || v > 2147483520.f) v = -100.f;
#else
        if (!std::isfinite(v) || v < -2147483648.f || v > 2147483520.f) v = -100.f;
#endif
        return v;
    }
    TP_QN_DEVICE float sample(int64_t base, int64_t h, int64_t w) const {
        return h >= 0 && h < height && w >= 0 && w < width ? x[base + h * xs[2] + w * xs[3]] : zp;
    }
    TP_QN_DEVICE void operator()(int64_t i) const {
        const int64_t w = i % out_width, h = i / out_width % out_height;
        const int64_t c = i / (out_width * out_height) % channels;
        const int64_t n = i / (out_width * out_height * channels);
        const int64_t gi = n * gs[0] + h * gs[1] + w * gs[2];
        const float ix = coordinate(grid[gi], width), iy = coordinate(grid[gi + gs[3]], height);
        const int64_t left = static_cast<int64_t>(::floorf(ix)), top = static_cast<int64_t>(::floorf(iy));
        const float nw = (left + 1 - ix) * (top + 1 - iy);
        const float ne = (ix - left) * (top + 1 - iy);
        const float sw = (left + 1 - ix) * (iy - top);
        const float se = (ix - left) * (iy - top);
        const int64_t base = n * xs[0] + c * xs[1];
        float result = sample(base, top, left) * nw;
        result += sample(base, top, left + 1) * ne;
        result += sample(base, top + 1, left) * sw;
        result += sample(base, top + 1, left + 1) * se;
        y[i] = static_cast<uint8_t>(::nearbyintf(result));
    }
};

Tensor grid_sample(const Tensor& input, const Tensor& grid, int64_t interpolation,
                   int64_t padding, bool align) {
    Guard guard(input); validate(input);
    TP_CHECK(input.dtype() == DType::QUInt8, "Grid sampling requires unsigned byte quantization");
    TP_CHECK(interpolation == 0, "Quantized grid sampling supports bilinear interpolation only");
    TP_CHECK(padding >= 0 && padding <= 2, "Invalid grid sampling padding mode");
    TP_CHECK(grid.dtype() == DType::Float32 && grid.device() == input.device(),
             "Grid sampling requires a Float32 grid on the input device");
    TP_CHECK(input.dim() == 4 && grid.dim() == 4 && grid.size(0) == input.size(0) && grid.size(3) == 2,
             "Grid sampling requires four-dimensional input and a matching two-coordinate grid");
    TP_CHECK(input.size(2) > 0 && input.size(3) > 0, "Grid sampling requires nonempty input spatial dimensions");
    Tensor y = output(input, {input.size(0), input.size(1), grid.size(1), grid.size(2)},
                      quantized::q_scale(input), quantized::q_zero_point(input));
    GridSampleCodes fn{input.data_ptr<uint8_t>(), grid.data_ptr<float>(), y.data_ptr<uint8_t>(),
                      input.size(1), input.size(2), input.size(3), grid.size(1), grid.size(2),
                      {}, {}, quantized::q_zero_point(input), static_cast<int>(padding), align};
    for (int d = 0; d < 4; ++d) { fn.xs[d] = input.stride(d); fn.gs[d] = grid.stride(d); }
    apply(y.numel(), fn);
    return y;
}

Tensor unary(const Tensor& input, double scale, int64_t zp, int64_t op, double a, double b) {
    TP_CHECK(op >= 0 && op <= 9, "Invalid unary operation");
    if (op==8 || op==9) scale=static_cast<float>(scale);
    return TP_QN_NAME(activation)(input, scale, zp, {static_cast<int>(op), static_cast<float>(a), static_cast<float>(b)});
}

Tensor dropout(const Tensor& input, double, int64_t, double, bool) { validate(input); return input; }

Tensor fp16_prepack(const Tensor& weight) {
    Guard guard(weight);
    TP_CHECK(weight.dim() == 2 && (weight.dtype() == DType::Float32 || weight.dtype() == DType::Float16), "Half weight packing requires a floating matrix");
    Tensor x = weight.contiguous(), y = Tensor::empty(weight.shape(), DType::Float16, weight.device());
    if (weight.dtype() == DType::Float16) { y.copy_(x); return y; }
    const float* src = x.data_ptr<float>(); Half* dst = y.data_ptr<Half>();
    apply(y.numel(), [=] TP_QN_DEVICE (int64_t i) { dst[i] = Half(::fminf(65504.f, ::fmaxf(-65504.f, src[i]))); });
    return y;
}

Tensor fp16_linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias, bool relu) {
    Guard guard(input);
    TP_CHECK(input.dtype() == DType::Float32 && weight.dtype() == DType::Float16 && input.device() == weight.device(), "Half linear requires float input and half weights on the same device");
    TP_CHECK(input.dim() >= 2 && weight.dim() == 2 && input.size(input.dim() - 1) == weight.size(1), "Half linear matrix dimensions do not match");
    const int64_t n = weight.size(0), k = weight.size(1);
    Tensor x = input.contiguous(), w = weight.contiguous(), b;
    if (bias) {
        TP_CHECK(bias->dim() == 1 && bias->numel() == n && bias->device() == input.device(), "Half linear bias must have one value per output channel");
        b = bias->to(DType::Float32).contiguous();
    }
    auto shape = static_cast<std::vector<int64_t>>(input.shape()); shape.back() = n;
    Tensor y = Tensor::empty(shape, DType::Float32, input.device());
    const float* px = x.data_ptr<float>(); const Half* pw = w.data_ptr<Half>();
    const float* pb = bias ? b.data_ptr<float>() : nullptr; float* py = y.data_ptr<float>();
#ifdef TP_QUANT_CUDA
    if (y.numel() && k>0) {
        const int64_t rows=x.numel()/k;
        TP_CHECK(rows<=2147483647 && n<=2147483647 && k<=2147483647, "Half linear matrix dimensions exceed the BLAS index range");
        struct WideWeight { Tensor source,wide; uint32_t version=0; uintptr_t stream=0; };
        thread_local std::array<WideWeight,16> cache;
        thread_local size_t next=0;
        const auto stream=tensorplay::cuda::getCurrentCUDAStream().stream();
        const uintptr_t stream_key=reinterpret_cast<uintptr_t>(stream);
        Tensor wide;
        for (const auto& entry:cache) {
            if (entry.source.defined() && weight.impl()->version_counter().is_enabled() && entry.source.impl()==weight.impl() &&
                entry.version==weight._version() && entry.stream==stream_key) { wide=entry.wide; break; }
        }
        if (!wide.defined()) {
            wide=w.to(DType::Float32);
            if (weight.impl()->version_counter().is_enabled() && wide.numel()<=1024*1024) cache[next++%cache.size()]={weight,wide,weight._version(),stream_key};
        }
        const auto handle=tensorplay::cuda::CUDAContext::getCublasHandle();
        auto checked=[](cublasStatus_t status) {
            TP_CHECK(status==CUBLAS_STATUS_SUCCESS, "Half linear BLAS call failed with status ",static_cast<int>(status));
        };
        struct RestoreMode {
            cublasHandle_t handle; cublasMath_t math=CUBLAS_DEFAULT_MATH; cublasPointerMode_t pointer=CUBLAS_POINTER_MODE_HOST;
            ~RestoreMode() { cublasSetMathMode(handle,math); cublasSetPointerMode(handle,pointer); }
        } restore{handle};
        checked(cublasGetMathMode(handle,&restore.math));
        checked(cublasGetPointerMode(handle,&restore.pointer));
        checked(cublasSetStream(handle,stream));
        checked(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
        checked(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
        const float alpha=1.f,beta=0.f;
        checked(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,rows,k,&alpha,
            wide.data_ptr<float>(),k,px,k,&beta,py,n));
        if (pb || relu) apply(y.numel(), [=] TP_QN_DEVICE (int64_t i) {
            const float value=py[i]+(pb ? pb[i%n] : 0.f);
            py[i]=relu ? ::fmaxf(value,0.f) : value;
        });
    } else {
        apply(y.numel(), [=] TP_QN_DEVICE (int64_t i) {
            const float value=pb ? pb[i%n] : 0.f;
            py[i]=relu ? ::fmaxf(value,0.f) : value;
        });
    }
#else
    parallel::parallel_for(0, y.numel(), std::max<int64_t>(1, 32768 / std::max<int64_t>(k, 1)), [&](int64_t first, int64_t last) {
        tensorplay::cpu::quantized_fp16_linear_stub(DeviceType::CPU, px, pw, pb, py, first, last, n, k, relu);
    });
#endif
    return y;
}

#ifdef TP_QUANT_CUDA
Tensor cat_plain(const std::vector<Tensor>& inputs, int64_t dim, std::optional<double> scale, std::optional<int64_t> zp) {
    return cat(inputs,dim,scale,zp,false);
}
Tensor cat_rectified(const std::vector<Tensor>& inputs, int64_t dim, std::optional<double> scale, std::optional<int64_t> zp) {
    return cat(inputs,dim,scale,zp,true);
}
Tensor max_pool(const Tensor& input,const std::vector<int64_t>& kernel,const std::vector<int64_t>& stride,
        const std::vector<int64_t>& pad,const std::vector<int64_t>& dilation,bool ceil) {
    return pool(input,kernel,stride,pad,dilation,ceil,false,{},2);
}
#endif

#ifndef TP_QUANT_CUDA
Tensor int4_prepack(const Tensor& input, int64_t inner_tiles) {
    TP_CHECK(input.dim()==2 && input.dtype()==DType::Int32, "Int4 packing requires an integer matrix");
    TP_CHECK(input.size(0)%16==0 && input.size(1)%2==0, "Int4 packing requires output channels divisible by sixteen and an even feature count");
    const Tensor x=input.contiguous();
    Tensor y=Tensor::empty({input.size(0),input.size(1)/2},DType::UInt8,input.device());
    cpu::quantized_int4_pack_stub(DeviceType::CPU,x.data_ptr<int32_t>(),y.data_ptr<uint8_t>(),input.size(0),input.size(1));
    return y;
}

Tensor int4_mm(const Tensor& input,const Tensor& weight,int64_t group,const Tensor& parameters) {
    TP_CHECK(input.dim()==2 && (input.dtype()==DType::Float32 || input.dtype()==DType::Float16 || input.dtype()==DType::BFloat16) && input.is_contiguous(),
        "Int4 multiplication requires a contiguous floating matrix");
    TP_CHECK(weight.dim()==2 && weight.dtype()==DType::UInt8 && weight.is_contiguous() && weight.device()==input.device(),
        "Int4 weights must be a contiguous unsigned byte matrix on the input device");
    const int64_t m=input.size(0),n=weight.size(0),k=input.size(1);
    TP_CHECK(weight.size(1)==k/2 && k%2==0 && n%16==0, "Int4 packed weight dimensions do not match the input");
    TP_CHECK((group==32 || group==64 || group==128 || group==256) && k%group==0, "Int4 group size must divide the feature count and be thirty two, sixty four, one hundred twenty eight or two hundred fifty six");
    TP_CHECK(parameters.dim()==3 && parameters.size(0)==k/group && parameters.size(1)==n && parameters.size(2)==2 && parameters.dtype()==input.dtype() && parameters.device()==input.device(),
        "Int4 parameters must contain scale and zero per feature group and output channel");
    const Tensor params=parameters.contiguous();
    Tensor y=Tensor::empty({m,n},input.dtype(),input.device());
    cpu::quantized_int4_mm_stub(DeviceType::CPU,input.data_ptr(),weight.data_ptr<uint8_t>(),params.data_ptr(),y.data_ptr(),input.dtype(),m,n,k,group);
    return y;
}

Tensor int4_mm_tensor_group(const Tensor& input,const Tensor& weight,const Tensor& group,const Tensor& parameters) {
    TP_CHECK(group.numel()==1 && group.dtype()==DType::Int64, "Int4 group size must be one signed sixty four bit integer");
    return int4_mm(input,weight,group.item<int64_t>(),parameters);
}

template <typename scalar_t>
void rowwise_quantize_cpu_impl(const Tensor& x, Tensor& codes, Tensor& scales,
                               float eps, int quant_min, int quant_max) {
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const scalar_t* x_ptr = x.data_ptr<scalar_t>();
    int8_t* codes_ptr = codes.data_ptr<int8_t>();
    float* scales_ptr = scales.data_ptr<float>();
    tensorplay::parallel::parallel_for(0, M, 1, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            float absmax = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                absmax =
                    std::max(absmax, std::fabs(static_cast<float>(x_ptr[r * K + k])));
            }
            const float scale = std::max(absmax, eps) / quant_max;
            scales_ptr[r] = scale;
            for (int64_t k = 0; k < K; ++k) {
                int c = static_cast<int>(
                    std::nearbyint(static_cast<float>(x_ptr[r * K + k]) / scale));
                c = c < quant_min ? quant_min : (c > quant_max ? quant_max : c);
                codes_ptr[r * K + k] = static_cast<int8_t>(c);
            }
        }
    });
}

void launch_rowwise_quantize_cpu(const Tensor& x, Tensor& codes, Tensor& scales,
                                 float eps, int quant_min, int quant_max) {
    switch (x.dtype()) {
        case DType::Float16:
            rowwise_quantize_cpu_impl<tensorplay::Half>(x, codes, scales, eps,
                                                         quant_min, quant_max);
            break;
        case DType::BFloat16:
            rowwise_quantize_cpu_impl<tensorplay::BFloat16>(
                x, codes, scales, eps, quant_min, quant_max);
            break;
        case DType::Float32:
            rowwise_quantize_cpu_impl<float>(x, codes, scales, eps, quant_min,
                                       quant_max);
            break;
        default:
            TP_THROW(TypeError, "rowwise quantize: unsupported input dtype");
    }
}

std::tuple<Tensor, Tensor> _quantize_int8_rowwise_cpu(const Tensor& self,
                                                       double eps) {
    TP_CHECK(self.dim() == 2, "_quantize_int8_rowwise(): expected a 2-D tensor");
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const Tensor x = self.contiguous();
    Tensor codes = Tensor::empty({M, K}, DType::Int8, self.device());
    Tensor scales = Tensor::empty({M}, DType::Float32, self.device());
    if (M == 0 || K == 0) {
        return {codes, scales};
    }
    launch_rowwise_quantize_cpu(x, codes, scales, static_cast<float>(eps), -127,
                                127);
    return {codes, scales};
}

std::tuple<Tensor, Tensor> _quantize_int4_rowwise_cpu(const Tensor& self,
                                                     double eps) {
    TP_CHECK(self.dim() == 2, "_quantize_int4_rowwise(): expected a 2-D tensor");
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const Tensor x = self.contiguous();
    Tensor codes = Tensor::empty({M, K}, DType::Int8, self.device());
    Tensor scales = Tensor::empty({M}, DType::Float32, self.device());
    if (M == 0 || K == 0) {
        return {codes, scales};
    }
    launch_rowwise_quantize_cpu(x, codes, scales, static_cast<float>(eps), -8, 7);
    return {codes, scales};
}

Tensor _int8_mm_rescale_cpu(const Tensor& out, const Tensor& row_scale,
                             const Tensor& col_scale) {
    TP_CHECK(out.dim() == 2 && out.dtype() == DType::Int32,
             "_int8_mm_rescale(): expected a 2-D int32 output");
    const int64_t M = out.size(0);
    const int64_t N = out.size(1);
    Tensor result = Tensor::empty({M, N}, DType::Float32, out.device());
    if (M == 0 || N == 0) {
        return result;
    }
    const int32_t* out_ptr = out.data_ptr<int32_t>();
    const float* rs = row_scale.data_ptr<float>();
    const float* cs = col_scale.data_ptr<float>();
    float* res = result.data_ptr<float>();
    tensorplay::parallel::parallel_for(0, M, 1, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const float sr = rs[r];
            for (int64_t c = 0; c < N; ++c) {
                res[r * N + c] =
                    static_cast<float>(out_ptr[r * N + c]) * sr * cs[c];
            }
        }
    });
    return result;
}
#endif

} // namespace TP_QN_DETAIL
} // namespace tensorplay::quant
#include "QuantEmbeddingImpl.h"
#include "QuantConvolutionImpl.h"
#include "QuantRecurrentImpl.h"
#include "QuantResizeImpl.h"
#include "QuantBatchNormImpl.h"
#include "QuantWrappedImpl.h"
namespace tensorplay::quant {

std::tuple<Tensor,Tensor,Tensor> TP_QN_NAME(recurrent_packed)(const Tensor& input,
        const Tensor& hidden,const std::optional<Tensor>& cell,const std::vector<Tensor>& weights,
        const std::vector<Tensor>& biases,const std::vector<Tensor>& storage,const std::vector<bool>& ranges,
        int64_t layers,double dropout,bool training,bool bidirectional,bool batch_first,
        int64_t mode,const std::optional<Tensor>& batch_sizes) {
    return TP_QN_DETAIL::recurrent_impl(input,hidden,cell,weights,biases,layers,dropout,
        training,bidirectional,batch_first,mode,false,batch_sizes,storage,ranges);
}

Tensor TP_QN_NAME(batch_norm)(const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, const Tensor& mean, const Tensor& var,
        double eps, double scale, int64_t zp, bool relu) {
    return TP_QN_DETAIL::batch_norm(input, weight, bias, mean, var, eps, scale, zp, relu);
}

Tensor TP_QN_NAME(resize)(const Tensor& input, const std::vector<int64_t>& sizes,
        const std::vector<std::optional<double>>& scales, bool exact, bool bilinear, bool align) {
    return TP_QN_DETAIL::resize(input, sizes, scales, exact, bilinear, align);
}

Tensor TP_QN_NAME(binary)(const Tensor& a, const Tensor& b, double as, int64_t az, double bs, int64_t bz,
                         double ys, int64_t yz, int op, bool relu) {
    return TP_QN_DETAIL::binary(a, b, as, az, bs, bz, ys, yz, op, relu);
}
Tensor TP_QN_NAME(cat)(const std::vector<Tensor>& xs, int64_t dim, std::optional<double> scale,
                      std::optional<int64_t> zp, bool relu) {
    return TP_QN_DETAIL::cat(xs, dim, scale, zp, relu);
}
Tensor TP_QN_NAME(std_strided)(const Tensor& x, const std::vector<int64_t>& dims,
        bool keepdim, double correction) {
    return TP_QN_DETAIL::std_strided(x, dims, keepdim, correction);
}
Tensor TP_QN_NAME(moment_inner)(const Tensor& x, const std::vector<int64_t>& dims,
        bool keepdim, bool deviation, double correction) {
    return TP_QN_DETAIL::moment_inner(x, dims, keepdim, deviation, correction);
}
Tensor TP_QN_NAME(grid_sample)(const Tensor& x, const Tensor& grid, int64_t interpolation,
        int64_t padding, bool align) {
    return TP_QN_DETAIL::grid_sample(x, grid, interpolation, padding, align);
}
Tensor TP_QN_NAME(pool)(const Tensor& x, const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& pad,
        const std::vector<int64_t>& dilation, bool ceil, bool include_pad, std::optional<int64_t> divisor, int mode) {
    return TP_QN_DETAIL::pool(x, kernel, stride, pad, dilation, ceil, include_pad, divisor, mode);
}

#ifdef TP_QUANT_CUDA
TENSORPLAY_LIBRARY_IMPL(CUDA, QuantNative) {
    m.impl("quantized_cat", TP_QN_DETAIL::cat_plain);
    m.impl("quantized_cat_relu", TP_QN_DETAIL::cat_rectified);
    m.impl("quantized_max_pool1d", TP_QN_DETAIL::max_pool);
    m.impl("quantized_max_pool3d", TP_QN_DETAIL::max_pool);
#else
TENSORPLAY_LIBRARY_IMPL(CPU, QuantNative) {
    m.impl("fbgemm_pack_quantized_matrix.KN", TP_QN_DETAIL::legacy_integer_pack_sizes);
    m.impl("fbgemm_pack_gemm_matrix_fp16", TP_QN_DETAIL::wrapped_half_prepack);
    m.impl("fbgemm_linear_fp16_weight_fp32_activation", TP_QN_DETAIL::legacy_half_linear);
    m.impl("fbgemm_linear_fp16_weight_fp32_activation.out", TP_QN_DETAIL::legacy_half_linear_out);
    m.impl("fbgemm_linear_fp16_weight", TP_QN_DETAIL::legacy_half_linear_bias);
    m.impl("fbgemm_linear_fp16_weight.out", TP_QN_DETAIL::legacy_half_linear_bias_out);
    m.impl("fbgemm_pack_quantized_matrix", TP_QN_DETAIL::legacy_integer_pack);
    m.impl("fbgemm_linear_quantize_weight", TP_QN_DETAIL::legacy_quantize_weight);
    m.impl("fbgemm_linear_int8_weight_fp32_activation", TP_QN_DETAIL::legacy_integer_linear);
    m.impl("fbgemm_linear_int8_weight", TP_QN_DETAIL::legacy_integer_linear);
    m.impl("quantized_lstm_cell", TP_QN_DETAIL::legacy_lstm_cell);
    m.impl("quantized_gru_cell", TP_QN_DETAIL::legacy_gru_cell);
    m.impl("quantized_rnn_relu_cell", TP_QN_DETAIL::legacy_relu_cell);
    m.impl("quantized_rnn_tanh_cell", TP_QN_DETAIL::legacy_tanh_cell);
    m.impl("_wrapped_linear_prepack", TP_QN_DETAIL::wrapped_prepack);
    m.impl("_wrapped_quantized_linear_prepacked", TP_QN_DETAIL::wrapped_linear_prepacked);
    m.impl("_quantized_wrapped_linear", TP_QN_DETAIL::wrapped_linear);
    m.impl("_quantized_wrapped_half_prepack", TP_QN_DETAIL::wrapped_half_prepack);
    m.impl("_quantized_wrapped_half_linear", TP_QN_DETAIL::wrapped_half_linear);
    m.impl("_convert_weight_to_int4pack_for_cpu", TP_QN_DETAIL::int4_prepack);
    m.impl("_convert_weight_to_int4pack", TP_QN_DETAIL::int4_prepack);
    m.impl("_weight_int4pack_mm_for_cpu", TP_QN_DETAIL::int4_mm);
    m.impl("_quantized_int4_mm", TP_QN_DETAIL::int4_mm_tensor_group);
    m.impl("_quantize_int8_rowwise", TP_QN_DETAIL::_quantize_int8_rowwise_cpu);
    m.impl("_quantize_int4_rowwise", TP_QN_DETAIL::_quantize_int4_rowwise_cpu);
    m.impl("_int8_mm_rescale", TP_QN_DETAIL::_int8_mm_rescale_cpu);
#endif
    m.impl("_quantized_recurrent_step", TP_QN_DETAIL::recurrent_step_checked);
    m.impl("_quantized_recurrent_cell", TP_QN_DETAIL::recurrent_cell);
    m.impl("_quantized_recurrent", TP_QN_DETAIL::recurrent);
    m.impl("quantized_convolution", TP_QN_DETAIL::convolution);
    m.impl("_quantized_rowwise_pack", TP_QN_DETAIL::rowwise_pack);
    m.impl("_quantized_rowwise_unpack", TP_QN_DETAIL::rowwise_unpack);
    m.impl("_quantized_embedding_weight_pack", TP_QN_DETAIL::embedding_weight_pack);
    m.impl("_quantized_embedding", TP_QN_DETAIL::embedding);
    m.impl("_quantized_embedding_bag", TP_QN_DETAIL::embedding_bag);
    m.impl("quantized_unary", TP_QN_DETAIL::unary);
    m.impl("quantized_binary", TP_QN_DETAIL::binary_op);
    m.impl("quantized_scalar", TP_QN_DETAIL::scalar);
    m.impl("quantized_scalar_out", TP_QN_DETAIL::scalar_out);
    m.impl("quantized_softmax", TP_QN_DETAIL::softmax);
    m.impl("quantized_matmul", TP_QN_DETAIL::matmul);
    m.impl("quantized_prelu", TP_QN_DETAIL::prelu);
    m.impl("quantized_avg_pool", TP_QN_DETAIL::avg_pool);
    m.impl("quantized_adaptive_avg_pool", TP_QN_DETAIL::adaptive_pool);
    m.impl("quantized_dropout", TP_QN_DETAIL::dropout);
    m.impl("_quantized_fp16_prepack", TP_QN_DETAIL::fp16_prepack);
    m.impl("_quantized_fp16_linear", TP_QN_DETAIL::fp16_linear);
}
} // namespace tensorplay::quant
#undef TP_QN_DEVICE
#undef TP_QN_DETAIL
#undef TP_QN_NAME
