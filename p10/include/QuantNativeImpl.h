#pragma once
#include "QuantNative.h"
#include "Dispatcher.h"
#include "Parallel.h"
#include "Half.h"
#include <algorithm>
#include <array>
#include <numeric>
#ifdef TP_QUANT_CUDA
#include "CUDARuntime.h"
#include <cuda_runtime.h>
#define TP_QN_DEVICE __device__
#define TP_QN_DETAIL cuda_native
#define TP_QN_NAME(x) x##_cuda
#else
#include "cpu/QuantizedElementwiseKernels.h"
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
        if (ad >= 0) { g.a[d] = a.size(ad) == 1 ? 0 : sa; sa *= a.size(ad); }
        if (bd >= 0) { g.b[d] = b.size(bd) == 1 ? 0 : sb; sb *= b.size(bd); }
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
    int64_t inner = 1, offset = 0;
    for (int d = dim + 1; d < first.dim(); ++d) inner *= first.size(d);
    for (const auto& src : inputs) {
        Tensor x = src.contiguous();
        codes(src.dtype(), [&]<class T>() {
            apply(x.numel(), CatCodes<T>{x.data_ptr<T>(), y.data_ptr<T>(), src.size(dim), inner,
                shape[dim], offset, quantized::q_zero_point(src), yz,
                static_cast<float>(quantized::q_scale(src)), 1.f / static_cast<float>(ys), relu});
        });
        offset += src.size(dim);
    }
    return y;
}

struct PoolGeometry {
    int rank, mode; int64_t in[3]{1,1,1}, out[3]{1,1,1}, kernel[3]{1,1,1}, stride[3]{1,1,1}, pad[3]{}, dilation[3]{1,1,1};
    int64_t in_size = 1, out_size = 1, zp, divisor; bool include_pad;
};

template<class T>
struct PoolCodes {
    const T* x; T* y; PoolGeometry g;
    TP_QN_DEVICE void operator()(int64_t i) const {
        int64_t start[3]{}, end[3]{1,1,1}, count = 1, valid = 1, p = i % g.out_size;
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
        const T* row = x + i / g.out_size * g.in_size;
        int64_t sum = 0, maximum = std::numeric_limits<T>::lowest();
        for (int64_t a = start[0]; a < end[0]; a += g.dilation[0])
            for (int64_t b = start[1]; b < end[1]; b += g.dilation[1])
                for (int64_t c = start[2]; c < end[2]; c += g.dilation[2]) {
                    const int64_t code = row[(a * g.in[1] + b) * g.in[2] + c];
                    if (code > maximum) maximum = code;
                    sum += code - g.zp;
                }
        if (g.mode == 2) y[i] = static_cast<T>(maximum);
        else {
            const int64_t divisor = g.divisor ? g.divisor : (g.mode == 1 || !g.include_pad ? valid : count);
            const double code = divisor ? ::nearbyintf(static_cast<float>(sum) * (1.f / divisor) + static_cast<float>(g.zp)) : g.zp;
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
    Tensor x = input.contiguous(), y = output(input, shape, quantized::q_scale(input), g.zp);
    codes(input.dtype(), [&]<class T>() { apply(y.numel(), PoolCodes<T>{x.data_ptr<T>(), y.data_ptr<T>(), g}); });
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
        if (width) apply(input.numel() / width, SoftmaxCodes<T>{x.data_ptr<T>(), y.data_ptr<T>(), width, inner, zp,
            static_cast<float>(quantized::q_scale(input)), 1.f / static_cast<float>(scale)});
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
    std::vector<int64_t> ashape(a.shape().begin(), a.shape().end() - std::min<int64_t>(a.dim(), 2));
    std::vector<int64_t> bshape(b.shape().begin(), b.shape().end() - std::min<int64_t>(b.dim(), 2));
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
    const T *x, *w; T* y; int64_t channels, inner, weights, xz, wz, yz, one; float multiplier;
    TP_QN_DEVICE void operator()(int64_t i) const {
        const int64_t a = static_cast<int64_t>(x[i]) - xz;
        const int64_t b = static_cast<int64_t>(w[weights == 1 ? 0 : i / inner % channels]) - wz;
        const double v = static_cast<double>(a > 0 ? a : 0) * one + static_cast<double>(a < 0 ? a : 0) * b;
        y[i] = encode<T>(static_cast<float>(v), multiplier, yz);
    }
};

Tensor prelu(const Tensor& input, const Tensor& weight, double scale, int64_t zp) {
    Guard guard(input); validate(input); validate(weight);
    TP_CHECK(input.dim() > 0 && weight.dim() <= 1, "Parametric rectification requires nonscalar input and scalar or vector weight");
    TP_CHECK(input.dtype() == weight.dtype() && input.device() == weight.device(), "Input and slope must share dtype and device");
    const int64_t channels = input.dim() > 1 ? input.size(1) : 1;
    TP_CHECK(weight.numel() == 1 || weight.numel() == channels, "Slope count must be one or equal the channel count");
    int64_t inner = 1;
    for (int d = 2; d < input.dim(); ++d) inner *= input.size(d);
    if (input.dim() == 1) inner = input.numel();
    Tensor x = input.contiguous(), w = weight.contiguous(), y = output(input, input.shape(), scale, zp);
    const float ws = quantized::q_scale(weight), multiplier = static_cast<float>(quantized::q_scale(input)) * ws / static_cast<float>(scale);
    const int64_t wz = quantized::q_zero_point(weight);
    codes(input.dtype(), [&]<class T>() { apply(y.numel(), PreluCodes<T>{x.data_ptr<T>(), w.data_ptr<T>(), y.data_ptr<T>(), channels, inner,
        weight.numel(), quantized::q_zero_point(input), wz, zp, static_cast<int64_t>(encode<T>(1.f, 1.f / ws, wz)) - wz, multiplier}); });
    return y;
}

Tensor unary(const Tensor& input, double scale, int64_t zp, int64_t op, double a, double b) {
    TP_CHECK(op >= 0 && op <= 9, "Invalid unary operation");
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
    apply(y.numel(), [=] TP_QN_DEVICE (int64_t i) {
        float sum = 0;
        for (int64_t j = 0; j < k; ++j) sum += px[i / n * k + j] * static_cast<float>(pw[i % n * k + j]);
        if (pb) sum += pb[i % n];
        py[i] = relu ? ::fmaxf(sum, 0.f) : sum;
    });
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

} // namespace TP_QN_DETAIL
} // namespace tensorplay::quant
#include "QuantEmbeddingImpl.h"
#include "QuantConvolutionImpl.h"
#include "QuantRecurrentImpl.h"
#include "QuantResizeImpl.h"
#include "QuantBatchNormImpl.h"
namespace tensorplay::quant {

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
#endif
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
