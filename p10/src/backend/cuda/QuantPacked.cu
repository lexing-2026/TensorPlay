#include "QuantPacked.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "QuantElementwise.h"
#include <cuda_runtime.h>

namespace tensorplay::cuda {
namespace {

template<class T>
__global__ void activation_codes(const T* input, T* output, int64_t size,
        float scale, int64_t zp, float inverse_scale, int64_t output_zp,
        quant::Activation fn) {
    __shared__ T table[256];
    if constexpr (sizeof(T) == 1) {
        const int code = std::is_same_v<T, int8_t>
            ? static_cast<int8_t>(threadIdx.x) : threadIdx.x;
        table[threadIdx.x] = quant::encode<T>(fn(static_cast<float>(code - zp) * scale), inverse_scale, output_zp);
        __syncthreads();
    }
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        if constexpr (sizeof(T) == 1) output[i] = table[static_cast<uint8_t>(input[i])];
        else output[i] = quant::encode<T>(fn(static_cast<float>(static_cast<int64_t>(input[i]) - zp) * scale), inverse_scale, output_zp);
    }
}

template<class T>
__global__ void relu6_codes(const T* input, T* output, int64_t size, int64_t zp, T hi) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        output[i] = static_cast<T>(min(static_cast<int64_t>(hi), max(zp, static_cast<int64_t>(input[i]))));
}

Tensor activation(const Tensor& input, double scale, int64_t zp, quant::Activation fn) {
    CUDAGuard guard(input.device());
    auto grid = make_per_tensor_affine_quantizer(scale, zp, input.dtype());
    const float xs = static_cast<float>(quantized::q_scale(input));
    const int64_t xz = quantized::q_zero_point(input);
    const Tensor x = input.contiguous();
    Tensor out = Tensor::empty(input.shape(), input.dtype(), input.device());
    if (input.numel()) {
        const int blocks = static_cast<int>(std::min<int64_t>((input.numel() + 4095) / 4096, 4096));
        const auto stream = getCurrentCUDAStream().stream();
        const float inverse_scale = 1.f / static_cast<float>(scale);
        switch (input.dtype()) {
            case DType::QInt8: activation_codes<<<blocks,256,0,stream>>>(x.data_ptr<int8_t>(),out.data_ptr<int8_t>(),input.numel(),xs,xz,inverse_scale,zp,fn); break;
            case DType::QUInt8: activation_codes<<<blocks,256,0,stream>>>(x.data_ptr<uint8_t>(),out.data_ptr<uint8_t>(),input.numel(),xs,xz,inverse_scale,zp,fn); break;
            case DType::QInt32: activation_codes<<<blocks,256,0,stream>>>(x.data_ptr<int32_t>(),out.data_ptr<int32_t>(),input.numel(),xs,xz,inverse_scale,zp,fn); break;
            default: TP_THROW(TypeError, "Activation requires quantized integer input");
        }
        checkCuda(cudaGetLastError(), "Integer code activation");
    }
    out.impl()->set_quantizer(std::move(grid));
    return out;
}

template<class T>
__global__ void relu_codes(const T* input, T* output, int64_t size, int64_t zp) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        output[i] = static_cast<T>(static_cast<int64_t>(input[i]) < zp ? zp : input[i]);
    }
}

__global__ void pack_codes(const int8_t* weight, int8_t* packed, int64_t* offsets,
                           int64_t N, int64_t K, int64_t padded_K) {
    const int64_t n = blockIdx.x;
    int64_t sum = 0;
    for (int64_t k = threadIdx.x; k < padded_K; k += blockDim.x) {
        const int8_t code = k < K ? weight[n * K + k] : 0;
        packed[n * padded_K + k] = code;
        sum += code;
    }
    __shared__ int64_t partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step) partial[threadIdx.x] += partial[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) offsets[n] = partial[0];
}

template<class X>
__global__ void make_columns(const X* input, int8_t* columns,
                              quant::PackedDeviceGeometry g, int64_t xz, int shift) {
    const int64_t total = g.groups * g.rows * g.padded_K;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t k = i % g.padded_K, row = (i / g.padded_K) % g.rows;
        if (k >= g.K) { columns[i] = 0; continue; }
        const int64_t group = i / (g.padded_K * g.rows);
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
        const int code = inside ? input[((row / g.spatial) * g.channels + channel) * g.input_spatial + index] : xz;
        columns[i] = static_cast<int8_t>(code - shift);
    }
}

__device__ int dot4(int a, int b, int acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, acc);
#else
    for (int s = 0; s < 32; s += 8) acc += static_cast<int8_t>(a >> s) * static_cast<int8_t>(b >> s);
    return acc;
#endif
}

template<class X, class Y>
__global__ void packed_gemm(const X* input, const int8_t* weight,
                            const int64_t* offsets, const float* scales, const int64_t* zps,
                            const float* bias, Y* out, quant::PackedDeviceGeometry g,
                            double xs, int64_t xz, double ys, int64_t yz,
                            int shift, bool relu, bool output_float) {
    __shared__ int8_t xtile[16][32];
    __shared__ int8_t wtile[16][32];
    const int64_t n_per_group = g.outputs / g.groups;
    const int64_t channel_tiles = (n_per_group + 15) / 16;
    const int64_t row_tiles = (g.rows + 15) / 16;
    const int64_t channel_tile = blockIdx.x % channel_tiles;
    const int64_t row_tile = (blockIdx.x / channel_tiles) % row_tiles;
    const int64_t group = blockIdx.x / (channel_tiles * row_tiles);
    const int64_t row = row_tile * 16 + threadIdx.y;
    const int64_t local_n = channel_tile * 16 + threadIdx.x;
    const int64_t n = group * n_per_group + local_n;
    int64_t acc = 0, row_sum = 0;
    for (int64_t k0 = 0; k0 < g.padded_K; k0 += 32) {
        for (int j = 0; j < 2; ++j) {
            const int64_t k = k0 + threadIdx.x + j * 16;
            const int64_t xindex = g.linear ? row * g.K + k : (group * g.rows + row) * g.padded_K + k;
            xtile[threadIdx.y][threadIdx.x + j * 16] = (row < g.rows && k < g.K)
                ? static_cast<int8_t>(static_cast<int>(input[xindex]) - shift) : 0;
            const int64_t wn = group * n_per_group + channel_tile * 16 + threadIdx.y;
            wtile[threadIdx.y][threadIdx.x + j * 16] = (wn < (group + 1) * n_per_group)
                ? weight[wn * g.padded_K + k] : 0;
        }
        __syncthreads();
        int part = 0, sum = 0;
#pragma unroll
        for (int k = 0; k < 32; k += 4) {
            const int a = *reinterpret_cast<const int*>(&xtile[threadIdx.y][k]);
            const int b = *reinterpret_cast<const int*>(&wtile[threadIdx.x][k]);
            part = dot4(a, b, part);
            sum = dot4(a, 0x01010101, sum);
        }
        acc += part; row_sum += sum;
        __syncthreads();
    }
    if (row >= g.rows || local_n >= n_per_group) return;
    acc = acc - xz * offsets[n] - zps[n] * row_sum + g.K * xz * zps[n];
    double value = static_cast<double>(acc) * xs * scales[n] + (bias ? bias[n] : 0.);
    if (relu) value = fmax(value, 0.);
    const int64_t index = g.linear ? row * g.outputs + n : ((row / g.spatial) * g.outputs + n) * g.spatial + row % g.spatial;
    if (output_float) out[index] = static_cast<Y>(value);
    else {
        const double lo = std::is_same_v<Y, uint8_t> ? 0. : -128.;
        out[index] = static_cast<Y>(fmin(lo + 255., fmax(lo, nearbyint(value / ys) + yz)));
    }
}

template<class X, class Y>
void launch(const Tensor& x, const Tensor& columns, const Tensor& packed,
            const Tensor& offsets, const Tensor& scales, const Tensor& zps,
            const std::optional<Tensor>& bias, Tensor& out,
            const quant::PackedGeometry& g, double ys, int64_t yz, bool relu, bool output_float) {
    const int shift = x.dtype() == DType::QUInt8 ? 128 : 0;
    const int64_t xz = quantized::q_zero_point(x) - shift;
    const double xs = quantized::q_scale(x);
    const int64_t tiles = ((g.outputs / g.groups + 15) / 16) * ((g.rows + 15) / 16) * g.groups;
    TP_CHECK(tiles <= 2147483647, "Packed output exceeds the supported launch size");
    const dim3 block(16, 16), grid(static_cast<unsigned int>(tiles));
    const auto stream = getCurrentCUDAStream().stream();
    if (g.linear) packed_gemm<X,Y><<<grid, block, 0, stream>>>(x.data_ptr<X>(), packed.data_ptr<int8_t>(), offsets.data_ptr<int64_t>(), scales.data_ptr<float>(), zps.data_ptr<int64_t>(), bias ? bias->data_ptr<float>() : nullptr, out.data_ptr<Y>(), g, xs, xz, ys, yz, shift, relu, output_float);
    else packed_gemm<int8_t,Y><<<grid, block, 0, stream>>>(columns.data_ptr<int8_t>(), packed.data_ptr<int8_t>(), offsets.data_ptr<int64_t>(), scales.data_ptr<float>(), zps.data_ptr<int64_t>(), bias ? bias->data_ptr<float>() : nullptr, out.data_ptr<Y>(), g, xs, xz, ys, yz, 0, relu, output_float);
}
} // namespace

Tensor quantized_relu_cuda(const Tensor& input) {
    CUDAGuard guard(input.device());
    const int64_t zp = quantized::q_zero_point(input);
    const Tensor x = input.contiguous();
    Tensor out = Tensor::empty(input.shape(), input.dtype(), input.device());
    if (input.numel()) {
        const int blocks = static_cast<int>(std::min<int64_t>((input.numel() + 255) / 256, 65535));
        const auto stream = getCurrentCUDAStream().stream();
        switch (input.dtype()) {
            case DType::QInt8: relu_codes<<<blocks,256,0,stream>>>(x.data_ptr<int8_t>(),out.data_ptr<int8_t>(),input.numel(),zp); break;
            case DType::QUInt8: relu_codes<<<blocks,256,0,stream>>>(x.data_ptr<uint8_t>(),out.data_ptr<uint8_t>(),input.numel(),zp); break;
            case DType::QInt32: relu_codes<<<blocks,256,0,stream>>>(x.data_ptr<int32_t>(),out.data_ptr<int32_t>(),input.numel(),zp); break;
            default: TP_THROW(TypeError, "Rectification requires quantized integer input");
        }
        checkCuda(cudaGetLastError(), "Integer rectification");
    }
    out.impl()->set_quantizer(input.impl()->quantizer());
    return out;
}

Tensor quantized_relu6_cuda(const Tensor& input) {
    CUDAGuard guard(input.device());
    const int64_t zp = quantized::q_zero_point(input);
    const float inverse_scale = 1.f / static_cast<float>(quantized::q_scale(input));
    const Tensor x = input.contiguous();
    Tensor out = Tensor::empty(input.shape(), input.dtype(), input.device());
    if (input.numel()) {
        const int blocks = static_cast<int>(std::min<int64_t>((input.numel() + 255) / 256, 65535));
        const auto stream = getCurrentCUDAStream().stream();
        switch (input.dtype()) {
            case DType::QInt8: relu6_codes<<<blocks,256,0,stream>>>(x.data_ptr<int8_t>(),out.data_ptr<int8_t>(),input.numel(),zp,quant::encode<int8_t>(6.f,inverse_scale,zp)); break;
            case DType::QUInt8: relu6_codes<<<blocks,256,0,stream>>>(x.data_ptr<uint8_t>(),out.data_ptr<uint8_t>(),input.numel(),zp,quant::encode<uint8_t>(6.f,inverse_scale,zp)); break;
            case DType::QInt32: relu6_codes<<<blocks,256,0,stream>>>(x.data_ptr<int32_t>(),out.data_ptr<int32_t>(),input.numel(),zp,quant::encode<int32_t>(6.f,inverse_scale,zp)); break;
            default: TP_THROW(TypeError, "Rectification requires quantized integer input");
        }
        checkCuda(cudaGetLastError(), "Integer bounded rectification");
    }
    out.impl()->set_quantizer(input.impl()->quantizer());
    return out;
}

Tensor quantized_leaky_relu_cuda(const Tensor& x, double slope, double scale, int64_t zp) {
    return activation(x, scale, zp, {0, static_cast<float>(slope)});
}
Tensor quantized_elu_cuda(const Tensor& x, double output_scale, int64_t zp,
                          double alpha, double scale, double input_scale) {
    return activation(x, output_scale, zp, {1, 0, static_cast<float>(alpha),
        static_cast<float>(scale), static_cast<float>(input_scale)});
}
Tensor quantized_hardswish_cuda(const Tensor& x, double scale, int64_t zp) {
    return activation(x, scale, zp, {2});
}
Tensor quantized_hardsigmoid_cuda(const Tensor& x, double scale, int64_t zp) {
    return activation(x, scale, zp, {3});
}
Tensor quantized_sigmoid_cuda(const Tensor& x, double scale, int64_t zp) {
    return activation(x, scale, zp, {4});
}
Tensor quantized_tanh_cuda(const Tensor& x, double scale, int64_t zp) {
    return activation(x, scale, zp, {5});
}

std::tuple<Tensor, Tensor> _quantized_weight_prepack_cuda(const Tensor& weight) {
    CUDAGuard guard(weight.device());
    const auto [K, padded_K] = quant::packed_weight_shape(weight);
    const Tensor w = weight.contiguous();
    Tensor packed = Tensor::empty({weight.size(0), padded_K}, DType::Int8, weight.device());
    Tensor offsets = Tensor::empty({weight.size(0)}, DType::Int64, weight.device());
    if (weight.size(0)) pack_codes<<<weight.size(0), 256, 0, getCurrentCUDAStream().stream()>>>(w.data_ptr<int8_t>(), packed.data_ptr<int8_t>(), offsets.data_ptr<int64_t>(), weight.size(0), K, padded_K);
    checkCuda(cudaGetLastError(), "Integer weight packing");
    return {packed, offsets};
}

Tensor _quantized_packed_run_cuda(
    const Tensor& input, const Tensor& packed, const Tensor& offsets,
    const Tensor& scales, const Tensor& zero_points, const std::optional<Tensor>& bias,
    const std::vector<int64_t>& weight_sizes, double output_scale, int64_t output_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups, bool relu, bool output_float) {
    CUDAGuard guard(input.device());
    const auto g = quant::packed_geometry(input, packed, offsets, scales, zero_points, bias,
        weight_sizes, stride, padding, dilation, groups, output_scale, output_zero_point, output_float);
    const Tensor x = input.contiguous();
    const bool unsigned_input = input.dtype() == DType::QUInt8;
    Tensor out = Tensor::empty(g.shape, output_float ? DType::Float32 : (unsigned_input ? DType::UInt8 : DType::Int8), input.device());
    Tensor columns = Tensor::empty({0}, DType::Int8, input.device());
    if (out.numel()) {
        if (!g.linear && g.K) {
            columns = Tensor::empty({g.groups, g.rows, g.padded_K}, DType::Int8, input.device());
            const int blocks = static_cast<int>(std::min<int64_t>((columns.numel() + 255) / 256, 65535));
            const auto stream = getCurrentCUDAStream().stream();
            const int64_t xz = quantized::q_zero_point(input);
            if (unsigned_input) make_columns<uint8_t><<<blocks,256,0,stream>>>(x.data_ptr<uint8_t>(), columns.data_ptr<int8_t>(), g, xz, 128);
            else make_columns<int8_t><<<blocks,256,0,stream>>>(x.data_ptr<int8_t>(), columns.data_ptr<int8_t>(), g, xz, 0);
        }
        if (output_float) {
            if (unsigned_input) launch<uint8_t,float>(x, columns, packed, offsets, scales, zero_points, bias, out, g, output_scale, output_zero_point, relu, true);
            else launch<int8_t,float>(x, columns, packed, offsets, scales, zero_points, bias, out, g, output_scale, output_zero_point, relu, true);
        } else if (unsigned_input) launch<uint8_t,uint8_t>(x, columns, packed, offsets, scales, zero_points, bias, out, g, output_scale, output_zero_point, relu, false);
        else launch<int8_t,int8_t>(x, columns, packed, offsets, scales, zero_points, bias, out, g, output_scale, output_zero_point, relu, false);
        checkCuda(cudaGetLastError(), "Packed integer compute");
    }
    if (output_float) return out;
    return quantized::make_qtensor(out, make_per_tensor_affine_quantizer(output_scale, output_zero_point, input.dtype()), input.dtype());
}

template<class X>
__global__ void bn_relu_codes(const X* x, X* out, const float* weight, const float* bias,
                               const float* mean, const float* var, int64_t total, int64_t C,
                               int64_t spatial, float eps, double xs, int64_t xz, double ys, int64_t yz) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t c = (i / spatial) % C;
    const float inv = 1.f / sqrtf(var[c] + eps);
    const float gamma = weight ? weight[c] : 1.f;
    const float alpha = inv * gamma * (xs / ys);
    const float beta = ((bias ? bias[c] : 0.f) - mean[c] * inv * gamma) / ys;
    const float value = fmaf(alpha, static_cast<float>(static_cast<int64_t>(x[i]) - xz), beta);
    out[i] = static_cast<X>(fmin(std::is_same_v<X, uint8_t> ? 255. : 127.,
                                fmax(static_cast<double>(yz), nearbyint(static_cast<double>(value)) + yz)));
}

Tensor quantized_batch_norm_relu_cuda(
    const Tensor& input, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias,
    const Tensor& mean, const Tensor& var, double eps, double output_scale, int64_t output_zero_point) {
    CUDAGuard guard(input.device());
    quant::check_bn_relu(input, weight, bias, mean, var, eps, output_scale, output_zero_point);
    const Tensor x = input.contiguous(), m = mean.contiguous(), v = var.contiguous();
    const Tensor w = weight ? weight->contiguous() : Tensor(), b = bias ? bias->contiguous() : Tensor();
    const bool unsigned_input = input.dtype() == DType::QUInt8;
    Tensor out = Tensor::empty(input.shape(), unsigned_input ? DType::UInt8 : DType::Int8, input.device());
    int64_t spatial = 1;
    for (int64_t d = 2; d < input.dim(); ++d) spatial *= input.size(d);
    if (input.numel()) {
        const auto stream = getCurrentCUDAStream().stream();
        const int blocks = static_cast<int>((input.numel() + 255) / 256);
        const double xs = quantized::q_scale(input);
        const int64_t xz = quantized::q_zero_point(input);
        if (unsigned_input) bn_relu_codes<uint8_t><<<blocks,256,0,stream>>>(x.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), weight ? w.data_ptr<float>() : nullptr, bias ? b.data_ptr<float>() : nullptr, m.data_ptr<float>(), v.data_ptr<float>(), input.numel(), input.size(1), spatial, eps, xs, xz, output_scale, output_zero_point);
        else bn_relu_codes<int8_t><<<blocks,256,0,stream>>>(x.data_ptr<int8_t>(), out.data_ptr<int8_t>(), weight ? w.data_ptr<float>() : nullptr, bias ? b.data_ptr<float>() : nullptr, m.data_ptr<float>(), v.data_ptr<float>(), input.numel(), input.size(1), spatial, eps, xs, xz, output_scale, output_zero_point);
        checkCuda(cudaGetLastError(), "Fused normalization rectifier");
    }
    return quantized::make_qtensor(out, make_per_tensor_affine_quantizer(output_scale, output_zero_point, input.dtype()), input.dtype());
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantPacked) {
    m.impl("quantized_relu", quantized_relu_cuda);
    m.impl("quantized_relu6", quantized_relu6_cuda);
    m.impl("quantized_leaky_relu", quantized_leaky_relu_cuda);
    m.impl("quantized_elu", quantized_elu_cuda);
    m.impl("quantized_hardswish", quantized_hardswish_cuda);
    m.impl("quantized_hardsigmoid", quantized_hardsigmoid_cuda);
    m.impl("quantized_sigmoid", quantized_sigmoid_cuda);
    m.impl("quantized_tanh", quantized_tanh_cuda);
    m.impl("quantized_batch_norm_relu", quantized_batch_norm_relu_cuda);
    m.impl("_quantized_weight_prepack", _quantized_weight_prepack_cuda);
    m.impl("_quantized_packed_run", _quantized_packed_run_cuda);
}
} // namespace tensorplay::cuda
