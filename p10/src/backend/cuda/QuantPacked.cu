#include "QuantPacked.h"
#include "QuantNative.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "QuantElementwise.h"
#include "DispatchStub.h"
#include "QuantizedActivationMath.cuh"
#include <cuda_runtime.h>
#include <array>

namespace tensorplay::cuda {
namespace {

__device__ float sigmoid_exponential(float d) {
    if (d != d) return d;
    if (d < -104.f) return 0.f;
    if (d > 100.f) return CUDART_INF_F;
    const int q = __float2int_rn(__fmul_rn(d, 1.4426950408889634f));
    const float fq = static_cast<float>(q);
    float s = __fmaf_rn(fq, -.693145751953125f, d);
    s = __fmaf_rn(fq, -1.428606765330187045e-6f, s);
    float u = .000198527617612853646f;
    u = __fmaf_rn(u, s, .00139304355252534151f);
    u = __fmaf_rn(u, s, .00833336077630519867f);
    u = __fmaf_rn(u, s, .0416664853692054749f);
    u = __fmaf_rn(u, s, .166666671633720398f);
    u = __fmaf_rn(u, s, .5f);
    u = __fadd_rn(1.f, __fmaf_rn(__fmul_rn(s, s), u, s));
    const int half = q >> 1;
    const float a = __int_as_float((half + 127) << 23);
    const float b = __int_as_float((q - half + 127) << 23);
    return __fmul_rn(__fmul_rn(u, a), b);
}

struct ActivationPair { float high, low; };

__device__ ActivationPair activation_pair_add(ActivationPair x, float y) {
    const float sum = __fadd_rn(x.high, y);
    const float v = __fsub_rn(sum, x.high);
    const float error = __fadd_rn(__fsub_rn(x.high, __fsub_rn(sum, v)), __fsub_rn(y, v));
    return {sum, __fadd_rn(error, x.low)};
}

__device__ ActivationPair activation_pair_add(ActivationPair x, ActivationPair y) {
    const float sum = __fadd_rn(x.high, y.high);
    const float v = __fsub_rn(sum, x.high);
    const float error = __fadd_rn(__fsub_rn(x.high, __fsub_rn(sum, v)), __fsub_rn(y.high, v));
    return {sum, __fadd_rn(error, __fadd_rn(x.low, y.low))};
}

__device__ ActivationPair activation_pair_ordered_add(ActivationPair x, ActivationPair y) {
    const float sum = __fadd_rn(x.high, y.high);
    return {sum, __fadd_rn(__fadd_rn(__fadd_rn(__fsub_rn(x.high, sum), y.high), x.low), y.low)};
}

__device__ ActivationPair activation_pair_multiply(ActivationPair x, float y) {
    const float product = __fmul_rn(x.high, y);
    return {product, __fmaf_rn(x.low, y, __fmaf_rn(x.high, y, -product))};
}

__device__ ActivationPair activation_pair_multiply(ActivationPair x, ActivationPair y) {
    const float product = __fmul_rn(x.high, y.high);
    return {product, __fmaf_rn(x.high, y.low, __fmaf_rn(x.low, y.high, __fmaf_rn(x.high, y.high, -product)))};
}

__device__ ActivationPair activation_pair_square(ActivationPair x) {
    const float product = __fmul_rn(x.high, x.high);
    return {product, __fmaf_rn(__fadd_rn(x.high, x.high), x.low, __fmaf_rn(x.high, x.high, -product))};
}

__device__ ActivationPair activation_pair_reciprocal(ActivationPair x) {
    const float reciprocal = __fdiv_rn(1.f, x.high);
    return {reciprocal, __fmul_rn(reciprocal,
        __fmaf_rn(-x.low, reciprocal, __fmaf_rn(-x.high, reciprocal, 1.f)))};
}

__device__ ActivationPair activation_pair_divide(ActivationPair numerator, ActivationPair denominator) {
    const float reciprocal = __fdiv_rn(1.f, denominator.high);
    const float quotient = __fmul_rn(numerator.high, reciprocal);
    const float error = __fmaf_rn(reciprocal, numerator.high, -quotient);
    const float residual = __fmaf_rn(-denominator.low, reciprocal,
        __fmaf_rn(-denominator.high, reciprocal, 1.f));
    return {quotient, __fmaf_rn(quotient, residual, __fmaf_rn(numerator.low, reciprocal, error))};
}

__device__ float tanh_vector_value(float x) {
    if (x != x) return x;
    const float magnitude = ::fabsf(x);
    if (magnitude > 8.664339742f) return ::copysignf(1.f, x);
    const int q = __float2int_rn(__fmul_rn(magnitude, 1.4426950408889634f));
    ActivationPair s = activation_pair_add({magnitude, 0.f}, __fmul_rn(static_cast<float>(q), -.693145751953125f));
    s = activation_pair_add(s, __fmul_rn(static_cast<float>(q), -1.428606765330187045e-6f));
    float u = .1980960224e-3f;
    u = __fmaf_rn(u, s.high, .1394256484e-2f);
    u = __fmaf_rn(u, s.high, .8333456703e-2f);
    u = __fmaf_rn(u, s.high, .4166637361e-1f);
    auto t = activation_pair_add(activation_pair_multiply(s, u), .166666659414234245f);
    t = activation_pair_add(activation_pair_multiply(s, t), .5f);
    t = activation_pair_add(s, activation_pair_multiply(activation_pair_square(s), t));
    const float sum = __fadd_rn(1.f, t.high);
    t = {sum, __fadd_rn(__fadd_rn(__fsub_rn(1.f, sum), t.high), t.low)};
    const int half = q >> 1;
    const float a = __int_as_float((half + 127) << 23);
    const float b = __int_as_float((q - half + 127) << 23);
    t = {__fmul_rn(__fmul_rn(t.high, a), b), __fmul_rn(__fmul_rn(t.low, a), b)};
    const auto reciprocal = activation_pair_reciprocal(t);
    const auto result = activation_pair_divide(
        activation_pair_ordered_add(t, {-reciprocal.high, -reciprocal.low}),
        activation_pair_ordered_add(t, reciprocal));
    return ::copysignf(__fadd_rn(result.high, result.low), x);
}

__global__ void word_activation_codes(const int32_t* input, int32_t* output, int64_t size,
        float scale, int64_t zp, float inverse_scale, int64_t output_zp, int64_t vector_end,
        bool hyperbolic, bool vector_math) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        float value;
        if (i < vector_end) {
            if (!vector_math) {
                const float real = static_cast<float>((static_cast<double>(input[i]) - zp) * scale);
                if (hyperbolic) value = activation_math::tanh_scalar(real);
                else value = __fdiv_rn(1.f, __fadd_rn(1.f,
                    static_cast<float>(::exp(-static_cast<double>(real)))));
            } else if (hyperbolic) {
                const float real = __fmaf_rn(scale, __int2float_rn(input[i]), __fmul_rn(scale, -static_cast<float>(zp)));
                value = tanh_vector_value(real);
            } else {
                const float real = __fmul_rn(__fsub_rn(__int2float_rn(input[i]), static_cast<float>(zp)), scale);
                value = __fdiv_rn(1.f, __fadd_rn(1.f, sigmoid_exponential(-real)));
            }
        } else {
            const float real = static_cast<float>((static_cast<double>(input[i]) - zp) * scale);
            if (hyperbolic) value = activation_math::tanh_scalar(real);
            else {
                const float exponential = static_cast<float>(::exp(-static_cast<double>(real)));
                value = static_cast<float>(1. / (1. + static_cast<double>(exponential)));
            }
        }
        output[i] = quant::encode<int32_t>(value, inverse_scale, output_zp);
    }
}

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
    const auto format = quant::activation_format(input);
    const Tensor x = input.contiguous(static_cast<int64_t>(format));
    Tensor out = quant::activation_output(input, format);
    if (input.numel()) {
        const int blocks = static_cast<int>(std::min<int64_t>((input.numel() + 4095) / 4096, 4096));
        const auto stream = getCurrentCUDAStream().stream();
        const float inverse_scale = 1.f / static_cast<float>(scale);
        if (fn.op == 7 && scale == quantized::q_scale(input) && zp == xz) {
            auto clip = [&]<class T>() {
                const int64_t lo = quant::encode<T>(fn.slope, inverse_scale, zp);
                const T hi = quant::encode<T>(fn.alpha, inverse_scale, zp);
                relu6_codes<<<blocks,256,0,stream>>>(x.data_ptr<T>(), out.data_ptr<T>(), input.numel(), lo, hi);
            };
            switch (input.dtype()) {
                case DType::QInt8: clip.template operator()<int8_t>(); break;
                case DType::QUInt8: clip.template operator()<uint8_t>(); break;
                case DType::QInt32: clip.template operator()<int32_t>(); break;
                default: TP_THROW(TypeError, "Clipping requires quantized integer input");
            }
        } else if ((fn.op == 4 || fn.op == 5) && input.dtype() == DType::QInt32) {
            int lanes = 8;
            bool vector_math = false;
#if defined(__x86_64__) || defined(_M_X64)
            const auto capability = cpu::get_cpu_capability();
            vector_math = capability != cpu::CPUCapability::DEFAULT;
            if (capability == cpu::CPUCapability::AVX512) lanes = 16;
#endif
            const int64_t vector_end = input.impl()->is_contiguous_in(format)
                ? input.numel() / (2 * lanes) * (2 * lanes) : 0;
            word_activation_codes<<<blocks,256,0,stream>>>(x.data_ptr<int32_t>(), out.data_ptr<int32_t>(),
                input.numel(), xs, xz, inverse_scale, zp, vector_end, fn.op == 5, vector_math);
        } else switch (input.dtype()) {
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
            if (g.linear && output_float && shift==128) {
#pragma unroll
                for (int pair=0;pair<4;pair+=2) {
                    const int v0=static_cast<int>(xtile[threadIdx.y][k+pair])+shift;
                    const int v1=static_cast<int>(xtile[threadIdx.y][k+pair+1])+shift;
                    const int pair_sum=v0*wtile[threadIdx.x][k+pair]+v1*wtile[threadIdx.x][k+pair+1];
                    part+=min(32767,max(-32768,pair_sum));
                }
            } else part = dot4(a, b, part);
            sum = dot4(a, 0x01010101, sum);
        }
        acc += part; row_sum += sum;
        __syncthreads();
    }
    if (row >= g.rows || local_n >= n_per_group) return;
    if (g.linear && output_float && shift==128) acc-=shift*offsets[n];
    acc = acc - xz * offsets[n] - zps[n] * row_sum + g.K * xz * zps[n];
    double value = quant::packed_real_value(acc,xs,scales[n],bias ? bias[n] : 0.f,output_float);
    const int64_t index = g.linear ? row * g.outputs + n : ((row / g.spatial) * g.outputs + n) * g.spatial + row % g.spatial;
    value = quant::packed_activation(value, index, g);
    if (relu) value = fmax(value, 0.);
    if (output_float) out[index] = static_cast<Y>(value);
    else {
        const double lo = std::is_same_v<Y, uint8_t> ? 0. : -128.;
        out[index] = static_cast<Y>(fmin(lo + 255., fmax(lo, nearbyint(value / ys) + yz)));
    }
}

template<class X, class Y>
__global__ void packed_gemv(const X* input,const int8_t* weight,const int64_t* offsets,
        const float* scales,const int64_t* zps,const float* bias,Y* out,quant::PackedDeviceGeometry g,
        double xs,int64_t xz,double ys,int64_t yz,int shift,bool relu,bool output_float) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int64_t n=static_cast<int64_t>(blockIdx.x)*8+warp,row=blockIdx.y;
    if (n>=g.outputs) return;
    int64_t acc=0,sum=0;
    for (int64_t k=lane*4;k<g.K;k+=128) {
        int a=0,b=0;
        if (g.K%4==0) {
            a=*reinterpret_cast<const int*>(input+row*g.K+k);
            if (shift) a^=static_cast<int>(0x80808080u);
            b=*reinterpret_cast<const int*>(weight+n*g.padded_K+k);
        } else {
            for (int j=0;j<4 && k+j<g.K;++j) {
                a|=static_cast<int>(static_cast<uint8_t>(static_cast<int>(input[row*g.K+k+j])-shift))<<(j*8);
                b|=static_cast<int>(static_cast<uint8_t>(weight[n*g.padded_K+k+j]))<<(j*8);
            }
        }
        if (output_float && shift==128) {
            for (int pair=0;pair<4;pair+=2) {
                const int i0=static_cast<int8_t>(a>>(pair*8))+shift;
                const int i1=static_cast<int8_t>(a>>((pair+1)*8))+shift;
                const int product=i0*static_cast<int8_t>(b>>(pair*8))+i1*static_cast<int8_t>(b>>((pair+1)*8));
                acc+=min(32767,max(-32768,product));
            }
        } else acc+=dot4(a,b,0);
        sum+=dot4(a,0x01010101,0);
    }
    for (int step=16;step;step/=2) {
        acc+=__shfl_down_sync(0xffffffffffffffffull,acc,step);
        sum+=__shfl_down_sync(0xffffffffffffffffull,sum,step);
    }
    if (lane) return;
    if (output_float && shift==128) acc-=shift*offsets[n];
    acc=acc-xz*offsets[n]-zps[n]*sum+g.K*xz*zps[n];
    const int64_t index=row*g.outputs+n;
    double value=quant::packed_real_value(acc,xs,scales[n],bias ? bias[n] : 0.f,output_float);
    value=quant::packed_activation(value,index,g);
    if (relu) value=fmax(value,0.);
    if (output_float) out[index]=static_cast<Y>(value);
    else {
        const double lo=std::is_same_v<Y,uint8_t> ? 0. : -128.;
        out[index]=static_cast<Y>(fmin(lo+255.,fmax(lo,nearbyint(value/ys)+yz)));
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
    if (g.linear && g.rows<=4) {
        packed_gemv<X,Y><<<dim3(static_cast<unsigned>((g.outputs+7)/8),static_cast<unsigned>(g.rows)),256,0,stream>>>(
            x.data_ptr<X>(),packed.data_ptr<int8_t>(),offsets.data_ptr<int64_t>(),scales.data_ptr<float>(),
            zps.data_ptr<int64_t>(),bias ? bias->data_ptr<float>() : nullptr,out.data_ptr<Y>(),g,xs,xz,ys,yz,shift,relu,output_float);
        return;
    }
    if (g.linear) packed_gemm<X,Y><<<grid, block, 0, stream>>>(x.data_ptr<X>(), packed.data_ptr<int8_t>(), offsets.data_ptr<int64_t>(), scales.data_ptr<float>(), zps.data_ptr<int64_t>(), bias ? bias->data_ptr<float>() : nullptr, out.data_ptr<Y>(), g, xs, xz, ys, yz, shift, relu, output_float);
    else packed_gemm<int8_t,Y><<<grid, block, 0, stream>>>(columns.data_ptr<int8_t>(), packed.data_ptr<int8_t>(), offsets.data_ptr<int64_t>(), scales.data_ptr<float>(), zps.data_ptr<int64_t>(), bias ? bias->data_ptr<float>() : nullptr, out.data_ptr<Y>(), g, xs, xz, ys, yz, 0, relu, output_float);
}

template<class X>
__global__ void prepare_integer_rows(const X* x,int8_t* signed_x,int64_t* sums,int64_t rows,int64_t k) {
    const int64_t row=blockIdx.x;
    int64_t sum=0;
    for (int64_t col=threadIdx.x;col<k;col+=blockDim.x) {
        const int value=static_cast<int>(x[row*k+col])-(std::is_same_v<X,uint8_t> ? 128 : 0);
        if (signed_x) signed_x[row*k+col]=static_cast<int8_t>(value);
        sum+=value;
    }
    __shared__ int64_t partial[256];
    partial[threadIdx.x]=sum;
    __syncthreads();
    for (int step=128;step;step/=2) {
        if (threadIdx.x<step) partial[threadIdx.x]+=partial[threadIdx.x+step];
        __syncthreads();
    }
    if (!threadIdx.x && sums) sums[row]=partial[0];
}

template<class Y,bool Symmetric>
__global__ void integer_product_epilogue(const int32_t* raw,const int64_t* sums,
        const int64_t* offsets,const float* scales,const int64_t* zps,const float* bias,Y* out,
        quant::PackedDeviceGeometry g,double xs,int64_t xz,double ys,int64_t yz,bool relu) {
    for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        i<g.rows*g.outputs;i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        const int64_t n=i%g.outputs,row=i/g.outputs;
        int64_t acc=raw[i];
        if (xz) acc-=xz*offsets[n];
        if constexpr (!Symmetric) acc-=zps[n]*sums[row]-g.K*xz*zps[n];
        double value=quant::packed_real_value(acc,xs,scales[n],bias ? bias[n] : 0.f,false);
        value=quant::packed_activation(value,i,g);
        if (relu) value=fmax(value,0.);
        const double lo=std::is_same_v<Y,uint8_t> ? 0. : -128.;
        out[i]=static_cast<Y>(fmin(lo+255.,fmax(lo,nearbyint(value/ys)+yz)));
    }
}

bool symmetric_zero_points(const Tensor& zps) {
    struct Entry { Tensor source; uint32_t version=0; uintptr_t stream=0; bool symmetric=false; };
    thread_local std::array<Entry,32> cache;
    thread_local size_t next=0;
    const bool cacheable=zps.impl()->version_counter().is_enabled();
    const uint32_t version=cacheable ? zps._version() : 0;
    const uintptr_t stream=reinterpret_cast<uintptr_t>(getCurrentCUDAStream().stream());
    for (const auto& entry:cache)
        if (cacheable && entry.source.defined() && entry.source.impl()==zps.impl()
            && entry.version==version && entry.stream==stream) return entry.symmetric;
    const Tensor host=zps.to(Device(DeviceType::CPU)).contiguous();
    const auto* values=host.data_ptr<int64_t>();
    bool symmetric=true;
    for (int64_t i=0;i<host.numel();++i) if (values[i]) { symmetric=false; break; }
    if (cacheable && zps.numel()<=65536) cache[next++%cache.size()]={zps,version,stream,symmetric};
    return symmetric;
}

bool integer_product(const Tensor& x,const Tensor& packed,const Tensor& offsets,const Tensor& scales,
        const Tensor& zps,const std::optional<Tensor>& bias,Tensor& out,const quant::PackedGeometry& g,
        double ys,int64_t yz,bool relu,bool output_float) {
    if (!g.linear || output_float || g.K==0 || g.K>65536 || g.K%4 || g.outputs%4 || g.rows<8
        || g.rows*g.outputs*g.K<1048576 || g.rows>2147483647 || g.outputs>2147483647) return false;
    const auto stream=getCurrentCUDAStream().stream();
    const bool symmetric=symmetric_zero_points(zps);
    Tensor sums=Tensor::empty({symmetric ? 0 : g.rows},DType::Int64,x.device());
    Tensor signed_x;
    const bool unsigned_input=x.dtype()==DType::QUInt8;
    if (unsigned_input) {
        signed_x=Tensor::empty(x.shape(),DType::Int8,x.device());
        prepare_integer_rows<<<g.rows,256,0,stream>>>(x.data_ptr<uint8_t>(),signed_x.data_ptr<int8_t>(),symmetric ? nullptr : sums.data_ptr<int64_t>(),g.rows,g.K);
    } else if (!symmetric) prepare_integer_rows<<<g.rows,256,0,stream>>>(x.data_ptr<int8_t>(),nullptr,sums.data_ptr<int64_t>(),g.rows,g.K);
    Tensor raw=Tensor::empty({g.rows,g.outputs},DType::Int32,x.device());
    const auto handle=CUDAContext::getCublasHandle();
    auto checked=[](cublasStatus_t status) {
        TP_CHECK(status==CUBLAS_STATUS_SUCCESS,"Integer matrix BLAS call failed with status ",static_cast<int>(status));
    };
    struct RestoreMode {
        cublasHandle_t handle; cublasMath_t math=CUBLAS_DEFAULT_MATH; cublasPointerMode_t pointer=CUBLAS_POINTER_MODE_HOST;
        ~RestoreMode() { cublasSetMathMode(handle,math); cublasSetPointerMode(handle,pointer); }
    } restore{handle};
    checked(cublasGetMathMode(handle,&restore.math)); checked(cublasGetPointerMode(handle,&restore.pointer));
    checked(cublasSetStream(handle,stream)); checked(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
    checked(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
    const int32_t alpha=1,beta=0;
    const auto status=cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,g.outputs,g.rows,g.K,&alpha,
        packed.data_ptr<int8_t>(),CUDA_R_8I,g.padded_K,unsigned_input ? signed_x.data_ptr<int8_t>() : x.data_ptr<int8_t>(),
        CUDA_R_8I,g.K,&beta,raw.data_ptr<int32_t>(),CUDA_R_32I,g.outputs,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status==CUBLAS_STATUS_NOT_SUPPORTED) return false;
    checked(status);
    const int blocks=static_cast<int>(std::min<int64_t>((out.numel()+255)/256,65535));
    const double xs=quantized::q_scale(x);
    const int64_t xz=quantized::q_zero_point(x)-(unsigned_input ? 128 : 0);
    auto launch_epilogue=[&]<class Y,bool Symmetric>() {
        integer_product_epilogue<Y,Symmetric><<<blocks,256,0,stream>>>(raw.data_ptr<int32_t>(),
            Symmetric ? nullptr : sums.data_ptr<int64_t>(),offsets.data_ptr<int64_t>(),scales.data_ptr<float>(),
            zps.data_ptr<int64_t>(),bias ? bias->data_ptr<float>() : nullptr,out.data_ptr<Y>(),g,xs,xz,ys,yz,relu);
    };
    if (unsigned_input) {
        if (symmetric) launch_epilogue.template operator()<uint8_t,true>();
        else launch_epilogue.template operator()<uint8_t,false>();
    } else {
        if (symmetric) launch_epilogue.template operator()<int8_t,true>();
        else launch_epilogue.template operator()<int8_t,false>();
    }
    checkCuda(cudaGetLastError(),"Integer matrix epilogue");
    return true;
}
} // namespace

Tensor quantized_relu_cuda(const Tensor& input) {
    CUDAGuard guard(input.device());
    const int64_t zp = quantized::q_zero_point(input);
    const auto format = quant::activation_format(input);
    const Tensor x = input.contiguous(static_cast<int64_t>(format));
    Tensor out = quant::activation_output(input, format);
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
    const auto format = quant::activation_format(input);
    const Tensor x = input.contiguous(static_cast<int64_t>(format));
    Tensor out = quant::activation_output(input, format);
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
    const std::vector<int64_t>& dilation, int64_t groups, bool relu, bool output_float,
    int64_t activation, double negative_slope, const std::optional<Tensor>& residual) {
    CUDAGuard guard(input.device());
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
    const bool unsigned_input = input.dtype() == DType::QUInt8;
    Tensor out = Tensor::empty(g.shape, output_float ? DType::Float32 : (unsigned_input ? DType::UInt8 : DType::Int8), input.device());
    Tensor columns = Tensor::empty({0}, DType::Int8, input.device());
    if (out.numel() && !integer_product(x,packed,offsets,scales,zero_points,bias,out,g,
        output_scale,output_zero_point,relu,output_float)) {
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
    Tensor result = quantized::make_qtensor(out, make_per_tensor_affine_quantizer(output_scale, output_zero_point, input.dtype()), input.dtype());
    if (residual && result.numel()) {
        Tensor destination = residual->contiguous(static_cast<int64_t>(MemoryFormat::ChannelsLast));
        destination.impl()->set_quantizer(result.impl()->quantizer());
        destination.copy_(result);
        return destination;
    }
    return result;
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
    return quant::batch_norm_cuda(input, weight, bias, mean, var,
        eps, output_scale, output_zero_point, true);
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

namespace tensorplay::quant {
Tensor activation_cuda(const Tensor& input, double scale, int64_t zero_point, Activation fn) {
    return tensorplay::cuda::activation(input, scale, zero_point, fn);
}
}
