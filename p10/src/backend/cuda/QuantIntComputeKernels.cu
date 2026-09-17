// Quantized integer compute ops for the CUDA backend: the quantized linear
// layer, the four elementwise binary ops over QInt8 tensors with per-tensor
// scales, and quantized_clamp.  Split from QuantKernels.cu so edits to the
// int8 compute path do not recompile the observer / fake-quant families.

#include "QuantKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"
#include "Utils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

__global__ void quantized_linear_kernel(int64_t total, int64_t k_size,
                                        int64_t out_features,
                                        const int8_t* x, const int8_t* w,
                                        double input_scale,
                                        int64_t input_zero_point,
                                        const float* w_scales,
                                        const int64_t* w_zps,
                                        const float* bias, int8_t* out,
                                        double out_scale,
                                        int64_t out_zero_point) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int64_t m = e / out_features;
    const int64_t n = e - m * out_features;
    const int8_t* x_row = x + m * k_size;
    const int8_t* w_row = w + n * k_size;
    const int64_t w_zp = w_zps[n];
    int64_t acc = 0;
    for (int64_t k = 0; k < k_size; ++k) {
        acc += static_cast<int64_t>(x_row[k] - input_zero_point) *
               static_cast<int64_t>(w_row[k] - w_zp);
    }
    // exact integer accumulation, requantized only at the output
    const double y =
        static_cast<double>(acc) *
            (input_scale * static_cast<double>(w_scales[n])) +
        static_cast<double>(bias[n]);
    const double rounded =
        ::nearbyint(y / out_scale) + static_cast<double>(out_zero_point);
    out[e] = static_cast<int8_t>(::min(static_cast<long long>(127),
                                       ::max(static_cast<long long>(-128),
                                             static_cast<long long>(rounded))));
}

// ---------------------------------------------------------------------------
// Dynamic quantized linear: activations are quantized per row from their
// observed range, the GEMM runs in the integer domain, and the result is
// returned in the float domain.
// ---------------------------------------------------------------------------

__global__ void dynamic_row_quantize_kernel(int64_t m_size, int64_t k_size,
                                            const float* x, double* x_scales,
                                            int64_t* x_zps, int8_t* x_q) {
    const int64_t m = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= m_size) return;
    const float* row = x + m * k_size;
    double rmin = static_cast<double>(row[0]);
    double rmax = rmin;
    for (int64_t k = 1; k < k_size; ++k) {
        const double v = static_cast<double>(row[k]);
        rmin = ::min(rmin, v);
        rmax = ::max(rmax, v);
    }
    // zero stays exactly representable on the grid
    rmin = ::min(rmin, 0.0);
    rmax = ::max(rmax, 0.0);
    double scale = (rmax - rmin) / 255.0;
    if (scale <= 0.0) scale = 0.1;
    const int64_t zero_point =
        static_cast<int64_t>(::nearbyint(-rmin / scale)) - 128;
    x_scales[m] = scale;
    x_zps[m] = zero_point;
    int8_t* qrow = x_q + m * k_size;
    for (int64_t k = 0; k < k_size; ++k) {
        const double rounded = ::nearbyint(static_cast<double>(row[k]) / scale) +
                               static_cast<double>(zero_point);
        qrow[k] = static_cast<int8_t>(
            ::min(static_cast<long long>(127),
                  ::max(static_cast<long long>(-128),
                        static_cast<long long>(rounded))));
    }
}

__global__ void quantized_linear_dynamic_kernel(
    int64_t total, int64_t k_size, int64_t out_features, const int8_t* x_q,
    const int64_t* x_zps, const double* x_scales, const int8_t* w,
    const float* w_scales, const int64_t* w_zps, const float* bias,
    float* out) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int64_t m = e / out_features;
    const int64_t n = e - m * out_features;
    const int8_t* x_row = x_q + m * k_size;
    const int8_t* w_row = w + n * k_size;
    const int64_t x_zp = x_zps[m];
    const int64_t w_zp = w_zps[n];
    int64_t acc = 0;
    for (int64_t k = 0; k < k_size; ++k) {
        acc += static_cast<int64_t>(x_row[k] - x_zp) *
               static_cast<int64_t>(w_row[k] - w_zp);
    }
    out[e] = static_cast<float>(static_cast<double>(acc) * x_scales[m] *
                                    static_cast<double>(w_scales[n]) +
                                static_cast<double>(bias[n]));
}


Tensor quantized_linear_dynamic_cuda(const Tensor& input, const Tensor& weight,
                                     const Tensor& weight_scales,
                                     const Tensor& weight_zero_points,
                                     std::optional<Tensor> bias,
                                     bool reduce_range) {
    // Dynamic quantized linear: one pass quantizes each activation row from
    // its observed range, then the integer GEMM returns float results.
    if (weight.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "quantized_linear_dynamic(): weight must be QInt8");
    }
    if (input.dim() != 2 || weight.dim() != 2) {
        TP_THROW(ValueError,
                 "quantized_linear_dynamic(): expected 2-D [M,K] activations "
                 "and [N,K] weights");
    }
    if (input.size(1) != weight.size(1)) {
        TP_THROW(ValueError,
                 "quantized_linear_dynamic(): incompatible K dimensions (" +
                     std::to_string(input.size(1)) + " vs " +
                     std::to_string(weight.size(1)) + ")");
    }
    const int64_t out_features = weight.size(0);
    if (weight_scales.dim() != 1 || weight_scales.size(0) != out_features ||
        weight_zero_points.shape() != weight_scales.shape()) {
        TP_THROW(ValueError,
                 "quantized_linear_dynamic(): weight scales/zero_points must "
                 "be 1-D of length out_features");
    }
    if (input.dtype() != DType::Float32) {
        TP_THROW(TypeError,
                 "quantized_linear_dynamic(): activations must be Float32");
    }
    (void)reduce_range;

    Tensor x = input.contiguous();
    Tensor w = weight.contiguous();
    Tensor sc = weight_scales.to(DType::Float32).contiguous();
    Tensor zp = weight_zero_points.to(DType::Int64).contiguous();
    Tensor zps_host = zp.to(Device(DeviceType::CPU));
    const int64_t* host_zps = zps_host.data_ptr<int64_t>();
    for (int64_t n = 0; n < out_features; ++n) {
        if (host_zps[n] < -128 || host_zps[n] > 127) {
            TP_THROW(ValueError,
                     "quantized_linear_dynamic(): weight zero_point outside "
                     "the Int8 range");
        }
    }

    Tensor bias_f;
    if (bias.has_value()) {
        if (!isFloatingType(bias->dtype()) || bias->dim() != 1 ||
            bias->size(0) != out_features) {
            TP_THROW(ValueError,
                     "quantized_linear_dynamic(): bias must be a 1-D floating "
                     "tensor of length out_features");
        }
        bias_f = bias->to(DType::Float32).contiguous();
    } else {
        bias_f = Tensor::zeros({out_features}, DType::Float32, x.device());
    }

    const int64_t m_size = x.size(0);
    const int64_t k_size = x.size(1);
    Tensor xs_t = Tensor::empty({m_size}, DType::Float64, x.device());
    Tensor xzp_t = Tensor::empty({m_size}, DType::Int64, x.device());
    Tensor xq_t = Tensor::empty({m_size, k_size}, DType::Int8, x.device());
    Tensor out = Tensor::empty({m_size, out_features}, DType::Float32,
                               x.device());
    const int64_t total = m_size * out_features;
    if (total == 0) return out;

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    dynamic_row_quantize_kernel<<<
        static_cast<int>((m_size + threads - 1) / threads), threads, 0,
        stream>>>(m_size, k_size, x.data_ptr<float>(), xs_t.data_ptr<double>(),
                  xzp_t.data_ptr<int64_t>(), xq_t.data_ptr<int8_t>());
    checkCuda(cudaGetLastError(), "CUDA dynamic row quantize kernel");
    quantized_linear_dynamic_kernel<<<static_cast<int>((total + threads - 1) / threads),
                                    threads, 0, stream>>>(
        total, k_size, out_features, xq_t.data_ptr<int8_t>(),
        xzp_t.data_ptr<int64_t>(), xs_t.data_ptr<double>(),
        w.data_ptr<int8_t>(), sc.data_ptr<float>(), zp.data_ptr<int64_t>(),
        bias_f.data_ptr<float>(), out.data_ptr<float>());
    checkCuda(cudaGetLastError(), "CUDA quantized_linear_dynamic kernel");
    return out;
}

Tensor quantized_linear_cuda(const Tensor& input, const Tensor& weight,
                             double input_scale, int64_t input_zero_point,
                             const Tensor& weight_scales,
                             const Tensor& weight_zero_points,
                             std::optional<Tensor> bias,
                             double out_scale, int64_t out_zero_point) {
    // Fused Int8 GEMM with per-channel weight requantization; one thread per
    // (m, n) output element streams both operand rows over K.
    if (input.dtype() != DType::QInt8 || weight.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "quantized_linear(): activations and weights must be QInt8");
    }
    if (input.dim() != 2 || weight.dim() != 2) {
        TP_THROW(ValueError,
                 "quantized_linear(): expected 2-D [M,K] activations and "
                 "[N,K] weights");
    }
    if (!(input_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_linear(): scale must be positive");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_linear(): out_scale must be positive");
    }
    if (input.size(1) != weight.size(1)) {
        TP_THROW(ValueError,
                 "quantized_linear(): incompatible K dimensions (" +
                     std::to_string(input.size(1)) + " vs " +
                     std::to_string(weight.size(1)) + ")");
    }
    const int64_t out_features = weight.size(0);
    if (weight_scales.dim() != 1 || weight_scales.size(0) != out_features ||
        weight_zero_points.shape() != weight_scales.shape()) {
        TP_THROW(ValueError,
                 "quantized_linear(): weight scales/zero_points must be 1-D "
                 "of length out_features");
    }

    Tensor x = input.contiguous();
    Tensor w = weight.contiguous();
    Tensor sc = weight_scales.to(DType::Float32).contiguous();
    Tensor zp = weight_zero_points.to(DType::Int64).contiguous();
    Tensor zps_host = zp.to(Device(DeviceType::CPU));
    const int64_t* host_zps = zps_host.data_ptr<int64_t>();
    for (int64_t n = 0; n < out_features; ++n) {
        if (host_zps[n] < -128 || host_zps[n] > 127) {
            TP_THROW(ValueError,
                     "quantized_linear(): zero_point out of the Int8 range");
        }
    }

    Tensor bias_f;
    if (bias.has_value()) {
        if (!isFloatingType(bias->dtype()) || bias->dim() != 1 ||
            bias->size(0) != out_features) {
            TP_THROW(ValueError,
                     "quantized_linear(): bias must be a 1-D floating tensor "
                     "of length out_features");
        }
        bias_f = bias->to(DType::Float32).contiguous();
    } else {
        bias_f = Tensor::zeros({out_features}, DType::Float32, x.device());
    }

    const int64_t m_size = x.size(0);
    const int64_t k_size = x.size(1);
    Tensor out = Tensor::empty({m_size, out_features}, DType::Int8,
                               x.device());
    const int64_t total = m_size * out_features;
    if (total == 0) {
        return quantized::make_qtensor(
            out, make_per_tensor_affine_quantizer(out_scale, out_zero_point,
                                                  DType::QInt8),
            DType::QInt8);
    }

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    quantized_linear_kernel<<<blocks, threads, 0, stream>>>(
        total, k_size, out_features, x.data_ptr<int8_t>(),
        w.data_ptr<int8_t>(), input_scale, input_zero_point,
        sc.data_ptr<float>(), zp.data_ptr<int64_t>(),
        bias_f.data_ptr<float>(), out.data_ptr<int8_t>(), out_scale,
        out_zero_point);
    checkCuda(cudaGetLastError(), "CUDA quantized_linear kernel");
    return quantized::make_qtensor(
        out, make_per_tensor_affine_quantizer(out_scale, out_zero_point,
                                              DType::QInt8),
        DType::QInt8);
}

// ---------------------------------------------------------------------------
// Quantized elementwise arithmetic over Int8 storage with explicit qparams:
// dequantize both operands as (q - zero_point) * scale, apply the float
// operation, requantize with round-to-nearest-even into [-128, 127] under
// the output qparams.  Division by zero follows IEEE float rules.
// ---------------------------------------------------------------------------

namespace {

__global__ void quantized_binary_kernel(
    int64_t numel,
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int8_t* __restrict__ out,
    const int64_t* __restrict__ a_strides,
    const int64_t* __restrict__ b_strides,
    const int64_t* __restrict__ out_strides,
    int rank,
    float a_scale, float a_zp,
    float b_scale, float b_zp,
    float inv_out_scale, float out_zp,
    int op) {
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= numel) return;

    int64_t rem = flat;
    int64_t ia = 0, ib = 0;
    for (int d = 0; d < rank; ++d) {
        const int64_t coord = rem / out_strides[d];
        rem -= coord * out_strides[d];
        ia += coord * a_strides[d];
        ib += coord * b_strides[d];
    }

    const float xa = (static_cast<float>(a[ia]) - a_zp) * a_scale;
    const float xb = (static_cast<float>(b[ib]) - b_zp) * b_scale;
    float y;
    switch (op) {
        case 0: y = xa + xb; break;
        case 1: y = xa - xb; break;
        case 2: y = xa * xb; break;
        default: y = xa / xb; break;
    }
    const float q = rintf(y * inv_out_scale) + out_zp;
    out[flat] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, q)));
}

void check_quantized_binary(
    const Tensor& a, const Tensor& b, double out_scale) {
    if (a.dtype() != DType::QInt8 || b.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "quantized op(): operands must be QInt8");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized op(): out_scale must be positive");
    }
}

Tensor quantized_binary_cuda_impl(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point,
    int op) {
    check_quantized_binary(a, b, out_scale);
    const std::vector<int64_t> out_sizes =
        broadcast_shapes(static_cast<std::vector<int64_t>>(a.shape()),
                         static_cast<std::vector<int64_t>>(b.shape()));
    Tensor out = Tensor::empty(out_sizes, DType::QInt8, a.device());

    // Broadcast strides pin singleton dimensions with a zero step, so one
    // flat index decomposition addresses both operands without materializing
    // an expanded copy.
    const std::vector<int64_t> sa = broadcast_strides(a, out_sizes);
    const std::vector<int64_t> sb = broadcast_strides(b, out_sizes);
    const std::vector<int64_t> so =
        SizesAndStrides::compute_contiguous_strides(out_sizes);
    const int rank = static_cast<int>(out_sizes.size());

    Tensor strides_gpu = Tensor::empty(
        {static_cast<int64_t>(rank) * 3}, DType::Int64, a.device());
    int64_t* host = static_cast<int64_t*>(
        malloc(sizeof(int64_t) * rank * 3));
    for (int d = 0; d < rank; ++d) {
        host[d] = sa[d];
        host[rank + d] = sb[d];
        host[2 * rank + d] = so[d];
    }
    cudaMemcpy(strides_gpu.data_ptr<int64_t>(), host,
               sizeof(int64_t) * rank * 3, cudaMemcpyHostToDevice);
    free(host);

    const Tensor ac = a.is_contiguous() ? a : a.contiguous();
    const Tensor bc = b.is_contiguous() ? b : b.contiguous();

    const int64_t numel = out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantized_binary_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, ac.data_ptr<int8_t>(), bc.data_ptr<int8_t>(),
        out.data_ptr<int8_t>(),
        strides_gpu.data_ptr<int64_t>(),
        strides_gpu.data_ptr<int64_t>() + rank,
        strides_gpu.data_ptr<int64_t>() + 2 * rank,
        rank,
        static_cast<float>(a_scale), static_cast<float>(a_zero_point),
        static_cast<float>(b_scale), static_cast<float>(b_zero_point),
        static_cast<float>(1.0 / out_scale),
        static_cast<float>(out_zero_point), op);
    checkCuda(cudaGetLastError(), "CUDA quantized binary kernel");
    out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
        out_scale, out_zero_point, DType::QInt8));
    return out;
}

__global__ void quantized_clamp_kernel(
    int64_t numel,
    const int8_t* __restrict__ in,
    int8_t* __restrict__ out,
    float in_scale, float in_zp,
    float inv_out_scale, float out_zp,
    int has_min, int has_max,
    float min_value, float max_value) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    float y = (static_cast<float>(in[i]) - in_zp) * in_scale;
    if (has_min) y = fmaxf(y, min_value);
    if (has_max) y = fminf(y, max_value);
    const float q = rintf(y * inv_out_scale) + out_zp;
    out[i] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, q)));
}

} // namespace

Tensor quantized_add_cuda(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cuda_impl(a, b, a_scale, a_zero_point, b_scale,
                                      b_zero_point, out_scale, out_zero_point,
                                      0);
}

Tensor quantized_sub_cuda(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cuda_impl(a, b, a_scale, a_zero_point, b_scale,
                                      b_zero_point, out_scale, out_zero_point,
                                      1);
}

Tensor quantized_mul_cuda(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cuda_impl(a, b, a_scale, a_zero_point, b_scale,
                                      b_zero_point, out_scale, out_zero_point,
                                      2);
}

Tensor quantized_div_cuda(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cuda_impl(a, b, a_scale, a_zero_point, b_scale,
                                      b_zero_point, out_scale, out_zero_point,
                                      3);
}

Tensor quantized_clamp_cuda(
    const Tensor& self, double self_scale, int64_t self_zero_point,
    double out_scale, int64_t out_zero_point,
    std::optional<Scalar> min, std::optional<Scalar> max) {
    if (self.dtype() != DType::Int8) {
        TP_THROW(TypeError, "quantized_clamp(): expected an Int8 tensor");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_clamp(): out_scale must be positive");
    }
    const Tensor sc = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::QInt8, self.device());

    const int64_t numel = self.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    quantized_clamp_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        numel, sc.data_ptr<int8_t>(), out.data_ptr<int8_t>(),
        static_cast<float>(self_scale), static_cast<float>(self_zero_point),
        static_cast<float>(1.0 / out_scale),
        static_cast<float>(out_zero_point),
        min.has_value() ? 1 : 0, max.has_value() ? 1 : 0,
        min.has_value() ? static_cast<float>(min->toDouble()) : 0.0f,
        max.has_value() ? static_cast<float>(max->toDouble()) : 0.0f);
    checkCuda(cudaGetLastError(), "CUDA quantized_clamp kernel");
    out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
        out_scale, out_zero_point, DType::QInt8));
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantIntComputeKernels) {
    m.impl("quantized_linear", quantized_linear_cuda);
    m.impl("quantized_linear_dynamic", quantized_linear_dynamic_cuda);
    m.impl("quantized_add", quantized_add_cuda);
    m.impl("quantized_sub", quantized_sub_cuda);
    m.impl("quantized_mul", quantized_mul_cuda);
    m.impl("quantized_div", quantized_div_cuda);
    m.impl("quantized_clamp", quantized_clamp_cuda);
}

} // namespace cuda
} // namespace tensorplay
