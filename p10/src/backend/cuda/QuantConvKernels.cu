// Native quantized convolution kernels for the CUDA backend.
//
// Every output code is computed by accumulating in the integer domain
//   acc = sum_k (in_q[k] - in_zero_point) * (w_q[k] - w_zero_point)
// with a single requantization at the output
//   out_q = clamp(nearbyint(acc * input_scale * weight_scale + bias
//                           / out_scale) + out_zero_point).
// The 1d and 3d variants share the 2d integer kernel through shape lifting
// and a dedicated 3d kernel respectively.  prepack/unpack are provided for
// API completeness and operate on the float-domain weight payload.
//
// The fast 2d path is an implicit-GEMM kernel with shared-memory tiling:
// the weight matrix (M x K, K = Cg*kH*kW) and the implicit im2col input
// matrix (K x OH*OW) are streamed in BK-wide tiles, each thread block
// accumulates a BM x BN output tile.  Input re-use comes from the shared
// memory tiles, so every global load is amortized across the tile.
// Depthwise/large-K shapes fall back to the scalar kernel below.

#include "QuantCudaKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"
#include "QuantUtils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

__global__ void quantized_conv2d_kernel(
    const int8_t* __restrict__ in, const int8_t* __restrict__ w,
    const float* __restrict__ bias, int8_t* __restrict__ out,
    int64_t total, int64_t C, int64_t H, int64_t W, int64_t M, int64_t Cg,
    int64_t kH, int64_t kW, int64_t OH, int64_t OW, int64_t groups,
    int64_t sH, int64_t sW, int64_t pH, int64_t pW, int64_t dH, int64_t dW,
    double input_scale, double input_zero_point, double weight_scale,
    double weight_zero_point, double inv_out_scale, double out_zero_point) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t ow = idx % OW;
    const int64_t oh = (idx / OW) % OH;
    const int64_t m = (idx / (OW * OH)) % M;
    const int64_t n = idx / (OW * OH * M);
    const int64_t per_group_out = M / groups;
    const int64_t g = m / per_group_out;
    int64_t acc = 0;
    for (int64_t kh = 0; kh < kH; ++kh) {
        const int64_t ih = oh * sH - pH + kh * dH;
        if (ih < 0 || ih >= H) continue;
        for (int64_t kw = 0; kw < kW; ++kw) {
            const int64_t iw = ow * sW - pW + kw * dW;
            if (iw < 0 || iw >= W) continue;
            for (int64_t c = 0; c < Cg; ++c) {
                const int64_t in_c = g * Cg + c;
                const int8_t xv = in[((n * C + in_c) * H + ih) * W + iw];
                const int8_t wv = w[((m * Cg + c) * kH + kh) * kW + kw];
                acc += static_cast<int64_t>(xv - input_zero_point) *
                       static_cast<int64_t>(wv - weight_zero_point);
            }
        }
    }
    double y = static_cast<double>(acc) * (input_scale * weight_scale);
    if (bias != nullptr) y += static_cast<double>(bias[m]);
    out[idx] = quant::requantize_value(y, inv_out_scale,
                                       static_cast<int64_t>(out_zero_point));
}

__global__ void quantized_conv3d_kernel(
    const int8_t* __restrict__ in, const int8_t* __restrict__ w,
    const float* __restrict__ bias, int8_t* __restrict__ out,
    int64_t total, int64_t C, int64_t D, int64_t H, int64_t W, int64_t M,
    int64_t Cg, int64_t kD, int64_t kH, int64_t kW, int64_t OD, int64_t OH,
    int64_t OW, int64_t groups, int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW, int64_t dD, int64_t dH, int64_t dW,
    double input_scale, double input_zero_point, double weight_scale,
    double weight_zero_point, double inv_out_scale, double out_zero_point) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t ow = idx % OW;
    const int64_t oh = (idx / OW) % OH;
    const int64_t od = (idx / (OW * OH)) % OD;
    const int64_t m = (idx / (OW * OH * OD)) % M;
    const int64_t n = idx / (OW * OH * OD * M);
    const int64_t per_group_out = M / groups;
    const int64_t g = m / per_group_out;
    int64_t acc = 0;
    for (int64_t kd = 0; kd < kD; ++kd) {
        const int64_t id = od * sD - pD + kd * dD;
        if (id < 0 || id >= D) continue;
        for (int64_t kh = 0; kh < kH; ++kh) {
            const int64_t ih = oh * sH - pH + kh * dH;
            if (ih < 0 || ih >= H) continue;
            for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t iw = ow * sW - pW + kw * dW;
                if (iw < 0 || iw >= W) continue;
                for (int64_t c = 0; c < Cg; ++c) {
                    const int64_t in_c = g * Cg + c;
                    const int8_t xv =
                        in[((n * C + in_c) * D + id) * H * W + ih * W + iw];
                    const int8_t wv =
                        w[((m * Cg + c) * kD + kd) * kH * kW + kh * kW + kw];
                    acc += static_cast<int64_t>(xv - input_zero_point) *
                           static_cast<int64_t>(wv - weight_zero_point);
                }
            }
        }
    }
    double y = static_cast<double>(acc) * (input_scale * weight_scale);
    if (bias != nullptr) y += static_cast<double>(bias[m]);
    out[idx] = quant::requantize_value(y, inv_out_scale,
                                       static_cast<int64_t>(out_zero_point));
}

// Implicit-GEMM quantized 2d convolution with shared-memory tiling.
// Block mapping:
//   blockIdx.x -> spatial column tile (BN consecutive flattened OH*OW cols)
//   blockIdx.y -> output channel tile within the current group
//   blockIdx.z -> n * groups + g
// Each thread accumulates a TM x TN sub-tile of the BM x BN output tile.
// The K dimension (Cg*kH*kW) is streamed in BK-wide chunks; A and B tiles
// hold int16 differences (q - zero_point) so the inner product is a plain
// int32 accumulation with no per-element zero-point fixup.
__global__ void quantized_conv2d_gemm_kernel(
    const int8_t* __restrict__ in, const int8_t* __restrict__ w,
    const float* __restrict__ bias, int8_t* __restrict__ out,
    int64_t C, int64_t H, int64_t W, int64_t M, int64_t Cg,
    int64_t kH, int64_t kW, int64_t OH, int64_t OW, int64_t groups,
    int64_t sH, int64_t sW, int64_t pH, int64_t pW, int64_t dH, int64_t dW,
    int64_t per_group_out, int64_t K,
    double input_scale, int64_t input_zero_point, double weight_scale,
    int64_t weight_zero_point, double inv_out_scale, int64_t out_zero_point) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int THREADS = (BM / TM) * (BN / TN); // 256
    __shared__ int16_t A_s[BM * BK];
    __shared__ int16_t B_s[BK * BN];

    const int tid = threadIdx.x;
    const int tid_i = tid / (BN / TN);
    const int tid_j = tid % (BN / TN);

    const int groups_i = static_cast<int>(groups);
    const int n = blockIdx.z;
    const int m0 = static_cast<int>(blockIdx.y) * BM;
    const int p0 = static_cast<int>(blockIdx.x) * BN;
    const int i_zp = static_cast<int>(input_zero_point);
    const int w_zp = static_cast<int>(weight_zero_point);
    const int row_base = tid_i * TM;
    const int col_base = tid_j * TN;

    // One block owns one batch sample and walks the groups serially; the
    // tiles are re-loaded per group so grouped convolutions never rely on
    // cross-block input reuse.
    for (int g = 0; g < groups_i; ++g) {
        const int in_c_base = g * static_cast<int>(Cg);
        const int out_c_base = g * static_cast<int>(per_group_out);
        int32_t acc[TM][TN];
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) acc[i][j] = 0;
        }
        for (int k0 = 0; k0 < K; k0 += BK) {
            for (int idx = tid; idx < BM * BK; idx += THREADS) {
                const int row = idx / BK;
                const int kk = idx % BK;
                const int gcol = k0 + kk;
                int16_t v = 0;
                if (gcol < K) {
                    const int m = out_c_base + m0 + row;
                    if (m < M) {
                        const int c = gcol / (kH * kW);
                        const int rem = gcol % (kH * kW);
                        const int kh = rem / kW;
                        const int kw = rem % kW;
                        v = static_cast<int16_t>(
                            static_cast<int>(
                                w[((static_cast<int64_t>(m) * Cg + c) * kH +
                                    kh) *
                                  kW + kw]) -
                            w_zp);
                    }
                }
                A_s[idx] = v;
            }
            for (int idx = tid; idx < BK * BN; idx += THREADS) {
                const int kk = idx / BN;
                const int p = idx % BN;
                const int gcol = k0 + kk;
                int16_t v = 0;
                if (gcol < K) {
                    const int p_global = p0 + p;
                    if (p_global < OH * OW) {
                        const int c = gcol / (kH * kW);
                        const int rem = gcol % (kH * kW);
                        const int kh = rem / kW;
                        const int kw = rem % kW;
                        const int oh = p_global / OW;
                        const int ow = p_global % OW;
                        const int ih =
                            static_cast<int>(oh * sH - pH + kh * dH);
                        const int iw =
                            static_cast<int>(ow * sW - pW + kw * dW);
                        int8_t xv;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                            xv = static_cast<int8_t>(input_zero_point);
                        } else {
                            xv = in[((static_cast<int64_t>(n) * C +
                                     in_c_base + c) *
                                        H +
                                    ih) *
                                       W +
                                   iw];
                        }
                        v = static_cast<int16_t>(static_cast<int>(xv) - i_zp);
                    }
                }
                B_s[idx] = v;
            }
            __syncthreads();

            for (int kk = 0; kk < BK; ++kk) {
                for (int i = 0; i < TM; ++i) {
                    const int a =
                        static_cast<int>(A_s[(row_base + i) * BK + kk]);
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j] +=
                            a * static_cast<int>(B_s[kk * BN + col_base + j]);
                    }
                }
            }
            __syncthreads();
        }

        for (int i = 0; i < TM; ++i) {
            const int m = out_c_base + m0 + row_base + i;
            if (m >= M) continue;
            for (int j = 0; j < TN; ++j) {
                const int p_global = p0 + col_base + j;
                if (p_global >= OH * OW) continue;
                const int oh = p_global / OW;
                const int ow = p_global % OW;
                double y = static_cast<double>(acc[i][j]) *
                           (input_scale * weight_scale);
                if (bias != nullptr) y += static_cast<double>(bias[m]);
                out[((static_cast<int64_t>(n) * M + m) * OH + oh) * OW + ow] =
                    quant::requantize_value(y, inv_out_scale,
                                            static_cast<int64_t>(
                                                out_zero_point));
            }
        }
    }
}

} // namespace

Tensor quantized_conv2d_cuda(
    const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& bias, double input_scale,
    int64_t input_zero_point, double weight_scale, int64_t weight_zero_point,
    double out_scale, int64_t out_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dtype() != DType::QInt8 || weight.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "quantized_conv2d(): activations and weights must be QInt8");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_conv2d(): out_scale must be positive");
    }
    if (input.dim() != 4 || weight.dim() != 4) {
        TP_THROW(ValueError,
                 "quantized_conv2d(): expected 4d activations and weights");
    }
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t H = input.size(2), W = input.size(3);
    const int64_t M = weight.size(0), Cg = weight.size(1);
    const int64_t kH = weight.size(2), kW = weight.size(3);
    const int64_t sH = stride.empty() ? 1 : stride[0];
    const int64_t sW = stride.size() > 1 ? stride[1] : sH;
    const int64_t pH = padding.empty() ? 0 : padding[0];
    const int64_t pW = padding.size() > 1 ? padding[1] : pH;
    const int64_t dH = dilation.empty() ? 1 : dilation[0];
    const int64_t dW = dilation.size() > 1 ? dilation[1] : dH;
    if (groups <= 0 || M % groups != 0 || C % groups != 0) {
        TP_THROW(ValueError, "quantized_conv2d(): invalid groups");
    }
    if (Cg != C / groups) {
        TP_THROW(ValueError,
                 "quantized_conv2d(): weight channels do not match groups");
    }
    const int64_t OH = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    const int64_t OW = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;

    const Tensor in = input.contiguous();
    const Tensor w = weight.contiguous();
    Tensor bias_f;
    if (bias.has_value()) {
        bias_f = bias->to(DType::Float32).contiguous();
    }
    Tensor out = Tensor::empty({N, M, OH, OW}, DType::QInt8, input.device());
    const int64_t total = N * M * OH * OW;
    if (total == 0) {
        out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
            out_scale, out_zero_point, DType::QInt8));
        return out;
    }
    const int64_t K = Cg * kH * kW;
    const int64_t per_group_out = M / groups;
    const int threads = 256;
    static const bool force_naive =
        std::getenv("TP_QUANT_CONV_NAIVE") != nullptr;
    if (!force_naive && Cg > 1 && K <= 32768 && N <= 65535 &&
        per_group_out > 0) {
        const dim3 grid(static_cast<unsigned>((OH * OW + 63) / 64),
                        static_cast<unsigned>((per_group_out + 63) / 64),
                        static_cast<unsigned>(N));
        quantized_conv2d_gemm_kernel<<<grid, threads, 0,
                                      getCurrentCUDAStream().stream()>>>(
            in.data_ptr<int8_t>(), w.data_ptr<int8_t>(),
            bias_f.defined() ? bias_f.data_ptr<float>() : nullptr,
            out.data_ptr<int8_t>(), C, H, W, M, Cg, kH, kW, OH, OW, groups,
            sH, sW, pH, pW, dH, dW, per_group_out, K, input_scale,
            input_zero_point, weight_scale, weight_zero_point, 1.0 / out_scale,
            out_zero_point);
    } else {
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        quantized_conv2d_kernel<<<blocks, threads, 0,
                                  getCurrentCUDAStream().stream()>>>(
            in.data_ptr<int8_t>(), w.data_ptr<int8_t>(),
            bias_f.defined() ? bias_f.data_ptr<float>() : nullptr,
            out.data_ptr<int8_t>(), total, C, H, W, M, Cg, kH, kW, OH, OW,
            groups, sH, sW, pH, pW, dH, dW, input_scale,
            static_cast<double>(input_zero_point), weight_scale,
            static_cast<double>(weight_zero_point), 1.0 / out_scale,
            static_cast<double>(out_zero_point));
    }
    checkCuda(cudaGetLastError(), "CUDA quantized_conv2d kernel");
    out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
        out_scale, out_zero_point, DType::QInt8));
    return out;
}

Tensor quantized_conv1d_cuda(
    const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& bias, double input_scale,
    int64_t input_zero_point, double weight_scale, int64_t weight_zero_point,
    double out_scale, int64_t out_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dtype() != DType::QInt8 || weight.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "quantized_conv1d(): activations and weights must be QInt8");
    }
    if (input.dim() != 3 || weight.dim() != 3) {
        TP_THROW(ValueError,
                 "quantized_conv1d(): expected 3d activations and weights");
    }
    const int64_t n = input.size(0), c = input.size(1), l = input.size(2);
    const int64_t o = weight.size(0), cw = weight.size(1), k = weight.size(2);
    Tensor in4 = quantized::make_qtensor(
        quantized::strip_quantizer(input).contiguous().view({n, c, 1, l}),
        input.impl()->quantizer(), DType::QInt8);
    Tensor w4 = quantized::make_qtensor(
        quantized::strip_quantizer(weight).contiguous().view({o, cw, 1, k}),
        weight.impl()->quantizer(), DType::QInt8);
    auto expand1 = [](const std::vector<int64_t>& v, int64_t fill) {
        return v.empty() ? std::vector<int64_t>{fill, fill}
                         : std::vector<int64_t>{fill, v[0]};
    };
    const std::vector<int64_t> stride2 = expand1(stride, 1);
    const std::vector<int64_t> padding2 = expand1(padding, 0);
    const std::vector<int64_t> dilation2 = expand1(dilation, 1);
    Tensor out4 = quantized_conv2d_cuda(
        in4, w4, bias, input_scale, input_zero_point, weight_scale,
        weight_zero_point, out_scale, out_zero_point, stride2, padding2,
        dilation2, groups);
    const int64_t l_out =
        static_cast<std::vector<int64_t>>(out4.shape())[3];
    Tensor out = quantized::strip_quantizer(out4).contiguous().view({n, o, l_out});
    return quantized::make_qtensor(
        out, make_per_tensor_affine_quantizer(out_scale, out_zero_point,
                                              DType::QInt8),
        DType::QInt8);
}

Tensor quantized_conv3d_cuda(
    const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& bias, double input_scale,
    int64_t input_zero_point, double weight_scale, int64_t weight_zero_point,
    double out_scale, int64_t out_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dtype() != DType::QInt8 || weight.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "quantized_conv3d(): activations and weights must be QInt8");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_conv3d(): out_scale must be positive");
    }
    if (input.dim() != 5 || weight.dim() != 5) {
        TP_THROW(ValueError,
                 "quantized_conv3d(): expected 5d activations and weights");
    }
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t D = input.size(2), H = input.size(3), W = input.size(4);
    const int64_t M = weight.size(0), Cg = weight.size(1);
    const int64_t kD = weight.size(2), kH = weight.size(3), kW = weight.size(4);
    const int64_t sD = stride.empty() ? 1 : stride[0];
    const int64_t sH = stride.size() > 1 ? stride[1] : sD;
    const int64_t sW = stride.size() > 2 ? stride[2] : sD;
    const int64_t pD = padding.empty() ? 0 : padding[0];
    const int64_t pH = padding.size() > 1 ? padding[1] : pD;
    const int64_t pW = padding.size() > 2 ? padding[2] : pD;
    const int64_t dD = dilation.empty() ? 1 : dilation[0];
    const int64_t dH = dilation.size() > 1 ? dilation[1] : dD;
    const int64_t dW = dilation.size() > 2 ? dilation[2] : dD;
    if (groups <= 0 || M % groups != 0 || C % groups != 0) {
        TP_THROW(ValueError, "quantized_conv3d(): invalid groups");
    }
    if (Cg != C / groups) {
        TP_THROW(ValueError,
                 "quantized_conv3d(): weight channels do not match groups");
    }
    const int64_t OD = (D + 2 * pD - dD * (kD - 1) - 1) / sD + 1;
    const int64_t OH = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    const int64_t OW = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;

    const Tensor in = input.contiguous();
    const Tensor w = weight.contiguous();
    Tensor bias_f;
    if (bias.has_value()) {
        bias_f = bias->to(DType::Float32).contiguous();
    }
    Tensor out =
        Tensor::empty({N, M, OD, OH, OW}, DType::QInt8, input.device());
    const int64_t total = N * M * OD * OH * OW;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    quantized_conv3d_kernel<<<blocks, threads, 0,
                              getCurrentCUDAStream().stream()>>>(
        in.data_ptr<int8_t>(), w.data_ptr<int8_t>(),
        bias_f.defined() ? bias_f.data_ptr<float>() : nullptr,
        out.data_ptr<int8_t>(), total, C, D, H, W, M, Cg, kD, kH, kW, OD, OH,
        OW, groups, sD, sH, sW, pD, pH, pW, dD, dH, dW, input_scale,
        static_cast<double>(input_zero_point), weight_scale,
        static_cast<double>(weight_zero_point), 1.0 / out_scale,
        static_cast<double>(out_zero_point));
    checkCuda(cudaGetLastError(), "CUDA quantized_conv3d kernel");
    out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
        out_scale, out_zero_point, DType::QInt8));
    return out;
}

std::tuple<Tensor, Tensor> quantized_conv2d_prepack_cuda(
    const Tensor& weight, const Tensor& weight_scales,
    const Tensor& weight_zero_points, const std::optional<Tensor>& bias,
    bool transposed) {
    TP_CHECK(weight.dim() == 4,
             "quantized conv prepack: expected a 4-D weight");
    TP_CHECK(weight_scales.dim() == 1 &&
                 weight_zero_points.shape() == weight_scales.shape(),
             "quantized conv prepack: scales/zero_points must be 1-D");
    const int64_t O_raw = weight.size(0);
    const int64_t C_raw = weight.size(1);
    const int64_t KH = weight.size(2);
    const int64_t KW = weight.size(3);
    const int64_t param_channels = transposed ? C_raw : O_raw;
    TP_CHECK(weight_scales.size(0) == param_channels,
             "quantized conv prepack: scales length must match the weight's "
             "output channel count");

    const Tensor w = weight.contiguous().to(DType::Float32);
    const float* pw = w.data_ptr<float>();
    const Tensor sc = weight_scales.to(DType::Float32).contiguous();
    const Tensor zp = weight_zero_points.to(DType::Float32).contiguous();
    const float* psc = sc.data_ptr<float>();
    const float* pzp = zp.data_ptr<float>();
    Tensor wq = Tensor::empty(w.shape(), DType::Float32, w.device());
    float* pwq = wq.data_ptr<float>();
    for (int64_t o = 0; o < O_raw; ++o) {
        for (int64_t c = 0; c < C_raw; ++c) {
            const float s = transposed ? psc[c] : psc[o];
            const float z = transposed ? pzp[c] : pzp[o];
            const int64_t base = (o * C_raw + c) * KH * KW;
            for (int64_t i = 0; i < KH * KW; ++i) {
                pwq[base + i] = (pw[base + i] - z) * s;
            }
        }
    }
    Tensor bias_packed;
    if (bias.has_value()) {
        bias_packed = bias->to(DType::Float32).contiguous();
    }
    return std::make_tuple(wq.contiguous(), bias_packed);
}

std::tuple<Tensor, Tensor> quantized_conv2d_unpack_cuda(
    const Tensor& weight_packed, const Tensor& bias_packed,
    const std::vector<int64_t>& weight_sizes, bool transposed, bool depthwise) {
    TP_CHECK(weight_sizes.size() == 4,
             "quantized conv unpack: expected a 4-D weight size list");
    const int64_t O = weight_sizes[0];
    const int64_t C = weight_sizes[1];
    const int64_t KH = weight_sizes[2];
    const int64_t KW = weight_sizes[3];
    Tensor w = weight_packed.reshape({O, C, KH, KW}).contiguous();
    Tensor b;
    if (bias_packed.defined()) {
        b = bias_packed.contiguous();
    }
    return std::make_tuple(w, b);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantConvKernels) {
    m.impl("quantized_conv1d", quantized_conv1d_cuda);
    m.impl("quantized_conv2d", quantized_conv2d_cuda);
    m.impl("quantized_conv3d", quantized_conv3d_cuda);
    m.impl("quantized_conv2d_prepack", quantized_conv2d_prepack_cuda);
    m.impl("quantized_conv2d_unpack", quantized_conv2d_unpack_cuda);
}

} // namespace cuda
} // namespace tensorplay