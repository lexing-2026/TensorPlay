// Double-precision convolutions for the CUDA backend: im2col + GEMM, with
// no DNN library involvement.  The column matrix is laid out row-major
// (CP, L) with CP = C/group * kH * kW planes of L output sites; the GEMMs
// run through gemm_impl, which maps row-major operands onto the
// column-major BLAS call internally.  Split from ConvKernels.cu so the
// reference fp64 path and the cuDNN paths compile independently.

#include "Tensor.h"
#include "Convolution.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "CudaGemm.h"
#include "ConvIm2colKernels.cuh"

#include <cuda_runtime.h>

#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

inline std::vector<int64_t> expand_param_if_needed(
    const std::vector<int64_t>& list, int64_t n, int64_t default_val) {
    if (list.empty()) return std::vector<int64_t>(n, default_val);
    if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
    if (list.size() != n) TP_THROW(ValueError, "Parameter size mismatch");
    return list;
}

} // namespace

// =========================================================================
// Double-precision convolutions: im2col + GEMM, no DNN library involved.
//
// Column matrix layout (shared with the im2col kernel above): row-major
// (CP, L) with CP = C/group * kH * kW planes of L output sites.  The GEMMs
// run through gemm_impl, which consumes row-major 2-D tensors and maps them
// onto the column-major BLAS call internally, so every operand below is
// spelled directly in that row-major layout.  Overlapping strided / dilated
// patches make each output site a sum over kernel positions, so the batch
// GEMMs accumulate with beta=1 into an output pre-seeded with the bias.
// =========================================================================

namespace {

void slow_conv_fp64_shape_check(const Tensor& input, const Tensor& grad_output,
                                const Tensor& weight, int64_t groups,
                                const std::vector<int64_t>& stride,
                                const std::vector<int64_t>& padding,
                                const std::vector<int64_t>& dilation) {
    TP_CHECK(input.dim() == 4, "conv2d: expected 4D input");
    TP_CHECK(weight.dim() == 4, "conv2d: expected 4D weight");
    TP_CHECK(groups >= 1 && weight.size(0) % groups == 0 && input.size(1) % groups == 0,
             "conv2d: channel counts must be divisible by groups");
    const int64_t C = input.size(1);
    const int64_t KH = weight.size(2), KW = weight.size(3);
    TP_CHECK(KH > 0 && KW > 0, "conv2d: kernel size must be positive");
    TP_CHECK(stride[0] > 0 && stride[1] > 0, "conv2d: stride must be positive");
    TP_CHECK(dilation[0] > 0 && dilation[1] > 0, "conv2d: dilation must be positive");
    TP_CHECK(padding[0] >= 0 && padding[1] >= 0, "conv2d: padding must be non-negative");
    const int64_t IH = input.size(2), IW = input.size(3);
    const int64_t OH = (IH + 2 * padding[0] - dilation[0] * (KH - 1) - 1) / stride[0] + 1;
    const int64_t OW = (IW + 2 * padding[1] - dilation[1] * (KW - 1) - 1) / stride[1] + 1;
    TP_CHECK(OH >= 1 && OW >= 1,
             "conv2d: calculated shape is too small (output ", OH, "x", OW, ")");
    TP_CHECK(weight.size(1) == C / groups,
             "conv2d: expected ", C / groups, " input channels per group but got ",
             weight.size(1));
    if (grad_output.defined()) {
        TP_CHECK(grad_output.dim() == 4, "conv2d: expected 4D grad_output");
        TP_CHECK(grad_output.size(1) == weight.size(0),
                 "conv2d: grad_output channel count does not match weight");
        TP_CHECK(grad_output.size(2) == OH && grad_output.size(3) == OW,
                 "conv2d: grad_output spatial shape does not match the computed output");
    }
}

// One im2col launch over a single (batch, group) input frame into a dense
// (CP, L) column matrix.
void im2col_frame_fp64(const double* in, double* col, int64_t C, int64_t H, int64_t W,
                       int64_t kH, int64_t kW, int64_t pH, int64_t pW,
                       int64_t sH, int64_t sW, int64_t dH, int64_t dW,
                       int64_t OH, int64_t OW) {
    const int64_t total = C * kH * kW * OH * OW;
    if (total == 0) return;
    const dim3 block(256, 1, 1);
    const dim3 grid(cuda_blocks(total, 256), 1, 1);
    im2col_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
        in, col, /*N=*/1, C, H, W, kH, kW, pH, pW, sH, sW, dH, dW, OH, OW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("im2col fp64: ") + cudaGetErrorString(err));
    }
}

// Inverse of im2col_frame_fp64: scatters one (CP, L) column matrix back into
// a single (C, H, W) frame (overwrite, not accumulate; frames never overlap).
void col2im_frame_fp64(const double* col, double* im, int64_t C, int64_t H, int64_t W,
                       int64_t kH, int64_t kW, int64_t pH, int64_t pW,
                       int64_t sH, int64_t sW, int64_t dH, int64_t dW,
                       int64_t OH, int64_t OW) {
    const int64_t frame = C * H * W;
    if (frame == 0) return;
    const dim3 block(128, 1, 1);
    const dim3 grid(cuda_blocks(frame, 128), 1, 1);
    col2im_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
        col, im, C, H, W, kH, kW, pH, pW, sH, sW, dH, dW, OH, OW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("col2im fp64: ") + cudaGetErrorString(err));
    }
}

}  // namespace

Tensor conv2d_slow_fp64(const Tensor& input, const Tensor& weight, const Tensor& bias,
                        const std::vector<int64_t>& stride_arg,
                        const std::vector<int64_t>& padding_arg,
                        const std::vector<int64_t>& dilation_arg, int64_t groups) {
    const std::vector<int64_t> stride = expand_param_if_needed(stride_arg, 2, 1);
    const std::vector<int64_t> padding = expand_param_if_needed(padding_arg, 2, 0);
    const std::vector<int64_t> dilation = expand_param_if_needed(dilation_arg, 2, 1);
    slow_conv_fp64_shape_check(input, Tensor(), weight, groups, stride, padding, dilation);

    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor w = weight.is_contiguous() ? weight : weight.contiguous();

    const int64_t N = in.size(0), C = in.size(1);
    const int64_t H = in.size(2), W = in.size(3);
    const int64_t KH = w.size(2), KW = w.size(3);
    const int64_t OH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) / stride[0] + 1;
    const int64_t OW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) / stride[1] + 1;
    const int64_t K_out = w.size(0);
    const int64_t Cg = C / groups;
    const int64_t Kg = K_out / groups;
    const int64_t CP = Cg * KH * KW;
    const int64_t L = OH * OW;

    Tensor out = Tensor::empty({N, K_out, OH, OW}, DType::Float64, in.device());
    if (out.numel() == 0) return out;
    if (bias.defined() && bias.numel() != 0) {
        TP_CHECK(bias.dim() == 1 && bias.size(0) == K_out,
                 "conv2d: bias must be 1D with one element per output channel");
        Tensor bias_c = bias.is_contiguous() ? bias : bias.contiguous();
        out.copy_(bias_c.view({1, K_out, 1, 1}).expand({N, K_out, OH, OW}));
    } else {
        out.zero_();
    }

    // 1x1 stride-1 undilated convolutions need no im2col at all; the input
    // frame is already the column matrix in that case.
    const bool requires_columns =
        (KW != 1 || KH != 1 || stride[0] != 1 || stride[1] != 1 ||
         padding[0] != 0 || padding[1] != 0 || dilation[0] != 1 || dilation[1] != 1);
    Tensor columns;
    if (requires_columns) {
        columns = Tensor::empty({CP, L}, DType::Float64, in.device());
    }

    Tensor w2 = w.view({K_out, CP});
    const double* in_base = in.data_ptr<double>();
    for (int64_t n = 0; n < N; ++n) {
        const double* in_n = in_base + n * C * H * W;
        Tensor out_n = out.select(0, n);
        for (int64_t g = 0; g < groups; ++g) {
            if (CP == 0 || L == 0) continue;
            Tensor cols = requires_columns
                ? columns
                : in.select(0, n).slice(0, g * Cg, (g + 1) * Cg).reshape({Cg, L});
            if (requires_columns) {
                im2col_frame_fp64(in_n + static_cast<int64_t>(g) * Cg * H * W,
                                  columns.data_ptr<double>(), Cg, H, W,
                                  KH, KW, padding[0], padding[1],
                                  stride[0], stride[1], dilation[0], dilation[1], OH, OW);
            }
            Tensor out_g = out_n.narrow(0, g * Kg, Kg).reshape({Kg, L});
            gemm_impl(w2.narrow(0, g * Kg, Kg), cols, out_g, 1.0, 1.0, nullptr);
        }
    }
    return out;
}

Tensor conv2d_slow_fp64_grad_input(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& weight,
                                   const std::vector<int64_t>& stride_arg,
                                   const std::vector<int64_t>& padding_arg,
                                   const std::vector<int64_t>& dilation_arg, int64_t groups) {
    const std::vector<int64_t> stride = expand_param_if_needed(stride_arg, 2, 1);
    const std::vector<int64_t> padding = expand_param_if_needed(padding_arg, 2, 0);
    const std::vector<int64_t> dilation = expand_param_if_needed(dilation_arg, 2, 1);
    slow_conv_fp64_shape_check(input, grad_output, weight, groups, stride, padding, dilation);

    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor w = weight.is_contiguous() ? weight : weight.contiguous();

    const int64_t N = input.size(0), C = input.size(1);
    const int64_t H = input.size(2), W = input.size(3);
    const int64_t KH = w.size(2), KW = w.size(3);
    const int64_t OH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) / stride[0] + 1;
    const int64_t OW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) / stride[1] + 1;
    const int64_t K_out = w.size(0);
    const int64_t Cg = C / groups;
    const int64_t Kg = K_out / groups;
    const int64_t CP = Cg * KH * KW;
    const int64_t L = OH * OW;

    Tensor grad_input = Tensor::empty({N, C, H, W}, input.dtype(), input.device());
    if (grad_input.numel() == 0 || go.numel() == 0) return grad_input;

    // grad_columns = w^T @ grad_out per (batch, group); the transposed weight
    // is materialized once per group so every GEMM operand stays dense.
    Tensor w2 = w.view({K_out, CP});
    std::vector<Tensor> w_t;
    w_t.reserve(static_cast<size_t>(groups));
    for (int64_t g = 0; g < groups; ++g) {
        w_t.push_back(w2.narrow(0, g * Kg, Kg).t().contiguous());
    }

    Tensor columns = Tensor::empty({CP, L}, input.dtype(), input.device());
    for (int64_t n = 0; n < N; ++n) {
        Tensor go_n = go.select(0, n);
        for (int64_t g = 0; g < groups; ++g) {
            if (CP == 0 || L == 0) continue;
            Tensor go_g = go_n.narrow(0, g * Kg, Kg).reshape({Kg, L});
            gemm_impl(w_t[static_cast<size_t>(g)], go_g, columns, 1.0, 0.0, nullptr);
            col2im_frame_fp64(columns.data_ptr<double>(),
                              grad_input.data_ptr<double>() +
                                  (n * C + g * Cg) * H * W,
                              Cg, H, W, KH, KW, padding[0], padding[1],
                              stride[0], stride[1], dilation[0], dilation[1], OH, OW);
        }
    }
    return grad_input;
}

Tensor conv2d_slow_fp64_grad_weight(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& weight,
                                    const std::vector<int64_t>& stride_arg,
                                    const std::vector<int64_t>& padding_arg,
                                    const std::vector<int64_t>& dilation_arg, int64_t groups) {
    const std::vector<int64_t> stride = expand_param_if_needed(stride_arg, 2, 1);
    const std::vector<int64_t> padding = expand_param_if_needed(padding_arg, 2, 0);
    const std::vector<int64_t> dilation = expand_param_if_needed(dilation_arg, 2, 1);
    slow_conv_fp64_shape_check(input, grad_output, weight, groups, stride, padding, dilation);

    Tensor in = input.is_contiguous() ? input : input.contiguous();
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor w = weight.is_contiguous() ? weight : weight.contiguous();

    const int64_t N = in.size(0), C = in.size(1);
    const int64_t H = in.size(2), W = in.size(3);
    const int64_t KH = w.size(2), KW = w.size(3);
    const int64_t OH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) / stride[0] + 1;
    const int64_t OW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) / stride[1] + 1;
    const int64_t K_out = w.size(0);
    const int64_t Cg = C / groups;
    const int64_t Kg = K_out / groups;
    const int64_t CP = Cg * KH * KW;
    const int64_t L = OH * OW;

    Tensor grad_weight = Tensor::empty(w.shape(), go.dtype(), go.device());
    if (grad_weight.numel() == 0) return grad_weight;
    // Every (batch, group) GEMM accumulates onto the zeroed buffer below.
    grad_weight.zero_();

    // 1x1 stride-1 undilated convolutions need no im2col at all; the input
    // frame is already the column matrix in that case.
    const bool requires_columns =
        (KW != 1 || KH != 1 || stride[0] != 1 || stride[1] != 1 ||
         padding[0] != 0 || padding[1] != 0 || dilation[0] != 1 || dilation[1] != 1);
    Tensor columns;
    if (requires_columns) {
        columns = Tensor::empty({CP, L}, go.dtype(), in.device());
    }

    Tensor gw2 = grad_weight.view({K_out, CP});
    const double* in_base = in.data_ptr<double>();
    for (int64_t n = 0; n < N; ++n) {
        const double* in_n = in_base + n * C * H * W;
        Tensor go_n = go.select(0, n);
        for (int64_t g = 0; g < groups; ++g) {
            if (CP == 0 || L == 0) continue;
            Tensor cols = requires_columns
                ? columns
                : in.select(0, n).slice(0, g * Cg, (g + 1) * Cg).reshape({Cg, L});
            if (requires_columns) {
                im2col_frame_fp64(in_n + static_cast<int64_t>(g) * Cg * H * W,
                                  columns.data_ptr<double>(), Cg, H, W,
                                  KH, KW, padding[0], padding[1],
                                  stride[0], stride[1], dilation[0], dilation[1], OH, OW);
            }
            Tensor go_g = go_n.narrow(0, g * Kg, Kg).reshape({Kg, L});
            Tensor gw_g = gw2.narrow(0, g * Kg, Kg);
            // grad_weight += grad_out @ columns^T; the transposed columns are
            // a live view, which the GEMM path consumes without a copy.
            gemm_impl(go_g, cols.t(), gw_g, 1.0, 1.0, nullptr);
        }
    }
    return grad_weight;
}

Tensor conv2d_slow_fp64_grad_bias(const Tensor& grad_output, const Tensor& input,
                                  const Tensor& weight,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation, int64_t groups) {
    (void)input;
    (void)weight;
    (void)stride;
    (void)padding;
    (void)dilation;
    (void)groups;
    // The gradient of the per-channel bias add is a spatial reduction.
    return grad_output.sum(std::vector<int64_t>{0, 2, 3});
}

} // namespace cuda
} // namespace tensorplay
