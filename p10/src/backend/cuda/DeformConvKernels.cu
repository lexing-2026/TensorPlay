// deform_conv2d / deform_conv2d_backward CUDA kernels.
//
// Forward: one thread per output position; each kernel tap samples the input
// bilinearly at the offset-shifted dilation-spaced position, multiplied by the
// optional mask tap. Out-of-range neighbours contribute zero.
// Backward: one thread per output position distributes the gradient into
// input / weight / offset / mask / bias through atomics.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "Atomic.cuh"
#include "Half.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace tensorplay {
namespace cuda {
namespace {

constexpr int kThreads = 256;

inline int64_t ceil_div_blocks(int64_t n, int64_t threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return blocks > 65535 ? 65535 : blocks;
}

// Bilinear sample with per-corner zero padding. Always fills the corner values
// and interpolation weights so the backward pass can differentiate even the
// border-clipped case (out-of-range corners stay zero, so the position
// derivative vanishes there).
template <typename T, typename CT>
__device__ inline T deform_bilinear_sample_cuda(
        const T* data, int64_t height, int64_t width,
        CT y, CT x, int64_t& y_low, int64_t& x_low,
        CT& ly, CT& lx, CT& hy, CT& hx,
        CT* v1, CT* v2, CT* v3, CT* v4) {
    y_low = static_cast<int64_t>(floor(y));
    x_low = static_cast<int64_t>(floor(x));
    *v1 = *v2 = *v3 = *v4 = CT(0);
    ly = lx = hy = hx = CT(0);
    if (y <= CT(-1) || CT(height) <= y || x <= CT(-1) || CT(width) <= x) {
        return T(0);
    }
    ly = y - y_low;
    lx = x - x_low;
    hy = CT(1) - ly;
    hx = CT(1) - lx;
    *v1 = (y_low >= 0 && x_low >= 0) ? static_cast<CT>(data[y_low * width + x_low]) : CT(0);
    *v2 = (y_low >= 0 && x_low + 1 <= width - 1) ? static_cast<CT>(data[y_low * width + x_low + 1]) : CT(0);
    *v3 = (y_low + 1 <= height - 1 && x_low >= 0) ? static_cast<CT>(data[(y_low + 1) * width + x_low]) : CT(0);
    *v4 = (y_low + 1 <= height - 1 && x_low + 1 <= width - 1) ? static_cast<CT>(data[(y_low + 1) * width + x_low + 1]) : CT(0);
    return T(hy * hx * (*v1) + hy * lx * (*v2) + ly * hx * (*v3) + ly * lx * (*v4));
}

template <typename storage_t, typename acc_t>
__global__ void deform_conv2d_forward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ weight,
        const storage_t* __restrict__ offset,
        const storage_t* __restrict__ mask,
        const storage_t* __restrict__ bias,
        storage_t* __restrict__ output,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const int64_t OC, const int64_t IC, const int64_t kh, const int64_t kw,
        const int64_t out_h, const int64_t out_w,
        const int64_t stride_h, const int64_t stride_w,
        const int64_t pad_h, const int64_t pad_w,
        const int64_t dil_h, const int64_t dil_w,
        const int64_t in_per_group, const int64_t out_per_group,
        const int64_t c_per_off, const int64_t offset_groups,
        const bool use_mask) {
    const int64_t out_per_batch = OC * out_h * out_w;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t ox = index % out_w;
        const int64_t oy = (index / out_w) % out_h;
        const int64_t oc = (index / (out_w * out_h)) % OC;
        const int64_t n = index / out_per_batch;
        const int64_t g_w = oc / out_per_group;
        const storage_t* in_n = input + n * C * H * W;
        const storage_t* off_n = offset + n * 2 * offset_groups * kh * kw * out_h * out_w;
        const storage_t* mask_n = use_mask ? mask + n * offset_groups * kh * kw * out_h * out_w : nullptr;
        const storage_t* w_oc = weight + oc * in_per_group * kh * kw;
        acc_t val = bias ? static_cast<acc_t>(bias[oc]) : acc_t(0);
        for (int64_t i = 0; i < kh; ++i) {
            for (int64_t j = 0; j < kw; ++j) {
                const int64_t k = i * kw + j;
                for (int64_t ic = 0; ic < in_per_group; ++ic) {
                    const int64_t in_c = g_w * in_per_group + ic;
                    const int64_t g_o = in_c / c_per_off;
                    const acc_t off_h = static_cast<acc_t>(
                        off_n[(2 * g_o * kh * kw + 2 * k) * out_h * out_w + oy * out_w + ox]);
                    const acc_t off_w = static_cast<acc_t>(
                        off_n[(2 * g_o * kh * kw + 2 * k + 1) * out_h * out_w + oy * out_w + ox]);
                    const acc_t y = static_cast<acc_t>(oy) * stride_h - pad_h + i * dil_h + off_h;
                    const acc_t x = static_cast<acc_t>(ox) * stride_w - pad_w + j * dil_w + off_w;
                    int64_t y_low, x_low;
                    acc_t ly, lx, hy, hx, v1, v2, v3, v4;
                    acc_t sample = static_cast<acc_t>(deform_bilinear_sample_cuda(
                        in_n + in_c * H * W, H, W, y, x,
                        y_low, x_low, ly, lx, hy, hx, &v1, &v2, &v3, &v4));
                    if (use_mask) {
                        sample *= static_cast<acc_t>(
                            mask_n[(g_o * kh * kw + k) * out_h * out_w + oy * out_w + ox]);
                    }
                    val += static_cast<acc_t>(w_oc[(ic * kh + i) * kw + j]) * sample;
                }
            }
        }
        output[index] = static_cast<storage_t>(val);
    }
}

template <typename storage_t, typename acc_t>
__global__ void deform_conv2d_backward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ grad_output,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ weight,
        const storage_t* __restrict__ offset,
        const storage_t* __restrict__ mask,
        storage_t* __restrict__ grad_input,
        storage_t* __restrict__ grad_weight,
        storage_t* __restrict__ grad_offset,
        storage_t* __restrict__ grad_mask,
        storage_t* __restrict__ grad_bias,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const int64_t OC, const int64_t IC, const int64_t kh, const int64_t kw,
        const int64_t out_h, const int64_t out_w,
        const int64_t stride_h, const int64_t stride_w,
        const int64_t pad_h, const int64_t pad_w,
        const int64_t dil_h, const int64_t dil_w,
        const int64_t in_per_group, const int64_t out_per_group,
        const int64_t c_per_off, const int64_t offset_groups,
        const bool use_mask,
        const bool need_input, const bool need_weight, const bool need_offset,
        const bool need_mask, const bool need_bias) {
    const int64_t out_per_batch = OC * out_h * out_w;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t ox = index % out_w;
        const int64_t oy = (index / out_w) % out_h;
        const int64_t oc = (index / (out_w * out_h)) % OC;
        const int64_t n = index / out_per_batch;
        const acc_t g = static_cast<acc_t>(grad_output[index]);
        if (g == acc_t(0)) continue;
        const int64_t g_w = oc / out_per_group;
        const storage_t* in_n = input + n * C * H * W;
        const storage_t* off_n = offset + n * 2 * offset_groups * kh * kw * out_h * out_w;
        const storage_t* mask_n = use_mask ? mask + n * offset_groups * kh * kw * out_h * out_w : nullptr;
        const storage_t* w_oc = weight + oc * in_per_group * kh * kw;
        storage_t* g_in_n = grad_input + n * C * H * W;
        storage_t* g_off_n = grad_offset + n * 2 * offset_groups * kh * kw * out_h * out_w;
        storage_t* g_mask_n = grad_mask + n * offset_groups * kh * kw * out_h * out_w;
        for (int64_t i = 0; i < kh; ++i) {
            for (int64_t j = 0; j < kw; ++j) {
                const int64_t k = i * kw + j;
                for (int64_t ic = 0; ic < in_per_group; ++ic) {
                    const int64_t in_c = g_w * in_per_group + ic;
                    const int64_t g_o = in_c / c_per_off;
                    const acc_t off_h = static_cast<acc_t>(
                        off_n[(2 * g_o * kh * kw + 2 * k) * out_h * out_w + oy * out_w + ox]);
                    const acc_t off_w = static_cast<acc_t>(
                        off_n[(2 * g_o * kh * kw + 2 * k + 1) * out_h * out_w + oy * out_w + ox]);
                    const acc_t y = static_cast<acc_t>(oy) * stride_h - pad_h + i * dil_h + off_h;
                    const acc_t x = static_cast<acc_t>(ox) * stride_w - pad_w + j * dil_w + off_w;
                    int64_t y_low, x_low;
                    acc_t ly, lx, hy, hx, v1, v2, v3, v4;
                    const acc_t sample = static_cast<acc_t>(deform_bilinear_sample_cuda(
                        in_n + in_c * H * W, H, W, y, x,
                        y_low, x_low, ly, lx, hy, hx, &v1, &v2, &v3, &v4));
                    acc_t mval = acc_t(1);
                    if (use_mask) {
                        mval = static_cast<acc_t>(
                            mask_n[(g_o * kh * kw + k) * out_h * out_w + oy * out_w + ox]);
                    }
                    const acc_t w_val = static_cast<acc_t>(w_oc[(ic * kh + i) * kw + j]);
                    if (need_weight) {
                        gpuAtomicAdd(&grad_weight[(oc * in_per_group + ic) * kh * kw + k],
                                     static_cast<storage_t>(g * sample * mval));
                    }
                    if (need_bias && i == 0 && j == 0 && ic == 0) {
                        gpuAtomicAdd(&grad_bias[oc], static_cast<storage_t>(g));
                    }
                    const acc_t contrib = g * w_val * mval;
                    if (need_input) {
                        if (y_low >= 0 && x_low >= 0) {
                            gpuAtomicAdd(&g_in_n[in_c * H * W + y_low * W + x_low],
                                         static_cast<storage_t>(contrib * hy * hx));
                        }
                        if (y_low >= 0 && x_low + 1 <= W - 1) {
                            gpuAtomicAdd(&g_in_n[in_c * H * W + y_low * W + x_low + 1],
                                         static_cast<storage_t>(contrib * hy * lx));
                        }
                        if (y_low + 1 <= H - 1 && x_low >= 0) {
                            gpuAtomicAdd(&g_in_n[in_c * H * W + (y_low + 1) * W + x_low],
                                         static_cast<storage_t>(contrib * ly * hx));
                        }
                        if (y_low + 1 <= H - 1 && x_low + 1 <= W - 1) {
                            gpuAtomicAdd(&g_in_n[in_c * H * W + (y_low + 1) * W + x_low + 1],
                                         static_cast<storage_t>(contrib * ly * lx));
                        }
                    }
                    if (need_offset) {
                        const acc_t dval_dy = -hx * v1 - lx * v2 + hx * v3 + lx * v4;
                        const acc_t dval_dx = -hy * v1 + hy * v2 - ly * v3 + ly * v4;
                        gpuAtomicAdd(
                            &g_off_n[(2 * g_o * kh * kw + 2 * k) * out_h * out_w + oy * out_w + ox],
                            static_cast<storage_t>(contrib * dval_dy));
                        gpuAtomicAdd(
                            &g_off_n[(2 * g_o * kh * kw + 2 * k + 1) * out_h * out_w + oy * out_w + ox],
                            static_cast<storage_t>(contrib * dval_dx));
                    }
                    if (need_mask && use_mask) {
                        gpuAtomicAdd(
                            &g_mask_n[(g_o * kh * kw + k) * out_h * out_w + oy * out_w + ox],
                            static_cast<storage_t>(g * w_val * sample));
                    }
                }
            }
        }
    }
}

} // namespace

template <typename storage_t, typename acc_t>
static Tensor deform_conv2d_cuda_impl(
        const Tensor& input, const Tensor& weight, const Tensor& offset,
        const Tensor& mask, const std::optional<Tensor>& bias,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, int64_t groups,
        int64_t offset_groups, bool use_mask) {
    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int64_t OC = weight.size(0), IC = weight.size(1), kh = weight.size(2), kw = weight.size(3);
    const int64_t ker_h = dilation[0] * (kh - 1) + 1;
    const int64_t ker_w = dilation[1] * (kw - 1) + 1;
    const int64_t out_h = (H + 2 * padding[0] - ker_h) / stride[0] + 1;
    const int64_t out_w = (W + 2 * padding[1] - ker_w) / stride[1] + 1;
    const int64_t in_per_group = C / groups;
    const int64_t out_per_group = OC / groups;
    const int64_t c_per_off = C / offset_groups;
    Tensor output = Tensor::empty({N, OC, out_h, out_w}, input.dtype(), input.device());
    const int64_t nthreads = N * OC * out_h * out_w;
    if (nthreads == 0) return output;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    deform_conv2d_forward_kernel<storage_t, acc_t><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
        nthreads,
        input.data_ptr<storage_t>(), weight.data_ptr<storage_t>(),
        offset.data_ptr<storage_t>(),
        use_mask ? mask.data_ptr<storage_t>() : nullptr,
        bias.has_value() ? bias->data_ptr<storage_t>() : nullptr,
        output.data_ptr<storage_t>(),
        N, C, H, W, OC, IC, kh, kw, out_h, out_w,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        in_per_group, out_per_group, c_per_off, offset_groups, use_mask);
    return output;
}

template <typename storage_t, typename acc_t>
static std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> deform_conv2d_backward_cuda_impl(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        const Tensor& offset, const Tensor& mask, const std::optional<Tensor>& bias,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, int64_t groups,
        int64_t offset_groups, bool use_mask, const std::vector<bool>& output_mask) {
    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int64_t OC = weight.size(0), IC = weight.size(1), kh = weight.size(2), kw = weight.size(3);
    const int64_t ker_h = dilation[0] * (kh - 1) + 1;
    const int64_t ker_w = dilation[1] * (kw - 1) + 1;
    const int64_t out_h = (H + 2 * padding[0] - ker_h) / stride[0] + 1;
    const int64_t out_w = (W + 2 * padding[1] - ker_w) / stride[1] + 1;
    const int64_t in_per_group = C / groups;
    const int64_t out_per_group = OC / groups;
    const int64_t c_per_off = C / offset_groups;
    const bool need_input = output_mask.size() > 0 && output_mask[0];
    const bool need_weight = output_mask.size() > 1 && output_mask[1];
    const bool need_offset = output_mask.size() > 2 && output_mask[2];
    const bool need_mask = output_mask.size() > 3 && output_mask[3];
    const bool need_bias = output_mask.size() > 4 && output_mask[4];
    Tensor grad_input = Tensor::zeros({N, C, H, W}, grad_output.dtype(), grad_output.device());
    Tensor grad_weight = Tensor::zeros({OC, IC, kh, kw}, grad_output.dtype(), grad_output.device());
    Tensor grad_offset = Tensor::zeros({N, 2 * offset_groups * kh * kw, out_h, out_w},
                                       grad_output.dtype(), grad_output.device());
    Tensor grad_mask = use_mask
        ? Tensor::zeros({N, offset_groups * kh * kw, out_h, out_w}, grad_output.dtype(), grad_output.device())
        : Tensor::zeros({1}, grad_output.dtype(), grad_output.device());
    Tensor grad_bias = bias.has_value()
        ? Tensor::zeros({OC}, grad_output.dtype(), grad_output.device())
        : Tensor::zeros({1}, grad_output.dtype(), grad_output.device());
    const int64_t nthreads = N * OC * out_h * out_w;
    if (nthreads == 0) {
        return {grad_input, grad_weight, grad_offset, grad_mask, grad_bias};
    }
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    deform_conv2d_backward_kernel<storage_t, acc_t><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
        nthreads,
        grad_output.data_ptr<storage_t>(),
        input.data_ptr<storage_t>(), weight.data_ptr<storage_t>(),
        offset.data_ptr<storage_t>(),
        use_mask ? mask.data_ptr<storage_t>() : nullptr,
        grad_input.data_ptr<storage_t>(), grad_weight.data_ptr<storage_t>(),
        grad_offset.data_ptr<storage_t>(), grad_mask.data_ptr<storage_t>(),
        grad_bias.data_ptr<storage_t>(),
        N, C, H, W, OC, IC, kh, kw, out_h, out_w,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        in_per_group, out_per_group, c_per_off, offset_groups, use_mask,
        need_input, need_weight, need_offset, need_mask, need_bias);
    return {grad_input, grad_weight, grad_offset, grad_mask, grad_bias};
}

Tensor deform_conv2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& offset,
                          const Tensor& mask, const std::optional<Tensor>& bias,
                          const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                          const std::vector<int64_t>& dilation, int64_t groups,
                          int64_t offset_groups, bool use_mask) {
    if (input.dim() != 4 || weight.dim() != 4 || offset.dim() != 4)
        TP_THROW(RuntimeError, "deform_conv2d: expected 4-D input, weight and offset");
    if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2)
        TP_THROW(RuntimeError, "deform_conv2d: stride, padding and dilation must have 2 entries");
    if (input.dtype() != weight.dtype() || input.dtype() != offset.dtype() ||
        (use_mask && input.dtype() != mask.dtype()))
        TP_THROW(RuntimeError, "deform_conv2d: input, weight, offset and mask must have the same dtype");
    const Tensor ic = input.contiguous();
    const Tensor wc = weight.contiguous();
    const Tensor oc = offset.contiguous();
    std::optional<Tensor> bc = std::nullopt;
    if (bias.has_value()) bc = bias->contiguous();
    const Tensor mc = use_mask ? mask.contiguous() : mask;
    switch (ic.dtype()) {
        case DType::Float32:
            return deform_conv2d_cuda_impl<float, float>(ic, wc, oc, mc, bc, stride, padding, dilation, groups, offset_groups, use_mask);
        case DType::Float64:
            return deform_conv2d_cuda_impl<double, double>(ic, wc, oc, mc, bc, stride, padding, dilation, groups, offset_groups, use_mask);
        case DType::Float16:
            return deform_conv2d_cuda_impl<Half, float>(ic, wc, oc, mc, bc, stride, padding, dilation, groups, offset_groups, use_mask);
        default: TP_THROW(TypeError, "deform_conv2d: unsupported dtype");
    }
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> deform_conv2d_backward_cuda(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        const Tensor& offset, const Tensor& mask, const std::optional<Tensor>& bias,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, int64_t groups,
        int64_t offset_groups, bool use_mask, const std::vector<bool>& output_mask) {
    if (output_mask.size() != 5)
        TP_THROW(RuntimeError, "deform_conv2d_backward: output_mask must have 5 entries");
    const Tensor gc = grad_output.contiguous();
    const Tensor ic = input.contiguous();
    const Tensor wc = weight.contiguous();
    const Tensor oc = offset.contiguous();
    std::optional<Tensor> bc = std::nullopt;
    if (bias.has_value()) bc = bias->contiguous();
    const Tensor mc = use_mask ? mask.contiguous() : mask;
    switch (ic.dtype()) {
        case DType::Float32:
            return deform_conv2d_backward_cuda_impl<float, float>(gc, ic, wc, oc, mc, bc, stride, padding, dilation, groups, offset_groups, use_mask, output_mask);
        case DType::Float64:
            return deform_conv2d_backward_cuda_impl<double, double>(gc, ic, wc, oc, mc, bc, stride, padding, dilation, groups, offset_groups, use_mask, output_mask);
        case DType::Float16:
            return deform_conv2d_backward_cuda_impl<Half, float>(gc, ic, wc, oc, mc, bc, stride, padding, dilation, groups, offset_groups, use_mask, output_mask);
        default: TP_THROW(TypeError, "deform_conv2d_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CUDA, DeformConvKernels) {
    m.impl("deform_conv2d", deform_conv2d_cuda);
    m.impl("deform_conv2d_backward", deform_conv2d_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay