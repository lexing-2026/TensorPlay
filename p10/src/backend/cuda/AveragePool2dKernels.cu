#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

Tensor avg_pool2d_cuda(const Tensor& input,
                       const std::vector<int64_t>& kernel_size_arg,
                       const std::vector<int64_t>& stride_arg,
                       const std::vector<int64_t>& padding_arg,
                       bool ceil_mode, bool count_include_pad,
                       std::optional<int64_t> divisor_override);

Tensor avg_pool2d_backward_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size_arg,
    const std::vector<int64_t>& stride_arg,
    const std::vector<int64_t>& padding_arg, bool ceil_mode,
    bool count_include_pad, std::optional<int64_t> divisor_override);

namespace {

template <typename T> struct PoolMath { using type = T; };
template <> struct PoolMath<tensorplay::Half> { using type = float; };
template <> struct PoolMath<tensorplay::BFloat16> { using type = float; };

int64_t pool_grid_blocks(int64_t n, int threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return blocks > 65535 ? 65535 : blocks;
}

int64_t pool_div_rtn(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

int64_t pool_output_shape(int64_t in, int64_t k, int64_t pad,
                          int64_t stride, int64_t dilation, bool ceil_mode) {
    if (stride == 0) TP_THROW(RuntimeError, "stride should not be zero");
    int64_t out = pool_div_rtn(
        in + 2 * pad - dilation * (k - 1) - 1 +
            (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode && (out - 1) * stride >= in + pad) --out;
    return out;
}

std::vector<int64_t> pool_expand_param(const std::vector<int64_t>& list,
                                       const char* name, int64_t n,
                                       int64_t default_val) {
    if (list.empty()) return std::vector<int64_t>(n, default_val);
    if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
    if (static_cast<int64_t>(list.size()) != n) {
        TP_THROW(ValueError,
                 std::string(name) + ": expected " + std::to_string(n) + " values");
    }
    return list;
}

Tensor& write_pooling_out(const char* op, Tensor value, Tensor& out) {
    if (!out.defined()) {
        out = std::move(value);
        return out;
    }
    if (out.dtype() != value.dtype()) {
        TP_THROW(TypeError, op, ": output dtype must match result dtype");
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 op, ": output device must match input device");
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

struct AvgPoolWindow {
    int64_t hstart, hend, wstart, wend, pool_size;
};

__device__ inline AvgPoolWindow avg_pool_window(
    int64_t h, int64_t w, int64_t H_in, int64_t W_in, int64_t kH, int64_t kW,
    int64_t sH, int64_t sW, int64_t pH, int64_t pW) {
    AvgPoolWindow win;
    win.hstart = h * sH - pH;
    win.wstart = w * sW - pW;
    win.hend = min(win.hstart + kH, H_in + pH);
    win.wend = min(win.wstart + kW, W_in + pW);
    win.pool_size = (win.hend - win.hstart) * (win.wend - win.wstart);
    win.hstart = max(win.hstart, int64_t(0));
    win.wstart = max(win.wstart, int64_t(0));
    win.hend = min(win.hend, H_in);
    win.wend = min(win.wend, W_in);
    return win;
}

__device__ inline int64_t avg_pool_divisor(const AvgPoolWindow& win,
                                           bool count_include_pad,
                                           bool use_divisor,
                                           int64_t divisor_override) {
    if (use_divisor) return divisor_override;
    return count_include_pad
        ? win.pool_size
        : (win.hend - win.hstart) * (win.wend - win.wstart);
}

template <typename T, typename M>
__global__ void avg_pool2d_fwd_kernel(
    int64_t total, int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW, int64_t pH, int64_t pW,
    bool count_include_pad, bool use_divisor, int64_t divisor_override,
    const T* __restrict__ input, T* __restrict__ output) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t out_spatial = H_out * W_out;
    for (; i < total; i += stride) {
        const int64_t w = i % W_out;
        const int64_t h = (i / W_out) % H_out;
        const int64_t nc = i / out_spatial;
        const T* plane = input + nc * H_in * W_in;
        const AvgPoolWindow win =
            avg_pool_window(h, w, H_in, W_in, kH, kW, sH, sW, pH, pW);
        if (win.hstart >= win.hend || win.wstart >= win.wend) {
            output[i] = T(0);
            continue;
        }
        M acc = M(0);
        for (int64_t hi = win.hstart; hi < win.hend; ++hi) {
            for (int64_t wi = win.wstart; wi < win.wend; ++wi) {
                acc += static_cast<M>(plane[hi * W_in + wi]);
            }
        }
        const M divisor = static_cast<M>(avg_pool_divisor(
            win, count_include_pad, use_divisor, divisor_override));
        output[i] = static_cast<T>(acc / divisor);
    }
}

template <typename T, typename M>
__global__ void avg_pool2d_bwd_kernel(
    int64_t total, int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW, int64_t pH, int64_t pW,
    bool count_include_pad, bool use_divisor, int64_t divisor_override,
    const T* __restrict__ grad_output, T* __restrict__ grad_input) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t out_spatial = H_out * W_out;
    const int64_t in_plane = H_in * W_in;
    for (; i < total; i += stride) {
        const int64_t w = i % W_out;
        const int64_t h = (i / W_out) % H_out;
        const int64_t nc = i / out_spatial;
        const AvgPoolWindow win =
            avg_pool_window(h, w, H_in, W_in, kH, kW, sH, sW, pH, pW);
        if (win.hstart >= win.hend || win.wstart >= win.wend) continue;
        const M divisor = static_cast<M>(avg_pool_divisor(
            win, count_include_pad, use_divisor, divisor_override));
        const M g = static_cast<M>(grad_output[i]) / divisor;
        T* plane = grad_input + nc * in_plane;
        for (int64_t hi = win.hstart; hi < win.hend; ++hi) {
            for (int64_t wi = win.wstart; wi < win.wend; ++wi) {
                gpuAtomicAdd(plane + hi * W_in + wi, static_cast<T>(g));
            }
        }
    }
}

}

#define POOL_CUDA_DISPATCH(ctype, name, ...)                                \
    case DType::name: {                                                     \
        using M = typename PoolMath<ctype>::type;                           \
        __VA_ARGS__;                                                        \
        break;                                                              \
    }

Tensor avg_pool2d_native_cuda(const Tensor& input,
                              const std::vector<int64_t>& kernel_size_arg,
                              const std::vector<int64_t>& stride_arg,
                              const std::vector<int64_t>& padding_arg,
                              bool ceil_mode, bool count_include_pad,
                              std::optional<int64_t> divisor_override) {
    if (input.dim() == 3) {
        return avg_pool2d_native_cuda(input.unsqueeze(0), kernel_size_arg,
                                      stride_arg, padding_arg, ceil_mode,
                                      count_include_pad, divisor_override).squeeze(0);
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "avg_pool2d: Expected 3D or 4D input");
    if (divisor_override.has_value() && *divisor_override == 0)
        TP_THROW(RuntimeError, "divisor must be not zero");
    const bool use_divisor = divisor_override.has_value();
    const int64_t divisor_value = divisor_override.value_or(0);
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t H_in = input_c.size(2), W_in = input_c.size(3);

    const auto ks = pool_expand_param(kernel_size_arg, "avg_pool2d kernel_size", 2, 1);
    const auto st = pool_expand_param(stride_arg.empty() ? kernel_size_arg : stride_arg,
                                      "avg_pool2d stride", 2, ks[0]);
    const auto pd = pool_expand_param(padding_arg, "avg_pool2d padding", 2, 0);
    const int64_t kH = ks[0], kW = ks[1], sH = st[0], sW = st[1];
    const int64_t pH = pd[0], pW = pd[1];

    const int64_t H_out = pool_output_shape(H_in, kH, pH, sH, 1, ceil_mode);
    const int64_t W_out = pool_output_shape(W_in, kW, pW, sW, 1, ceil_mode);
    if (H_out <= 0 || W_out <= 0)
        TP_THROW(RuntimeError, "avg_pool2d: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    const int64_t total = N * C * H_out * W_out;
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            avg_pool2d_fwd_kernel<float, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, use_divisor, divisor_value, input_c.data_ptr<float>(), out.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            avg_pool2d_fwd_kernel<double, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, use_divisor, divisor_value, input_c.data_ptr<double>(), out.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (avg_pool2d_fwd_kernel<tensorplay::Half, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, use_divisor, divisor_value, input_c.data_ptr<tensorplay::Half>(),
                out.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (avg_pool2d_fwd_kernel<tensorplay::BFloat16, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, use_divisor, divisor_value, input_c.data_ptr<tensorplay::BFloat16>(),
                out.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "avg_pool2d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return out;
}

Tensor avg_pool2d_backward_native_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size_arg,
    const std::vector<int64_t>& stride_arg,
    const std::vector<int64_t>& padding_arg, bool ceil_mode,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
    if (input.dim() == 3) {
        return avg_pool2d_backward_native_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0), kernel_size_arg,
                   stride_arg, padding_arg, ceil_mode, count_include_pad,
                   divisor_override).squeeze(0);
    }
    if (input.dim() != 4)
        TP_THROW(RuntimeError, "avg_pool2d_backward: Expected 3D or 4D input");
    if (divisor_override.has_value() && *divisor_override == 0)
        TP_THROW(RuntimeError, "divisor must be not zero");
    const bool use_divisor = divisor_override.has_value();
    const int64_t divisor_value = divisor_override.value_or(0);
    const Tensor input_c = input.contiguous();
    const Tensor go = grad_output.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t H_in = input_c.size(2), W_in = input_c.size(3);

    const auto ks = pool_expand_param(kernel_size_arg, "avg_pool2d_backward kernel_size", 2, 1);
    const auto st = pool_expand_param(stride_arg.empty() ? kernel_size_arg : stride_arg,
                                      "avg_pool2d_backward stride", 2, ks[0]);
    const auto pd = pool_expand_param(padding_arg, "avg_pool2d_backward padding", 2, 0);
    const int64_t kH = ks[0], kW = ks[1], sH = st[0], sW = st[1];
    const int64_t pH = pd[0], pW = pd[1];

    const int64_t H_out = pool_output_shape(H_in, kH, pH, sH, 1, ceil_mode);
    const int64_t W_out = pool_output_shape(W_in, kW, pW, sW, 1, ceil_mode);
    if (go.size(0) != N || go.size(1) != C || go.size(2) != H_out ||
        go.size(3) != W_out)
        TP_THROW(RuntimeError, "avg_pool2d_backward: grad_output shape mismatch");

    Tensor grad_input = Tensor::zeros({N, C, H_in, W_in}, input.dtype(), input.device());
    const int64_t total = go.numel();
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            avg_pool2d_bwd_kernel<float, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, use_divisor, divisor_value, go.data_ptr<float>(), grad_input.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            avg_pool2d_bwd_kernel<double, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, use_divisor, divisor_value, go.data_ptr<double>(), grad_input.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (avg_pool2d_bwd_kernel<tensorplay::Half, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, use_divisor, divisor_value, go.data_ptr<tensorplay::Half>(),
                grad_input.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (avg_pool2d_bwd_kernel<tensorplay::BFloat16, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, use_divisor, divisor_value, go.data_ptr<tensorplay::BFloat16>(),
                grad_input.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "avg_pool2d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

Tensor& interop_avg_pool2d_out_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, Tensor& out) {
    return write_pooling_out(
        "avg_pool2d",
        avg_pool2d_native_cuda(self, kernel_size, stride, padding, ceil_mode,
                               count_include_pad, divisor_override),
        out);
}

Tensor& interop_avg_pool2d_backward_grad_input_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override, Tensor& grad_input) {
    return write_pooling_out(
        "avg_pool2d_backward",
        avg_pool2d_backward_native_cuda(
            grad_output, input, kernel_size, stride, padding, ceil_mode,
            count_include_pad, divisor_override),
        grad_input);
}

#undef POOL_CUDA_DISPATCH

TENSORPLAY_LIBRARY_IMPL(CUDA, AveragePool2dKernels) {
#ifdef USE_ROCM
    auto avg_forward = avg_pool2d_native_cuda;
    auto avg_backward = avg_pool2d_backward_native_cuda;
#else
    auto avg_forward = avg_pool2d_cuda;
    auto avg_backward = avg_pool2d_backward_cuda;
#endif
    m.impl("avg_pool2d", avg_forward);
    m.impl("avg_pool2d_backward", avg_backward);
    m.impl("avg_pool2d.out", interop_avg_pool2d_out_cuda);
    m.impl("avg_pool2d_backward.grad_input",
           interop_avg_pool2d_backward_grad_input_cuda);
}

}
}
