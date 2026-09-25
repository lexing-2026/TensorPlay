#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

Tensor max_pool2d_backward_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size_arg,
    const std::vector<int64_t>& stride_arg,
    const std::vector<int64_t>& padding_arg,
    const std::vector<int64_t>& dilation_arg, bool ceil_mode);

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

template <typename T, typename M>
__global__ void max_pool2d_wi_fwd_kernel(
    int64_t total, int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW,
    int64_t pH, int64_t pW, int64_t dH, int64_t dW,
    const T* __restrict__ input, T* __restrict__ output,
    int64_t* __restrict__ indices) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t out_spatial = H_out * W_out;
    for (; i < total; i += stride) {
        const int64_t w = i % W_out;
        const int64_t h = (i / W_out) % H_out;
        const int64_t nc = i / out_spatial;
        const T* plane = input + nc * H_in * W_in;
        M max_val = -std::numeric_limits<M>::infinity();
        int64_t max_idx = -1;
        for (int64_t kh = 0; kh < kH; ++kh) {
            const int64_t hi = h * sH - pH + kh * dH;
            if (hi < 0 || hi >= H_in) continue;
            for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t wi = w * sW - pW + kw * dW;
                if (wi < 0 || wi >= W_in) continue;
                const int64_t idx = hi * W_in + wi;
                const M val = static_cast<M>(plane[idx]);
                if ((val > max_val) || ::isnan(val)) {
                    max_val = val;
                    max_idx = idx;
                }
            }
        }
        output[i] = static_cast<T>(max_val);
        indices[i] = max_idx;
    }
}

template <typename T>
__global__ void max_pool_wi_bwd_kernel(
    int64_t total, int64_t out_spatial, int64_t in_plane,
    const T* __restrict__ grad_output, const int64_t* __restrict__ indices,
    T* __restrict__ grad_input) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; i < total; i += stride) {
        const int64_t idx = indices[i];
        if (idx < 0) continue;
        const int64_t nc = i / out_spatial;
        gpuAtomicAdd(grad_input + nc * in_plane + idx, grad_output[i]);
    }
}

}

#define POOL_CUDA_DISPATCH(ctype, name, ...)                                \
    case DType::name: {                                                     \
        using M = typename PoolMath<ctype>::type;                           \
        __VA_ARGS__;                                                        \
        break;                                                              \
    }

std::tuple<Tensor, Tensor> max_pool2d_with_indices_cuda(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 3) {
        auto result = max_pool2d_with_indices_cuda(
            input.unsqueeze(0), kernel_size, stride, padding, dilation,
            ceil_mode);
        return std::make_tuple(std::get<0>(result).squeeze(0),
                               std::get<1>(result).squeeze(0));
    }
    if (input.dim() != 4) {
        TP_THROW(RuntimeError, "max_pool2d_with_indices: Expected 4D input");
    }
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t H_in = input_c.size(2), W_in = input_c.size(3);

    const auto ks = pool_expand_param(
        kernel_size, "max_pool2d_with_indices kernel_size", 2, 1);
    const auto st = pool_expand_param(
        stride.empty() ? ks : stride, "max_pool2d_with_indices stride", 2,
        ks[0]);
    const auto pd = pool_expand_param(
        padding, "max_pool2d_with_indices padding", 2, 0);
    const auto dl = pool_expand_param(
        dilation, "max_pool2d_with_indices dilation", 2, 1);
    const int64_t kH = ks[0], kW = ks[1], sH = st[0], sW = st[1];
    const int64_t pH = pd[0], pW = pd[1], dH = dl[0], dW = dl[1];

    const int64_t H_out = pool_output_shape(H_in, kH, pH, sH, dH, ceil_mode);
    const int64_t W_out = pool_output_shape(W_in, kW, pW, sW, dW, ceil_mode);
    if (H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError,
                 "max_pool2d_with_indices: Calculated output size is too small");
    }

    Tensor out = Tensor::empty(
        {N, C, H_out, W_out}, input.dtype(), input.device());
    Tensor indices = Tensor::empty(
        {N, C, H_out, W_out}, DType::Int64, input.device());
    const int64_t total = N * C * H_out * W_out;
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();

    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            max_pool2d_wi_fwd_kernel<float, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                dH, dW, input_c.data_ptr<float>(), out.data_ptr<float>(),
                indices.data_ptr<int64_t>()))
        POOL_CUDA_DISPATCH(double, Float64,
            max_pool2d_wi_fwd_kernel<double, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                dH, dW, input_c.data_ptr<double>(), out.data_ptr<double>(),
                indices.data_ptr<int64_t>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (max_pool2d_wi_fwd_kernel<tensorplay::Half, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                dH, dW, input_c.data_ptr<tensorplay::Half>(),
                out.data_ptr<tensorplay::Half>(), indices.data_ptr<int64_t>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (max_pool2d_wi_fwd_kernel<tensorplay::BFloat16, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                dH, dW, input_c.data_ptr<tensorplay::BFloat16>(),
                out.data_ptr<tensorplay::BFloat16>(), indices.data_ptr<int64_t>())))
        default:
            TP_THROW(NotImplementedError,
                     "max_pool2d_with_indices CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return std::make_tuple(out, indices);
}

Tensor max_pool2d_with_indices_backward_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    bool ceil_mode, const std::optional<Tensor>& indices_opt) {
    (void)kernel_size;
    (void)stride;
    (void)padding;
    (void)dilation;
    (void)ceil_mode;
    if (!indices_opt.has_value() || !indices_opt->defined()) {
        TP_THROW(RuntimeError,
                 "max_pool2d_with_indices_backward: indices is required");
    }
    if (input.dim() == 3) {
        return max_pool2d_with_indices_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0), kernel_size,
                   stride, padding, dilation, ceil_mode,
                   indices_opt->unsqueeze(0)).squeeze(0);
    }
    if (grad_output.dim() != 4 || input.dim() != 4) {
        TP_THROW(RuntimeError,
                 "max_pool2d_with_indices_backward: Expected 4D input and grad_output");
    }
    const Tensor& idx_shape_ref = *indices_opt;
    if (idx_shape_ref.dim() != 4 || idx_shape_ref.size(0) != grad_output.size(0) ||
        idx_shape_ref.size(1) != grad_output.size(1) ||
        idx_shape_ref.size(2) != grad_output.size(2) ||
        idx_shape_ref.size(3) != grad_output.size(3)) {
        TP_THROW(RuntimeError, "max_pool2d_with_indices_backward: expected grad_output with shape [",
                 grad_output.size(0), ", ", grad_output.size(1), ", ",
                 grad_output.size(2), ", ", grad_output.size(3),
                 "] to match indices shape [", idx_shape_ref.size(0), ", ",
                 idx_shape_ref.size(1), ", ", idx_shape_ref.size(2), ", ",
                 idx_shape_ref.size(3), "]");
    }
    const Tensor go = grad_output.contiguous();
    const Tensor idx = indices_opt->contiguous();
    Tensor grad_input = Tensor::zeros_like(input);
    const int64_t total = go.numel();
    const int64_t out_spatial = go.size(2) * go.size(3);
    const int64_t in_plane = input.size(2) * input.size(3);
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            max_pool_wi_bwd_kernel<float><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<float>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            max_pool_wi_bwd_kernel<double><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<double>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (max_pool_wi_bwd_kernel<tensorplay::Half><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<tensorplay::Half>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (max_pool_wi_bwd_kernel<tensorplay::BFloat16><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<tensorplay::BFloat16>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "max_pool2d_with_indices_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

Tensor max_pool2d_backward_native_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size_arg,
    const std::vector<int64_t>& stride_arg,
    const std::vector<int64_t>& padding_arg,
    const std::vector<int64_t>& dilation_arg, bool ceil_mode) {
    Tensor indices = std::get<1>(max_pool2d_with_indices_cuda(
        input, kernel_size_arg, stride_arg, padding_arg, dilation_arg,
        ceil_mode));
    return max_pool2d_with_indices_backward_cuda(
        grad_output, input, kernel_size_arg, stride_arg, padding_arg,
        dilation_arg, ceil_mode, indices);
}

std::tuple<Tensor, Tensor> interop_max_pool2d_with_indices_out_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out,
    Tensor& indices) {
    auto result = max_pool2d_with_indices_cuda(
        self, kernel_size, stride, padding, dilation, ceil_mode);
    write_pooling_out("max_pool2d_with_indices", std::get<0>(result), out);
    write_pooling_out("max_pool2d_with_indices", std::get<1>(result), indices);
    return {out, indices};
}

Tensor& interop_max_pool2d_with_indices_backward_grad_input_cuda(
    const Tensor& grad_output, const Tensor& self,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode,
    const Tensor& indices, Tensor& grad_input) {
    return write_pooling_out(
        "max_pool2d_with_indices_backward",
        max_pool2d_with_indices_backward_cuda(
            grad_output, self, kernel_size, stride, padding, dilation,
            ceil_mode, indices),
        grad_input);
}

#undef POOL_CUDA_DISPATCH

TENSORPLAY_LIBRARY_IMPL(CUDA, MaxPool2dKernels) {
#ifdef USE_ROCM
    auto max_backward = max_pool2d_backward_native_cuda;
#else
    auto max_backward = max_pool2d_backward_cuda;
#endif
    m.impl("max_pool2d_backward", max_backward);
    m.impl("max_pool2d_with_indices", max_pool2d_with_indices_cuda);
    m.impl("max_pool2d_with_indices_backward", max_pool2d_with_indices_backward_cuda);
    m.impl("max_pool2d_with_indices.out", interop_max_pool2d_with_indices_out_cuda);
    m.impl("max_pool2d_with_indices_backward.grad_input",
           interop_max_pool2d_with_indices_backward_grad_input_cuda);
}

}
}
