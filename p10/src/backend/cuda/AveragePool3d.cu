// Native CUDA average-pooling implementation. The two kernels keep the
// window arithmetic: the divisor includes the padded part when
// count_include_pad is true, and ceil_mode applies the last-window correction
// before accumulation. The backward path uses output-parallel scatter and
// dtype-aware CUDA atomics.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

template <typename T>
inline T div_rtn(T a, T b) {
    T q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

inline int64_t pooling_output_shape(int64_t input, int64_t kernel,
                                    int64_t pad, int64_t stride,
                                    bool ceil_mode) {
    if (stride == 0) {
        TP_THROW(ValueError, "avg_pool3d: stride should not be zero");
    }
    if (pad < 0) {
        TP_THROW(ValueError, "avg_pool3d: pad must be non-negative");
    }
    if (kernel <= 0) {
        TP_THROW(ValueError, "avg_pool3d: kernel_size must be greater than zero");
    }
    if (pad > (kernel - 1) / 2) {
        TP_THROW(ValueError,
                 "avg_pool3d: pad should be at most half of the kernel size");
    }
    int64_t output = div_rtn<int64_t>(
        input + 2 * pad - kernel - 1 + (ceil_mode ? stride - 1 : 0) + 1,
        stride) + 1;
    if (ceil_mode && (output - 1) * stride >= input + pad) {
        --output;
    }
    return output;
}

inline std::array<int64_t, 3> expand_pool_parameter(
    const std::vector<int64_t>& value, const char* name,
    int64_t default_value, bool allow_empty) {
    if (value.empty()) {
        if (!allow_empty) {
            TP_THROW(ValueError, std::string("avg_pool3d: ") + name +
                     " must have one or three values");
        }
        return {default_value, default_value, default_value};
    }
    if (value.size() == 1) {
        return {value[0], value[0], value[0]};
    }
    if (value.size() == 3) {
        return {value[0], value[1], value[2]};
    }
    TP_THROW(ValueError, std::string("avg_pool3d: ") + name +
             " must have one or three values");
}

struct Pool3dParams {
    std::array<int64_t, 3> kernel;
    std::array<int64_t, 3> stride;
    std::array<int64_t, 3> padding;
};

Pool3dParams check_pool3d_args(const Tensor& input,
                               const std::vector<int64_t>& kernel_size,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               bool ceil_mode,
                               const char* op) {
    if (kernel_size.empty() ||
        (kernel_size.size() != 1 && kernel_size.size() != 3)) {
        TP_THROW(ValueError, std::string(op) +
                 ": kernel_size must be a single int, or a tuple of three ints");
    }
    if (input.dim() != 4 && input.dim() != 5) {
        TP_THROW(ValueError, std::string(op) +
                 ": expected a non-empty 4D or 5D tensor");
    }
    const auto kernel = expand_pool_parameter(kernel_size, "kernel_size", 0, false);
    const auto stride_values = stride.empty()
        ? std::array<int64_t, 3>{kernel[0], kernel[1], kernel[2]}
        : expand_pool_parameter(stride, "stride", 0, false);
    const auto padding_values = expand_pool_parameter(padding, "padding", 0, true);

    for (int64_t d = 0; d < 3; ++d) {
        if (kernel[d] <= 0 || stride_values[d] <= 0) {
            TP_THROW(ValueError, std::string(op) +
                     ": kernel_size and stride must be greater than zero");
        }
        if (padding_values[d] < 0 ||
            padding_values[d] > (kernel[d] - 1) / 2) {
            TP_THROW(ValueError, std::string(op) +
                     ": padding must be non-negative and at most half the kernel");
        }
        if (input.size(input.dim() - 3 + d) <= 0) {
            TP_THROW(ValueError, std::string(op) +
                     ": non-batch dimensions must be non-zero");
        }
    }

    const int64_t d_out = pooling_output_shape(
        input.size(-3), kernel[0], padding_values[0], stride_values[0], ceil_mode);
    const int64_t h_out = pooling_output_shape(
        input.size(-2), kernel[1], padding_values[1], stride_values[1], ceil_mode);
    const int64_t w_out = pooling_output_shape(
        input.size(-1), kernel[2], padding_values[2], stride_values[2], ceil_mode);
    if (d_out <= 0 || h_out <= 0 || w_out <= 0) {
        TP_THROW(ValueError, std::string(op) +
                 ": calculated output size is too small");
    }
    return {kernel, stride_values, padding_values};
}

template <typename T>
struct PoolAccum {
    using type = T;
};

template <>
struct PoolAccum<tensorplay::Half> {
    using type = float;
};

template <>
struct PoolAccum<tensorplay::BFloat16> {
    using type = float;
};

template <typename scalar_t, typename acc_t>
__global__ void avg_pool3d_forward_kernel(
    int64_t total,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t channels,
    int64_t input_d, int64_t input_h, int64_t input_w,
    int64_t output_d, int64_t output_h, int64_t output_w,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    bool count_include_pad, int64_t divisor_override) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const int64_t ow = index % output_w;
        const int64_t oh = (index / output_w) % output_h;
        const int64_t od = (index / (output_w * output_h)) % output_d;
        const int64_t nc = index / (output_w * output_h * output_d);

        const int64_t d_start_raw = od * stride_d - pad_d;
        const int64_t h_start_raw = oh * stride_h - pad_h;
        const int64_t w_start_raw = ow * stride_w - pad_w;
        const int64_t d_end_raw = min(d_start_raw + kernel_d, input_d + pad_d);
        const int64_t h_end_raw = min(h_start_raw + kernel_h, input_h + pad_h);
        const int64_t w_end_raw = min(w_start_raw + kernel_w, input_w + pad_w);

        const int64_t pool_size = (d_end_raw - d_start_raw) *
                                  (h_end_raw - h_start_raw) *
                                  (w_end_raw - w_start_raw);
        const int64_t d_start = max(d_start_raw, int64_t(0));
        const int64_t h_start = max(h_start_raw, int64_t(0));
        const int64_t w_start = max(w_start_raw, int64_t(0));
        const int64_t d_end = min(d_end_raw, input_d);
        const int64_t h_end = min(h_end_raw, input_h);
        const int64_t w_end = min(w_end_raw, input_w);

        if (d_start >= d_end || h_start >= h_end || w_start >= w_end) {
            output[index] = static_cast<scalar_t>(0);
            continue;
        }

        const int64_t valid_size = (d_end - d_start) *
                                   (h_end - h_start) *
                                   (w_end - w_start);
        const int64_t divisor = divisor_override != 0
            ? divisor_override
            : (count_include_pad ? pool_size : valid_size);
        const scalar_t* input_plane = input + nc * input_d * input_h * input_w;
        acc_t sum = static_cast<acc_t>(0);
        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                const scalar_t* row = input_plane + (d * input_h + h) * input_w;
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += static_cast<acc_t>(row[w]);
                }
            }
        }
        output[index] = static_cast<scalar_t>(sum / static_cast<acc_t>(divisor));
    }
}

template <typename scalar_t, typename acc_t>
__global__ void avg_pool3d_backward_kernel(
    int64_t total,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    int64_t input_d, int64_t input_h, int64_t input_w,
    int64_t output_d, int64_t output_h, int64_t output_w,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    bool count_include_pad, int64_t divisor_override) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const int64_t ow = index % output_w;
        const int64_t oh = (index / output_w) % output_h;
        const int64_t od = (index / (output_w * output_h)) % output_d;
        const int64_t nc = index / (output_w * output_h * output_d);

        const int64_t d_start_raw = od * stride_d - pad_d;
        const int64_t h_start_raw = oh * stride_h - pad_h;
        const int64_t w_start_raw = ow * stride_w - pad_w;
        const int64_t d_end_raw = min(d_start_raw + kernel_d, input_d + pad_d);
        const int64_t h_end_raw = min(h_start_raw + kernel_h, input_h + pad_h);
        const int64_t w_end_raw = min(w_start_raw + kernel_w, input_w + pad_w);
        const int64_t pool_size = (d_end_raw - d_start_raw) *
                                  (h_end_raw - h_start_raw) *
                                  (w_end_raw - w_start_raw);
        const int64_t d_start = max(d_start_raw, int64_t(0));
        const int64_t h_start = max(h_start_raw, int64_t(0));
        const int64_t w_start = max(w_start_raw, int64_t(0));
        const int64_t d_end = min(d_end_raw, input_d);
        const int64_t h_end = min(h_end_raw, input_h);
        const int64_t w_end = min(w_end_raw, input_w);
        if (d_start >= d_end || h_start >= h_end || w_start >= w_end) continue;

        const int64_t valid_size = (d_end - d_start) *
                                   (h_end - h_start) *
                                   (w_end - w_start);
        const int64_t divisor = divisor_override != 0
            ? divisor_override
            : (count_include_pad ? pool_size : valid_size);
        const acc_t value = static_cast<acc_t>(grad_output[index]) /
                            static_cast<acc_t>(divisor);
        scalar_t* input_plane = grad_input + nc * input_d * input_h * input_w;
        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                scalar_t* row = input_plane + (d * input_h + h) * input_w;
                for (int64_t w = w_start; w < w_end; ++w) {
                    gpuAtomicAddNoReturn(&row[w], static_cast<scalar_t>(value));
                }
            }
        }
    }
}

template <typename scalar_t>
void launch_avg_pool3d_forward(const Tensor& input, Tensor& output,
                               const Pool3dParams& params, bool ceil_mode,
                               bool count_include_pad,
                               std::optional<int64_t> divisor_override) {
    using acc_t = typename PoolAccum<scalar_t>::type;
    const int64_t input_d = input.size(-3);
    const int64_t input_h = input.size(-2);
    const int64_t input_w = input.size(-1);
    const int64_t output_d = pooling_output_shape(
        input_d, params.kernel[0], params.padding[0], params.stride[0], ceil_mode);
    const int64_t output_h = pooling_output_shape(
        input_h, params.kernel[1], params.padding[1], params.stride[1], ceil_mode);
    const int64_t output_w = pooling_output_shape(
        input_w, params.kernel[2], params.padding[2], params.stride[2], ceil_mode);
    const int64_t total = output.numel();
    if (total == 0) return;
    const int64_t blocks = std::min<int64_t>((total + 255) / 256, 65535);
    avg_pool3d_forward_kernel<scalar_t, acc_t><<<static_cast<unsigned>(blocks), 256,
                                                   0, getCurrentCUDAStream().stream()>>>(
        total, input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        input.dim() == 5 ? input.size(1) : 1,
        input_d, input_h, input_w, output_d, output_h, output_w,
        params.kernel[0], params.kernel[1], params.kernel[2],
        params.stride[0], params.stride[1], params.stride[2],
        params.padding[0], params.padding[1], params.padding[2],
        count_include_pad, divisor_override.value_or(0));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("avg_pool3d CUDA kernel: ") +
                 cudaGetErrorString(error));
    }
}

template <typename scalar_t>
void launch_avg_pool3d_backward(const Tensor& grad_output, Tensor& grad_input,
                                const Pool3dParams& params, bool ceil_mode,
                                bool count_include_pad,
                                std::optional<int64_t> divisor_override) {
    using acc_t = typename PoolAccum<scalar_t>::type;
    const int64_t input_d = grad_input.size(-3);
    const int64_t input_h = grad_input.size(-2);
    const int64_t input_w = grad_input.size(-1);
    const int64_t output_d = pooling_output_shape(
        input_d, params.kernel[0], params.padding[0], params.stride[0], ceil_mode);
    const int64_t output_h = pooling_output_shape(
        input_h, params.kernel[1], params.padding[1], params.stride[1], ceil_mode);
    const int64_t output_w = pooling_output_shape(
        input_w, params.kernel[2], params.padding[2], params.stride[2], ceil_mode);
    const int64_t total = grad_output.numel();
    if (total == 0) return;
    const int64_t blocks = std::min<int64_t>((total + 255) / 256, 65535);
    avg_pool3d_backward_kernel<scalar_t, acc_t><<<static_cast<unsigned>(blocks), 256,
                                                    0, getCurrentCUDAStream().stream()>>>(
        total, grad_output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
        input_d, input_h, input_w, output_d, output_h, output_w,
        params.kernel[0], params.kernel[1], params.kernel[2],
        params.stride[0], params.stride[1], params.stride[2],
        params.padding[0], params.padding[1], params.padding[2],
        count_include_pad, divisor_override.value_or(0));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("avg_pool3d_backward CUDA kernel: ") +
                 cudaGetErrorString(error));
    }
}

Tensor avg_pool3d_native_cuda(const Tensor& input,
                              const std::vector<int64_t>& kernel_size,
                              const std::vector<int64_t>& stride,
                              const std::vector<int64_t>& padding,
                              bool ceil_mode, bool count_include_pad,
                              std::optional<int64_t> divisor_override) {
    if (input.dim() == 4) {
        return avg_pool3d_native_cuda(input.unsqueeze(0), kernel_size, stride,
                                      padding, ceil_mode, count_include_pad,
                                      divisor_override).squeeze(0);
    }
    const Pool3dParams params = check_pool3d_args(
        input, kernel_size, stride, padding, ceil_mode, "avg_pool3d");
    if (divisor_override.has_value() && *divisor_override == 0) {
        TP_THROW(ValueError, "avg_pool3d: divisor must be not zero");
    }
    const int64_t output_d = pooling_output_shape(
        input.size(-3), params.kernel[0], params.padding[0], params.stride[0], ceil_mode);
    const int64_t output_h = pooling_output_shape(
        input.size(-2), params.kernel[1], params.padding[1], params.stride[1], ceil_mode);
    const int64_t output_w = pooling_output_shape(
        input.size(-1), params.kernel[2], params.padding[2], params.stride[2], ceil_mode);
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor output = Tensor::empty({input.size(0), input.size(1), output_d,
                                   output_h, output_w}, input.dtype(), input.device());
    switch (input.dtype()) {
        case DType::Float32:
            launch_avg_pool3d_forward<float>(input_c, output, params, ceil_mode,
                                              count_include_pad, divisor_override);
            break;
        case DType::Float64:
            launch_avg_pool3d_forward<double>(input_c, output, params, ceil_mode,
                                               count_include_pad, divisor_override);
            break;
        case DType::Float16:
            launch_avg_pool3d_forward<tensorplay::Half>(input_c, output, params, ceil_mode,
                                                         count_include_pad, divisor_override);
            break;
        case DType::BFloat16:
            launch_avg_pool3d_forward<tensorplay::BFloat16>(input_c, output, params, ceil_mode,
                                                            count_include_pad, divisor_override);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "avg_pool3d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return output;
}

Tensor avg_pool3d_backward_native_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override) {
    if (input.dim() == 4 && grad_output.dim() == 4) {
        return avg_pool3d_backward_native_cuda(
            grad_output.unsqueeze(0), input.unsqueeze(0), kernel_size, stride,
            padding, ceil_mode, count_include_pad, divisor_override).squeeze(0);
    }
    const Pool3dParams params = check_pool3d_args(
        input, kernel_size, stride, padding, ceil_mode, "avg_pool3d_backward");
    if (divisor_override.has_value() && *divisor_override == 0) {
        TP_THROW(ValueError, "avg_pool3d_backward: divisor must be not zero");
    }
    const int64_t output_d = pooling_output_shape(
        input.size(-3), params.kernel[0], params.padding[0], params.stride[0], ceil_mode);
    const int64_t output_h = pooling_output_shape(
        input.size(-2), params.kernel[1], params.padding[1], params.stride[1], ceil_mode);
    const int64_t output_w = pooling_output_shape(
        input.size(-1), params.kernel[2], params.padding[2], params.stride[2], ceil_mode);
    if (grad_output.dim() != 5 || grad_output.size(-3) != output_d ||
        grad_output.size(-2) != output_h || grad_output.size(-1) != output_w) {
        TP_THROW(ValueError, "avg_pool3d_backward: grad_output has invalid shape");
    }
    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(
        static_cast<std::vector<int64_t>>(input.shape()), input.dtype(), input.device());
    switch (input.dtype()) {
        case DType::Float32:
            launch_avg_pool3d_backward<float>(grad_output_c, grad_input, params, ceil_mode,
                                               count_include_pad, divisor_override);
            break;
        case DType::Float64:
            launch_avg_pool3d_backward<double>(grad_output_c, grad_input, params, ceil_mode,
                                                count_include_pad, divisor_override);
            break;
        case DType::Float16:
            launch_avg_pool3d_backward<tensorplay::Half>(grad_output_c, grad_input, params,
                                                          ceil_mode, count_include_pad,
                                                          divisor_override);
            break;
        case DType::BFloat16:
            launch_avg_pool3d_backward<tensorplay::BFloat16>(grad_output_c, grad_input, params,
                                                             ceil_mode, count_include_pad,
                                                             divisor_override);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "avg_pool3d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeAveragePool3d) {
    m.impl("avg_pool3d", avg_pool3d_native_cuda);
    m.impl("avg_pool3d_backward", avg_pool3d_backward_native_cuda);
}

} // namespace cuda
} // namespace tensorplay
