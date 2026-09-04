// Native CUDA adaptive average-pooling implementation. Forward assigns one
// thread to each output element. Backward assigns one thread to each input
// element and walks only the adaptive output interval that contains it.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

__device__ inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
    return (a / b) * c + ((a % b) * c) / b;
}

__device__ inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
    return 1 + ((a + 1) * c - 1) / b;
}

inline int64_t start_index_host(int64_t a, int64_t b, int64_t c) {
    return (a / b) * c + ((a % b) * c) / b;
}

inline int64_t end_index_host(int64_t a, int64_t b, int64_t c) {
    return 1 + ((a + 1) * c - 1) / b;
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

inline void check_adaptive_pool3d_args(const Tensor& input,
                                       const std::vector<int64_t>& output_size,
                                       const char* op) {
    if (input.dim() != 4 && input.dim() != 5) {
        TP_THROW(ValueError, std::string(op) +
                 ": expected a 4D or 5D tensor");
    }
    if (output_size.size() != 3) {
        TP_THROW(ValueError, std::string(op) +
                 ": output_size must have three values");
    }
    for (int64_t d = 0; d < 3; ++d) {
        if (output_size[d] <= 0) {
            TP_THROW(ValueError, std::string(op) +
                     ": output_size must be greater than zero");
        }
        if (input.size(input.dim() - 3 + d) <= 0) {
            TP_THROW(ValueError, std::string(op) +
                     ": non-batch input dimensions must be non-zero");
        }
    }
}

template <typename scalar_t, typename acc_t>
__global__ void adaptive_avg_pool3d_forward_kernel(
    int64_t total, const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output, int64_t channels,
    int64_t input_d, int64_t input_h, int64_t input_w,
    int64_t output_d, int64_t output_h, int64_t output_w) {
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += step) {
        const int64_t ow = index % output_w;
        const int64_t oh = (index / output_w) % output_h;
        const int64_t od = (index / (output_w * output_h)) % output_d;
        const int64_t nc = index / (output_w * output_h * output_d);
        const int64_t d_start = start_index(od, output_d, input_d);
        const int64_t d_end = end_index(od, output_d, input_d);
        const int64_t h_start = start_index(oh, output_h, input_h);
        const int64_t h_end = end_index(oh, output_h, input_h);
        const int64_t w_start = start_index(ow, output_w, input_w);
        const int64_t w_end = end_index(ow, output_w, input_w);
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
        const int64_t divisor = (d_end - d_start) * (h_end - h_start) *
                                (w_end - w_start);
        output[index] = static_cast<scalar_t>(sum / static_cast<acc_t>(divisor));
    }
}

template <typename scalar_t, typename acc_t>
__global__ void adaptive_avg_pool3d_backward_kernel(
    int64_t total, const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input, int64_t input_d,
    int64_t input_h, int64_t input_w, int64_t output_d,
    int64_t output_h, int64_t output_w) {
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += step) {
        const int64_t iw = index % input_w;
        const int64_t ih = (index / input_w) % input_h;
        const int64_t id = (index / (input_w * input_h)) % input_d;
        const int64_t nc = index / (input_w * input_h * input_d);
        const int64_t od_start = start_index(id, input_d, output_d);
        const int64_t od_end = end_index(id, input_d, output_d);
        const int64_t oh_start = start_index(ih, input_h, output_h);
        const int64_t oh_end = end_index(ih, input_h, output_h);
        const int64_t ow_start = start_index(iw, input_w, output_w);
        const int64_t ow_end = end_index(iw, input_w, output_w);
        const scalar_t* output_plane = grad_output + nc * output_d * output_h * output_w;
        acc_t value = static_cast<acc_t>(0);
        for (int64_t od = od_start; od < od_end; ++od) {
            const int64_t d_begin = start_index(od, output_d, input_d);
            const int64_t d_finish = end_index(od, output_d, input_d);
            for (int64_t oh = oh_start; oh < oh_end; ++oh) {
                const int64_t h_begin = start_index(oh, output_h, input_h);
                const int64_t h_finish = end_index(oh, output_h, input_h);
                for (int64_t ow = ow_start; ow < ow_end; ++ow) {
                    const int64_t w_begin = start_index(ow, output_w, input_w);
                    const int64_t w_finish = end_index(ow, output_w, input_w);
                    const int64_t divisor = (d_finish - d_begin) *
                                            (h_finish - h_begin) *
                                            (w_finish - w_begin);
                    const int64_t output_index = (od * output_h + oh) * output_w + ow;
                    value += static_cast<acc_t>(output_plane[output_index]) /
                             static_cast<acc_t>(divisor);
                }
            }
        }
        grad_input[index] = static_cast<scalar_t>(value);
    }
}

template <typename scalar_t>
void launch_adaptive_avg_pool3d_forward(const Tensor& input, Tensor& output,
                                        const std::vector<int64_t>& output_size) {
    using acc_t = typename PoolAccum<scalar_t>::type;
    const int64_t total = output.numel();
    if (total == 0) return;
    const int64_t blocks = std::min<int64_t>((total + 255) / 256, 65535);
    adaptive_avg_pool3d_forward_kernel<scalar_t, acc_t>
        <<<static_cast<unsigned>(blocks), 256, 0, getCurrentCUDAStream().stream()>>>(
            total, input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), input.size(1),
            input.size(-3), input.size(-2), input.size(-1), output_size[0],
            output_size[1], output_size[2]);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("adaptive_avg_pool3d CUDA kernel: ") +
                 cudaGetErrorString(error));
    }
}

template <typename scalar_t>
void launch_adaptive_avg_pool3d_backward(const Tensor& grad_output, Tensor& grad_input) {
    using acc_t = typename PoolAccum<scalar_t>::type;
    const int64_t total = grad_input.numel();
    if (total == 0) return;
    const int64_t blocks = std::min<int64_t>((total + 255) / 256, 65535);
    adaptive_avg_pool3d_backward_kernel<scalar_t, acc_t>
        <<<static_cast<unsigned>(blocks), 256, 0, getCurrentCUDAStream().stream()>>>(
            total, grad_output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
            grad_input.size(-3), grad_input.size(-2), grad_input.size(-1),
            grad_output.size(-3), grad_output.size(-2), grad_output.size(-1));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("adaptive_avg_pool3d_backward CUDA kernel: ") +
                 cudaGetErrorString(error));
    }
}

Tensor adaptive_avg_pool3d_native_cuda(
    const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 4) {
        return adaptive_avg_pool3d_native_cuda(input.unsqueeze(0), output_size).squeeze(0);
    }
    check_adaptive_pool3d_args(input, output_size, "adaptive_avg_pool3d");
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor output = Tensor::empty({input.size(0), input.size(1), output_size[0],
                                   output_size[1], output_size[2]}, input.dtype(), input.device());
    switch (input.dtype()) {
        case DType::Float32:
            launch_adaptive_avg_pool3d_forward<float>(input_c, output, output_size);
            break;
        case DType::Float64:
            launch_adaptive_avg_pool3d_forward<double>(input_c, output, output_size);
            break;
        case DType::Float16:
            launch_adaptive_avg_pool3d_forward<tensorplay::Half>(input_c, output, output_size);
            break;
        case DType::BFloat16:
            launch_adaptive_avg_pool3d_forward<tensorplay::BFloat16>(input_c, output, output_size);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_avg_pool3d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return output;
}

Tensor adaptive_avg_pool3d_backward_native_cuda(const Tensor& grad_output,
                                                 const Tensor& input) {
    if (input.dim() == 4 && grad_output.dim() == 4) {
        return adaptive_avg_pool3d_backward_native_cuda(
            grad_output.unsqueeze(0), input.unsqueeze(0)).squeeze(0);
    }
    check_adaptive_pool3d_args(input,
                               {grad_output.size(-3), grad_output.size(-2), grad_output.size(-1)},
                               "adaptive_avg_pool3d_backward");
    if (grad_output.dim() != 5 || grad_output.size(0) != input.size(0) ||
        grad_output.size(1) != input.size(1)) {
        TP_THROW(ValueError,
                 "adaptive_avg_pool3d_backward: grad_output has invalid shape");
    }
    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(
        static_cast<std::vector<int64_t>>(input.shape()), input.dtype(), input.device());
    switch (input.dtype()) {
        case DType::Float32:
            launch_adaptive_avg_pool3d_backward<float>(grad_output_c, grad_input);
            break;
        case DType::Float64:
            launch_adaptive_avg_pool3d_backward<double>(grad_output_c, grad_input);
            break;
        case DType::Float16:
            launch_adaptive_avg_pool3d_backward<tensorplay::Half>(grad_output_c, grad_input);
            break;
        case DType::BFloat16:
            launch_adaptive_avg_pool3d_backward<tensorplay::BFloat16>(grad_output_c, grad_input);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_avg_pool3d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

} // namespace

namespace {

Tensor& adaptive_avg_pool3d_out_cuda(const Tensor& self,
                                     const std::vector<int64_t>& output_size,
                                     Tensor& out) {
    out = adaptive_avg_pool3d_native_cuda(self, output_size);
    return out;
}

Tensor& adaptive_avg_pool3d_backward_grad_input_cuda(const Tensor& grad_output,
                                                     const Tensor& input,
                                                     Tensor& grad_input) {
    grad_input =
        adaptive_avg_pool3d_backward_native_cuda(grad_output, input);
    return grad_input;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeAdaptiveAveragePool3d) {
    m.impl("adaptive_avg_pool3d", adaptive_avg_pool3d_native_cuda);
    m.impl("adaptive_avg_pool3d_backward", adaptive_avg_pool3d_backward_native_cuda);
    m.impl("adaptive_avg_pool3d.out", adaptive_avg_pool3d_out_cuda);
    m.impl("adaptive_avg_pool3d_backward.grad_input",
           adaptive_avg_pool3d_backward_grad_input_cuda);
}

} // namespace cuda
} // namespace tensorplay
