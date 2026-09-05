#pragma once

#include "CUDARuntime.h"
#include "Exception.h"
#include "Tensor.h"
#include "TypePromotion.h"
#include "Utils.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tensorplay::cuda::special_detail {

inline constexpr int kThreads = 256;

inline void check_cuda(cudaError_t error) {
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error));
    }
}

inline std::vector<int64_t> shape_of(const Tensor& tensor) {
    return static_cast<std::vector<int64_t>>(tensor.shape());
}

template <typename F>
__global__ void unary_f64_kernel(int64_t n, const double* input, double* output, F function) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < n; index += stride) {
        output[index] = function(input[index]);
    }
}

template <typename F>
__global__ void unary_f32_kernel(int64_t n, const float* input, float* output, F function) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < n; index += stride) {
        output[index] = static_cast<float>(function(static_cast<double>(input[index])));
    }
}

template <typename F>
__global__ void binary_f64_kernel(
    int64_t n, const double* lhs, const double* rhs, double* output, F function) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < n; index += stride) {
        output[index] = function(lhs[index], rhs[index]);
    }
}

template <typename F>
__global__ void binary_f32_kernel(
    int64_t n, const float* lhs, const float* rhs, float* output, F function) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < n; index += stride) {
        output[index] = static_cast<float>(function(
            static_cast<double>(lhs[index]), static_cast<double>(rhs[index])));
    }
}

inline void launch_ew(dim3& grid, dim3& block, int64_t n) {
    block = dim3(kThreads);
    grid = dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads));
}

template <typename F>
Tensor float_math_cuda(const Tensor& self, F function, const char*) {
    DType input_dtype = self.dtype();
    DType output_dtype = isFloatingType(input_dtype) ? input_dtype : DType::Float32;
    DType compute_dtype = (input_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor input = (input_dtype == compute_dtype)
        ? self.contiguous()
        : self.to(compute_dtype).contiguous();
    Tensor output = Tensor::empty(shape_of(input), compute_dtype, input.device());
    int64_t n = input.numel();
    if (n > 0) {
        dim3 grid, block;
        launch_ew(grid, block, n);
        auto stream = getCurrentCUDAStream().stream();
        if (compute_dtype == DType::Float64) {
            unary_f64_kernel<<<grid, block, 0, stream>>>(
                n, input.data_ptr<double>(), output.data_ptr<double>(), function);
        } else {
            unary_f32_kernel<<<grid, block, 0, stream>>>(
                n, input.data_ptr<float>(), output.data_ptr<float>(), function);
        }
        check_cuda(cudaGetLastError());
    }
    return (output_dtype == compute_dtype) ? output : output.to(output_dtype);
}

template <typename F>
Tensor binary_float_cuda(const Tensor& lhs_in, const Tensor& rhs_in, F function,
                         const char* name) {
    if (lhs_in.device() != rhs_in.device()) {
        TP_THROW(DeviceMismatchError, name,
                 ": inputs must be on the same device");
    }
    DType dtype = promoteTypes(lhs_in.dtype(), rhs_in.dtype());
    if (!isFloatingType(dtype)) {
        dtype = DType::Float32;
    }
    DType compute_dtype = (dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor lhs = lhs_in.to(compute_dtype)
                     .expand(broadcast_shapes(shape_of(lhs_in), shape_of(rhs_in)))
                     .contiguous();
    Tensor rhs = rhs_in.to(compute_dtype).expand(shape_of(lhs)).contiguous();
    Tensor output = Tensor::empty(shape_of(lhs), compute_dtype, lhs_in.device());
    int64_t n = output.numel();
    if (n == 0) {
        return (dtype == compute_dtype) ? output : output.to(dtype);
    }
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
    if (compute_dtype == DType::Float64) {
        binary_f64_kernel<<<grid, block, 0, stream>>>(
            n, lhs.data_ptr<double>(), rhs.data_ptr<double>(), output.data_ptr<double>(), function);
    } else {
        binary_f32_kernel<<<grid, block, 0, stream>>>(
            n, lhs.data_ptr<float>(), rhs.data_ptr<float>(), output.data_ptr<float>(), function);
    }
    check_cuda(cudaGetLastError());
    return (dtype == compute_dtype) ? output : output.to(dtype);
}

template <typename T, typename F>
__global__ void typed_unary_kernel_t(int64_t n, const T* input, T* output, F function) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < n; index += stride) {
        output[index] = function(input[index]);
    }
}

template <typename T, typename F>
__global__ void typed_binary_kernel_t(
    int64_t n, const T* lhs, const T* rhs, T* output, F function) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < n; index += stride) {
        output[index] = function(lhs[index], rhs[index]);
    }
}

template <typename F>
Tensor typed_math_cuda(const Tensor& self, F function) {
    DType input_dtype = self.dtype();
    DType output_dtype = isFloatingType(input_dtype) ? input_dtype : DType::Float32;
    DType compute_dtype = (input_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor input = (input_dtype == compute_dtype)
        ? self.contiguous()
        : self.to(compute_dtype).contiguous();
    Tensor output = Tensor::empty(shape_of(input), compute_dtype, input.device());
    int64_t n = input.numel();
    if (n > 0) {
        dim3 grid, block;
        launch_ew(grid, block, n);
        auto stream = getCurrentCUDAStream().stream();
        if (compute_dtype == DType::Float64) {
            typed_unary_kernel_t<double><<<grid, block, 0, stream>>>(
                n, input.data_ptr<double>(), output.data_ptr<double>(), function);
        } else {
            typed_unary_kernel_t<float><<<grid, block, 0, stream>>>(
                n, input.data_ptr<float>(), output.data_ptr<float>(), function);
        }
        check_cuda(cudaGetLastError());
    }
    return (output_dtype == compute_dtype) ? output : output.to(output_dtype);
}

template <typename F>
Tensor typed_binary_cuda(const Tensor& lhs_in, const Tensor& rhs_in, F function) {
    if (lhs_in.device() != rhs_in.device()) {
        TP_THROW(DeviceMismatchError,
                 "special binary inputs must be on the same device");
    }
    DType dtype = promoteTypes(lhs_in.dtype(), rhs_in.dtype());
    if (!isFloatingType(dtype)) {
        dtype = DType::Float32;
    }
    DType compute_dtype = (dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor lhs = lhs_in.to(compute_dtype)
                     .expand(broadcast_shapes(shape_of(lhs_in), shape_of(rhs_in)))
                     .contiguous();
    Tensor rhs = rhs_in.to(compute_dtype).expand(shape_of(lhs)).contiguous();
    Tensor output = Tensor::empty(shape_of(lhs), compute_dtype, lhs_in.device());
    int64_t n = output.numel();
    if (n > 0) {
        dim3 grid, block;
        launch_ew(grid, block, n);
        auto stream = getCurrentCUDAStream().stream();
        if (compute_dtype == DType::Float64) {
            typed_binary_kernel_t<double><<<grid, block, 0, stream>>>(
                n, lhs.data_ptr<double>(), rhs.data_ptr<double>(), output.data_ptr<double>(), function);
        } else {
            typed_binary_kernel_t<float><<<grid, block, 0, stream>>>(
                n, lhs.data_ptr<float>(), rhs.data_ptr<float>(), output.data_ptr<float>(), function);
        }
        check_cuda(cudaGetLastError());
    }
    return (dtype == compute_dtype) ? output : output.to(dtype);
}

}  // namespace tensorplay::cuda::special_detail
