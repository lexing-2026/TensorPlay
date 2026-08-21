#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "CUDNNUtils.h"
#include <cudnn.h>

namespace tensorplay {
namespace cuda {

#ifdef USE_CUDNN

// Helper generic activation
Tensor cudnn_activation(const Tensor& self, cudnnActivationMode_t mode, double coef = 0.0) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    cudnnTensorDescriptor_t xDesc = createTensorDescriptor(self);
    cudnnTensorDescriptor_t yDesc = createTensorDescriptor(result);
    
    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, mode, CUDNN_PROPAGATE_NAN, coef));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    double alpha_d = 1.0;
    double beta_d = 0.0;
    
    void* alpha_ptr = (self.dtype() == DType::Float64) ? (void*)&alpha_d : (void*)&alpha;
    void* beta_ptr = (self.dtype() == DType::Float64) ? (void*)&beta_d : (void*)&beta;
    
    CUDNN_CHECK(cudnnActivationForward(handle, actDesc, 
        alpha_ptr, xDesc, self.data_ptr(), 
        beta_ptr, yDesc, result.data_ptr()));
        
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    
    return result;
}

// Native implementation for Silu if cuDNN Swish fails or is unavailable
template <typename T>
__global__ void silu_kernel_n(int64_t n, const T* input, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T x = input[i];
        output[i] = x / (1.0 + exp(-x));
    }
}

// Half/bfloat16 do not have a native device exp implementation in the
// lightweight scalar wrappers used by TensorPlay.  Match PyTorch's usual
// mixed-precision behavior by evaluating the sigmoid in FP32 and rounding only
// the final result back to the input dtype.
template <typename T>
__global__ void silu_kernel_n_fp32(int64_t n, const T* input, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = static_cast<float>(input[i]);
        float y = x / (1.0f + expf(-x));
        output[i] = static_cast<T>(y);
    }
}

template <typename T>
__global__ void relu_kernel_n(int64_t n, const T* input, T* output);

Tensor silu_kernel_cuda_native(const Tensor& self) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    // Keep FP32/FP64 native, and use FP32 intermediates for reduced precision.
    if (self.dtype() == DType::Float32) {
        silu_kernel_n<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<float>(), result.data_ptr<float>());
    } else if (self.dtype() == DType::Float64) {
        silu_kernel_n<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<double>(), result.data_ptr<double>());
    } else if (self.dtype() == DType::Float16) {
        Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
        silu_kernel_n_fp32<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n, self_contig.data_ptr<tensorplay::Half>(), result.data_ptr<tensorplay::Half>());
    } else if (self.dtype() == DType::BFloat16) {
        Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
        silu_kernel_n_fp32<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n, self_contig.data_ptr<tensorplay::BFloat16>(), result.data_ptr<tensorplay::BFloat16>());
    } else {
        TP_THROW(NotImplementedError, "silu: only float/double/fp16/bf16 supported");
    }
    
    // CUDA_CHECK is defined in this file or Macros? 
    // It is defined in this file.
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error));
    }
    return result;
}

Tensor relu_kernel_cudnn(const Tensor& self) {
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) {
        return cudnn_activation(self, CUDNN_ACTIVATION_RELU);
    }
    // cuDNN path only accepts float/double; fall back to a native kernel
    // (fp16/bf16 and integral inputs keep their dtype, like PyTorch).
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    dim3 block(256);
    dim3 grid((self.numel() + 255) / 256);
    Tensor self_contig = self.contiguous();
    int64_t n = self.numel();
    if (self.dtype() == DType::Float16) {
        relu_kernel_n<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_contig.data_ptr<tensorplay::Half>(), result.data_ptr<tensorplay::Half>());
    } else if (self.dtype() == DType::BFloat16) {
        relu_kernel_n<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_contig.data_ptr<tensorplay::BFloat16>(), result.data_ptr<tensorplay::BFloat16>());
    } else if (self.dtype() == DType::Int32) {
        relu_kernel_n<int32_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_contig.data_ptr<int32_t>(), result.data_ptr<int32_t>());
    } else if (self.dtype() == DType::Int64) {
        relu_kernel_n<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_contig.data_ptr<int64_t>(), result.data_ptr<int64_t>());
    } else {
        TP_THROW(NotImplementedError, "relu: unsupported dtype");
    }
    checkCuda(cudaGetLastError(), "relu kernel launch");
    return result;
}

Tensor& cudnn_activation_inplace(Tensor& self, cudnnActivationMode_t mode, double coef = 0.0) {
    if (self.numel() == 0) return self;
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    cudnnTensorDescriptor_t xDesc = createTensorDescriptor(self);
    
    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, mode, CUDNN_PROPAGATE_NAN, coef));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    double alpha_d = 1.0;
    double beta_d = 0.0;
    
    void* alpha_ptr = (self.dtype() == DType::Float64) ? (void*)&alpha_d : (void*)&alpha;
    void* beta_ptr = (self.dtype() == DType::Float64) ? (void*)&beta_d : (void*)&beta;
    
    // In-place: yDesc = xDesc, y = x
    CUDNN_CHECK(cudnnActivationForward(handle, actDesc, 
        alpha_ptr, xDesc, self.data_ptr(), 
        beta_ptr, xDesc, self.data_ptr()));
        
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    
    return self;
}

Tensor& relu_inplace_kernel_cudnn(Tensor& self) { return cudnn_activation_inplace(self, CUDNN_ACTIVATION_RELU); }

Tensor sigmoid_kernel_cudnn(const Tensor& self) { return cudnn_activation(self, CUDNN_ACTIVATION_SIGMOID); }
Tensor tanh_kernel_cudnn(const Tensor& self) { return cudnn_activation(self, CUDNN_ACTIVATION_TANH); }

// Swish is Silu (beta=1.0)
// Check if defined
#ifndef CUDNN_ACTIVATION_SWISH
#define CUDNN_ACTIVATION_SWISH (cudnnActivationMode_t)5 // Usually 5 in newer cuDNN
#endif

Tensor silu_kernel_cudnn(const Tensor& self) { 
    // return cudnn_activation(self, CUDNN_ACTIVATION_SWISH, 1.0); 
    // Fallback to native implementation due to CUDNN_STATUS_BAD_PARAM issues with Swish in some versions
    return silu_kernel_cuda_native(self);
}

// Elu
Tensor elu_kernel_cudnn(const Tensor& self, Scalar alpha) { 
    return cudnn_activation(self, CUDNN_ACTIVATION_ELU, alpha.to<double>()); 
}

Tensor cudnn_softmax(const Tensor& self, int64_t dim, bool log) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    
    // Map to NCHW where C is the softmax dim
    // N = outer_size, C = softmax_size, H = inner_size, W = 1
    int64_t outer_size = 1;
    for(int i=0; i<dim; ++i) outer_size *= self.size(i);
    int64_t softmax_size = self.size(dim);
    int64_t inner_size = 1;
    for(int i=dim+1; i<ndim; ++i) inner_size *= self.size(i);
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    
    cudnnDataType_t c_dtype = (self.dtype() == DType::Float64) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    // Set 4D descriptor with logical dims
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, c_dtype, (int)outer_size, (int)softmax_size, (int)inner_size, 1));
    
    cudnnSoftmaxAlgorithm_t algo = log ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL; // Softmax over C
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (self.dtype() == DType::Float64) { alpha_p = &alpha_d; beta_p = &beta_d; }
    
    CUDNN_CHECK(cudnnSoftmaxForward(handle, algo, mode, alpha_p, desc, self.data_ptr(), beta_p, desc, result.data_ptr()));
    
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    
    return result;
}

Tensor softmax_kernel_cudnn(const Tensor& self, int64_t dim, DType dtype) {
    // Ignoring dtype arg for now (assuming input dtype)
    return cudnn_softmax(self, dim, false);
}

Tensor log_softmax_kernel_cudnn(const Tensor& self, int64_t dim, DType dtype) {
    return cudnn_softmax(self, dim, true);
}

// --- Backward Kernels ---

template <typename T>
__global__ void relu_kernel_n(int64_t n, const T* input, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T x = input[i];
        output[i] = x > T(0) ? x : T(0);
    }
}

template <typename T>
__global__ void threshold_backward_kernel_impl(int64_t n, const T* grad_output, const T* output, T threshold, T* grad_input) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_input[i] = (output[i] > threshold) ? grad_output[i] : static_cast<T>(0);
    }
}

Tensor threshold_backward_kernel(const Tensor& grad_output, const Tensor& output, Scalar threshold) {
    if (grad_output.numel() != output.numel()) {
        TP_THROW(RuntimeError, "threshold_backward: grad_output and output must have same size");
    }

    // Reduction backward can pass an expanded scalar tangent.  Its logical
    // shape matches ``output``, but its storage has a zero stride and is not
    // safe to consume as a flat CUDA pointer.  Materialize the same contiguous
    // tangent contract used by PyTorch's autograd kernels before launching.
    Tensor grad_contig = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor output_contig = output.is_contiguous() ? output : output.contiguous();
    
    Tensor grad_input = Tensor::empty_like(grad_contig, DType::Undefined, grad_contig.device());
    int64_t n = grad_contig.numel();
    if (n == 0) return grad_input;
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    if (grad_output.dtype() == DType::Float32) {
        Tensor output_cast = (output_contig.dtype() == DType::Float32) ? output_contig : output_contig.to(DType::Float32);
        threshold_backward_kernel_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n, 
            grad_contig.data_ptr<float>(), 
            output_cast.data_ptr<float>(), 
            threshold.to<float>(), 
            grad_input.data_ptr<float>());
    } else if (grad_output.dtype() == DType::Float16) {
        Tensor output_cast = (output_contig.dtype() == DType::Float16) ? output_contig : output_contig.to(DType::Float16);
        threshold_backward_kernel_impl<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n,
            grad_contig.data_ptr<tensorplay::Half>(),
            output_cast.data_ptr<tensorplay::Half>(),
            tensorplay::Half(threshold.to<float>()),
            grad_input.data_ptr<tensorplay::Half>());
    } else if (grad_output.dtype() == DType::BFloat16) {
        Tensor output_cast = (output_contig.dtype() == DType::BFloat16) ? output_contig : output_contig.to(DType::BFloat16);
        threshold_backward_kernel_impl<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n,
            grad_contig.data_ptr<tensorplay::BFloat16>(),
            output_cast.data_ptr<tensorplay::BFloat16>(),
            tensorplay::BFloat16(threshold.to<float>()),
            grad_input.data_ptr<tensorplay::BFloat16>());
    } else {
        TP_THROW(NotImplementedError, "threshold_backward: only float32/fp16/bf16 supported");
    }
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error));
    }
    
    return grad_input;
}

#endif

TENSORPLAY_LIBRARY_IMPL(CUDA, ActivationKernels) {
#ifdef USE_CUDNN
    m.impl("relu", relu_kernel_cudnn);
    m.impl("relu_", relu_inplace_kernel_cudnn);
    m.impl("sigmoid", sigmoid_kernel_cudnn);
    m.impl("tanh", tanh_kernel_cudnn);
    m.impl("silu", silu_kernel_cudnn);
    // m.impl("elu", elu_kernel_cudnn); // Not registered in native_functions yet
    m.impl("softmax", softmax_kernel_cudnn);
    m.impl("log_softmax", log_softmax_kernel_cudnn);
    m.impl("threshold_backward", threshold_backward_kernel);
#endif
}

} // namespace cuda
} // namespace tensorplay
