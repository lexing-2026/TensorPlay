#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "CUDAComplex.cuh"
#include "CUDNNUtils.h"
#include <cudnn.h>
#include <thrust/complex.h>
#include <type_traits>

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)


namespace tensorplay {
namespace cuda {

#ifdef USE_CUDNN

// Helper generic activation
Tensor cudnn_activation(const Tensor& self_in, cudnnActivationMode_t mode, double coef = 0.0) {
    // cuDNN activation rejects arbitrary strided layouts (e.g. chunk/split
    // views feeding gate math); materialize contiguous first.
    Tensor self = self_in.is_contiguous() ? self_in : self_in.contiguous();
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
    // cuDNN activation is broken on this stack (v9 + Pascal: EXECUTION_FAILED);
    // use the native kernel for every dtype (fp16/bf16/integral keep dtype).
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    dim3 block(256);
    dim3 grid((self.numel() + 255) / 256);
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    int64_t n = self.numel();
    auto stream = getCurrentCUDAStream().stream();
    switch (self.dtype()) {
        case DType::Float32:
            relu_kernel_n<float><<<grid, block, 0, stream>>>(n, self_contig.data_ptr<float>(), result.data_ptr<float>());
            break;
        case DType::Float64:
            relu_kernel_n<double><<<grid, block, 0, stream>>>(n, self_contig.data_ptr<double>(), result.data_ptr<double>());
            break;
        case DType::Float16:
            relu_kernel_n<tensorplay::Half><<<grid, block, 0, stream>>>(n, self_contig.data_ptr<tensorplay::Half>(), result.data_ptr<tensorplay::Half>());
            break;
        case DType::BFloat16:
            relu_kernel_n<tensorplay::BFloat16><<<grid, block, 0, stream>>>(n, self_contig.data_ptr<tensorplay::BFloat16>(), result.data_ptr<tensorplay::BFloat16>());
            break;
        case DType::Int32:
            relu_kernel_n<int32_t><<<grid, block, 0, stream>>>(n, self_contig.data_ptr<int32_t>(), result.data_ptr<int32_t>());
            break;
        case DType::Int64:
            relu_kernel_n<int64_t><<<grid, block, 0, stream>>>(n, self_contig.data_ptr<int64_t>(), result.data_ptr<int64_t>());
            break;
        default:
            TP_THROW(NotImplementedError, "relu: unsupported dtype");
    }
    checkCuda(cudaGetLastError(), "relu kernel launch");
    return result;
}

Tensor& cudnn_activation_inplace(Tensor& self_in, cudnnActivationMode_t mode, double coef = 0.0) {
    Tensor self = self_in.is_contiguous() ? self_in : self_in.contiguous();
    if (self.numel() == 0) return self_in;
    
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

Tensor& relu_inplace_kernel_cudnn(Tensor& self) {
    cudnn_activation_inplace(self, CUDNN_ACTIVATION_RELU);
    return self;
}

// Native elementwise sigmoid/tanh.  cuDNN activation is avoided here: it is
// slower than a flat kernel for elementwise work and CUDNN v9 + Pascal shows
// CUDNN_STATUS_EXECUTION_FAILED_CUDART on every shape (see remote P4).
template <typename T>
__global__ void sigmoid_kernel_n(int64_t n, const T* input, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T x = input[i];
        output[i] = T(1) / (T(1) + exp(-x));
    }
}

template <typename T>
__global__ void tanh_kernel_n(int64_t n, const T* input, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = tanh(input[i]);
    }
}

template <typename T>
__global__ void sigmoid_kernel_n_fp32(int64_t n, const T* input, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = static_cast<float>(input[i]);
        output[i] = static_cast<T>(1.0f / (1.0f + expf(-x)));
    }
}

template <typename T>
__global__ void tanh_kernel_n_fp32(int64_t n, const T* input, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = static_cast<T>(tanhf(static_cast<float>(input[i])));
    }
}

namespace {

struct ActCxSigmoid {
    template <typename T>
    __device__ thrust::complex<T> operator()(thrust::complex<T> z) const {
        return static_cast<T>(1) / (static_cast<T>(1) + thrust::exp(-z));
    }
};

struct ActCxTanh {
    template <typename T>
    __device__ thrust::complex<T> operator()(thrust::complex<T> z) const {
        return thrust::tanh(z);
    }
};

}  // namespace

static Tensor native_activation_dispatch(const Tensor& self, bool is_sigmoid) {
    if (isComplexType(self.dtype())) {
        if (self.dtype() != DType::ComplexFloat &&
            self.dtype() != DType::ComplexDouble) {
            TP_THROW(NotImplementedError,
                     "activation: half complexes are not supported yet");
        }
        Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                      self.dtype(), self.device());
        int64_t n = self.numel();
        if (n == 0) return result;
        const auto stream = getCurrentCUDAStream().stream();
        if (self.dtype() == DType::ComplexFloat) {
            if (is_sigmoid)
                cplx::launch_unary<float>(n, self_contig.data_ptr(), result.data_ptr(),
                                          ActCxSigmoid{}, stream);
            else
                cplx::launch_unary<float>(n, self_contig.data_ptr(), result.data_ptr(),
                                          ActCxTanh{}, stream);
        } else {
            if (is_sigmoid)
                cplx::launch_unary<double>(n, self_contig.data_ptr(), result.data_ptr(),
                                           ActCxSigmoid{}, stream);
            else
                cplx::launch_unary<double>(n, self_contig.data_ptr(), result.data_ptr(),
                                           ActCxTanh{}, stream);
        }
        checkCuda(cudaGetLastError(), "native activation complex kernel");
        return result;
    }
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                  self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    auto stream = getCurrentCUDAStream().stream();

    #define TP_NATIVE_ACT_CASE(ctype, name)                                  \
    case DType::name:                                                        \
        if (is_sigmoid)                                                      \
            sigmoid_kernel_n<ctype><<<grid, block, 0, stream>>>(             \
                n, self_contig.data_ptr<ctype>(), result.data_ptr<ctype>()); \
        else                                                                 \
            tanh_kernel_n<ctype><<<grid, block, 0, stream>>>(                \
                n, self_contig.data_ptr<ctype>(), result.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TP_NATIVE_ACT_CASE(float, Float32)
        TP_NATIVE_ACT_CASE(double, Float64)
        case DType::Float16:
            if (is_sigmoid)
                sigmoid_kernel_n_fp32<tensorplay::Half><<<grid, block, 0, stream>>>(
                    n, self_contig.data_ptr<tensorplay::Half>(),
                    result.data_ptr<tensorplay::Half>());
            else
                tanh_kernel_n_fp32<tensorplay::Half><<<grid, block, 0, stream>>>(
                    n, self_contig.data_ptr<tensorplay::Half>(),
                    result.data_ptr<tensorplay::Half>());
            break;
        case DType::BFloat16:
            if (is_sigmoid)
                sigmoid_kernel_n_fp32<tensorplay::BFloat16><<<grid, block, 0, stream>>>(
                    n, self_contig.data_ptr<tensorplay::BFloat16>(),
                    result.data_ptr<tensorplay::BFloat16>());
            else
                tanh_kernel_n_fp32<tensorplay::BFloat16><<<grid, block, 0, stream>>>(
                    n, self_contig.data_ptr<tensorplay::BFloat16>(),
                    result.data_ptr<tensorplay::BFloat16>());
            break;
        default:
            TP_THROW(NotImplementedError,
                     "activation: only float/double/fp16/bf16 supported");
    }
    #undef TP_NATIVE_ACT_CASE
    checkCuda(cudaGetLastError(), "native sigmoid/tanh kernel launch");
    return result;
}

Tensor sigmoid_kernel_cudnn(const Tensor& self) { return native_activation_dispatch(self, true); }
Tensor tanh_kernel_cudnn(const Tensor& self) { return native_activation_dispatch(self, false); }

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

// Fused gated activations stay in the activation translation unit, matching
// ATen/native/Activation.cpp and its CUDA ActivationSilu/ActivationGlu
// kernels.  The packed variant consumes [gate | up] on the last dimension.
namespace {

inline bool fused_activation_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float64 ||
           dtype == DType::Float16 || dtype == DType::BFloat16;
}

inline void check_silu_mul_inputs(const Tensor& gate, const Tensor& up,
                                  const char* op) {
    if (gate.device() != up.device()) {
        TP_THROW(DeviceMismatchError, op,
                 ": gate and up must be on the same device");
    }
    if (gate.shape() != up.shape()) {
        TP_THROW(RuntimeError, op, ": gate and up must have the same shape");
    }
    if (gate.dtype() != up.dtype()) {
        TP_THROW(RuntimeError, op, ": gate and up must have the same dtype");
    }
    if (!fused_activation_dtype(gate.dtype())) {
        TP_THROW(NotImplementedError, op,
                 ": only floating point dtypes are supported");
    }
}

template <typename T, typename Acc>
__global__ void fused_silu_mul_kernel(const T* gate, const T* up, T* output,
                                      int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i >= n) return;
    const Acc x = static_cast<Acc>(gate[i]);
    const Acc y = static_cast<Acc>(up[i]);
    const Acc sigmoid = Acc(1) / (Acc(1) + exp(-x));
    output[i] = static_cast<T>(x * sigmoid * y);
}

template <typename T, typename Acc>
__global__ void fused_silu_and_mul_kernel(const T* input, T* output,
                                          int64_t n, int64_t half_width) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i >= n) return;
    const int64_t row = i / half_width;
    const int64_t col = i - row * half_width;
    const int64_t base = row * (2 * half_width);
    const Acc gate = static_cast<Acc>(input[base + col]);
    const Acc up = static_cast<Acc>(input[base + half_width + col]);
    const Acc sigmoid = Acc(1) / (Acc(1) + exp(-gate));
    output[i] = static_cast<T>(gate * sigmoid * up);
}

template <typename T>
Tensor fused_silu_mul_typed(const Tensor& gate, const Tensor& up) {
    Tensor gate_c = gate.is_contiguous() ? gate : gate.contiguous();
    Tensor up_c = up.is_contiguous() ? up : up.contiguous();
    Tensor output = Tensor::empty(
        static_cast<std::vector<int64_t>>(gate_c.shape()), gate_c.dtype(),
        gate_c.device());
    if (gate_c.numel() == 0) return output;
    using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((gate_c.numel() + 255) / 256));
    fused_silu_mul_kernel<T, Acc><<<grid, block, 0,
                                   getCurrentCUDAStream().stream()>>>(
        gate_c.data_ptr<T>(), up_c.data_ptr<T>(), output.data_ptr<T>(),
        gate_c.numel());
    checkCuda(cudaGetLastError(), "silu_mul CUDA kernel launch");
    return output;
}

template <typename T>
Tensor fused_silu_and_mul_typed(const Tensor& input, int64_t half_width) {
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    std::vector<int64_t> output_shape =
        static_cast<std::vector<int64_t>>(input_c.shape());
    output_shape.back() = half_width;
    Tensor output = Tensor::empty(output_shape, input_c.dtype(),
                                  input_c.device());
    if (output.numel() == 0) return output;
    using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((output.numel() + 255) / 256));
    fused_silu_and_mul_kernel<T, Acc><<<grid, block, 0,
                                       getCurrentCUDAStream().stream()>>>(
        input_c.data_ptr<T>(), output.data_ptr<T>(), output.numel(),
        half_width);
    checkCuda(cudaGetLastError(), "silu_and_mul CUDA kernel launch");
    return output;
}

} // namespace

Tensor silu_mul_cuda(const Tensor& gate, const Tensor& up) {
    check_silu_mul_inputs(gate, up, "silu_mul");
    switch (gate.dtype()) {
        case DType::Float32:
            return fused_silu_mul_typed<float>(gate, up);
        case DType::Float64:
            return fused_silu_mul_typed<double>(gate, up);
        case DType::Float16:
            return fused_silu_mul_typed<Half>(gate, up);
        case DType::BFloat16:
            return fused_silu_mul_typed<BFloat16>(gate, up);
        default:
            TP_THROW(NotImplementedError, "silu_mul: unsupported dtype");
    }
}

Tensor fused_swiglu_cuda(const Tensor& gate, const Tensor& up) {
    return silu_mul_cuda(gate, up);
}

Tensor silu_and_mul_cuda(const Tensor& input) {
    if (input.dim() < 1) {
        TP_THROW(RuntimeError,
                 "silu_and_mul: input must have at least one dimension");
    }
    const int64_t width = input.size(-1);
    if ((width & 1) != 0) {
        TP_THROW(RuntimeError,
                 "silu_and_mul: the packed last dimension must be even");
    }
    if (!fused_activation_dtype(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "silu_and_mul: only floating point dtypes are supported");
    }
    const int64_t half_width = width / 2;
    switch (input.dtype()) {
        case DType::Float32:
            return fused_silu_and_mul_typed<float>(input, half_width);
        case DType::Float64:
            return fused_silu_and_mul_typed<double>(input, half_width);
        case DType::Float16:
            return fused_silu_and_mul_typed<Half>(input, half_width);
        case DType::BFloat16:
            return fused_silu_and_mul_typed<BFloat16>(input, half_width);
        default:
            TP_THROW(NotImplementedError, "silu_and_mul: unsupported dtype");
    }
}

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
    m.impl("silu_mul", silu_mul_cuda);
    m.impl("fused_swiglu", fused_swiglu_cuda);
    m.impl("silu_and_mul", silu_and_mul_cuda);
}

} // namespace cuda
} // namespace tensorplay
