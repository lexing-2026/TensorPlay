#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "CUDAComplex.cuh"
#include "CUDALoops.cuh"
#include "CUDNNUtils.h"
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <type_traits>
#include <limits>
#include "OutWrite.h"

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

Tensor silu_kernel_cuda_native(const Tensor& self) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(self)
        .build();
    switch (self.dtype()) {
        case DType::Float32:
            gpu_kernel(iter, [] __host__ __device__ (float value) -> float {
                return value / (1.0f + ::expf(-value));
            });
            break;
        case DType::Float64:
            gpu_kernel(iter, [] __host__ __device__ (double value) -> double {
                return value / (1.0 + ::exp(-value));
            });
            break;
        case DType::Float16:
            gpu_kernel(iter, [] __host__ __device__ (Half value) -> Half {
                const float value_acc = static_cast<float>(value);
                return static_cast<Half>(
                    value_acc / (1.0f + ::expf(-value_acc)));
            });
            break;
        case DType::BFloat16:
            gpu_kernel(iter, [] __host__ __device__ (BFloat16 value) -> BFloat16 {
                const float value_acc = static_cast<float>(value);
                return static_cast<BFloat16>(
                    value_acc / (1.0f + ::expf(-value_acc)));
            });
            break;
        default:
            TP_THROW(NotImplementedError,
                     "silu: only float/double/fp16/bf16 supported");
    }
    return result;
}

Tensor relu_kernel_cudnn(const Tensor& self) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(self)
        .build();
#define TP_RELU_CASE(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [] __host__ __device__ (ctype value) -> ctype { \
            return value > ctype(0) ? value : ctype(0); \
        }); \
        break;
    switch (self.dtype()) {
        TP_RELU_CASE(float, Float32)
        TP_RELU_CASE(double, Float64)
        TP_RELU_CASE(Half, Float16)
        TP_RELU_CASE(BFloat16, BFloat16)
        TP_RELU_CASE(int32_t, Int32)
        TP_RELU_CASE(int64_t, Int64)
        default:
            TP_THROW(NotImplementedError, "relu: unsupported dtype");
    }
#undef TP_RELU_CASE
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

namespace {

struct ActCxSigmoid {
    template <typename T>
    __device__ tensorplay::complex<T> operator()(
            tensorplay::complex<T> z) const {
        return static_cast<T>(1) / (static_cast<T>(1) + tensorplay::exp(-z));
    }
};

struct ActCxTanh {
    template <typename T>
    __device__ tensorplay::complex<T> operator()(
            tensorplay::complex<T> z) const {
        return tensorplay::tanh(z);
    }
};

}  // namespace

Tensor native_activation_dispatch(const Tensor& self, bool is_sigmoid) {
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
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                  self.dtype(), self.device());
    if (self.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(self)
        .build();
    switch (self.dtype()) {
        case DType::Float32:
            if (is_sigmoid) {
                gpu_kernel(iter, [] __host__ __device__ (float value) -> float {
                    return 1.0f / (1.0f + ::expf(-value));
                });
            } else {
                gpu_kernel(iter, [] __host__ __device__ (float value) -> float {
                    return ::tanhf(value);
                });
            }
            break;
        case DType::Float64:
            if (is_sigmoid) {
                gpu_kernel(iter, [] __host__ __device__ (double value) -> double {
                    return 1.0 / (1.0 + ::exp(-value));
                });
            } else {
                gpu_kernel(iter, [] __host__ __device__ (double value) -> double {
                    return ::tanh(value);
                });
            }
            break;
        case DType::Float16:
            if (is_sigmoid) {
                gpu_kernel(iter, [] __host__ __device__ (Half value) -> Half {
                    const float value_acc = static_cast<float>(value);
                    return static_cast<Half>(
                        1.0f / (1.0f + ::expf(-value_acc)));
                });
            } else {
                gpu_kernel(iter, [] __host__ __device__ (Half value) -> Half {
                    return static_cast<Half>(::tanhf(static_cast<float>(value)));
                });
            }
            break;
        case DType::BFloat16:
            if (is_sigmoid) {
                gpu_kernel(iter, [] __host__ __device__ (BFloat16 value) -> BFloat16 {
                    const float value_acc = static_cast<float>(value);
                    return static_cast<BFloat16>(
                        1.0f / (1.0f + ::expf(-value_acc)));
                });
            } else {
                gpu_kernel(iter, [] __host__ __device__ (BFloat16 value) -> BFloat16 {
                    return static_cast<BFloat16>(
                        ::tanhf(static_cast<float>(value)));
                });
            }
            break;
        default:
            TP_THROW(NotImplementedError,
                     "activation: only float/double/fp16/bf16 supported");
    }
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


// --- Backward Kernels ---

#endif  // USE_CUDNN

template <typename T>
inline void run_threshold_backward_iter(TensorIteratorBase& iter, T threshold) {
    gpu_kernel(iter, [threshold] __host__ __device__(T output_value, T grad_value) -> T {
        return output_value > threshold ? grad_value : T(0);
    });
}

Tensor threshold_backward_kernel(const Tensor& grad_output, const Tensor& output, const Scalar& threshold) {
    if (grad_output.numel() != output.numel()) {
        TP_THROW(RuntimeError, "threshold_backward: grad_output and output must have same size");
    }

    if (grad_output.dtype() != DType::Float32 &&
        grad_output.dtype() != DType::Float16 &&
        grad_output.dtype() != DType::BFloat16) {
        TP_THROW(NotImplementedError, "threshold_backward: only float32/fp16/bf16 supported");
    }

    Tensor grad_input = Tensor::empty_like(
        grad_output, DType::Undefined, grad_output.device());
    if (grad_input.numel() == 0) return grad_input;
    Tensor output_cast = output.dtype() == grad_output.dtype()
        ? output
        : output.to(grad_output.dtype());
    TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(true)
        .resize_outputs(false)
        .add_output(grad_input)
        .add_const_input(output_cast)
        .add_const_input(grad_output)
        .build();

    switch (grad_output.dtype()) {
        case DType::Float32:
            run_threshold_backward_iter<float>(iter, threshold.to<float>());
            break;
        case DType::Float16:
            run_threshold_backward_iter<Half>(iter, threshold.to<Half>());
            break;
        case DType::BFloat16:
            run_threshold_backward_iter<BFloat16>(iter, threshold.to<BFloat16>());
            break;
        default:
            TP_THROW(NotImplementedError,
                     "threshold_backward: only float32/fp16/bf16 supported");
    }

    return grad_input;
}

// Fused gated activations stay in the activation translation unit, matching
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
    const Acc sigmoid = Acc(1) / (Acc(1) + ::exp(-x));
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
    const Acc sigmoid = Acc(1) / (Acc(1) + ::exp(-gate));
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
#if defined(USE_CUDNN) && !defined(USE_ROCM)
    m.impl("relu", relu_kernel_cudnn);
    m.impl("relu_", relu_inplace_kernel_cudnn);
    m.impl("sigmoid", sigmoid_kernel_cudnn);
    m.impl("tanh", tanh_kernel_cudnn);
    m.impl("silu", silu_kernel_cudnn);
    // m.impl("elu", elu_kernel_cudnn); // Not registered in native_functions yet
#elif defined(USE_CUDNN) && defined(USE_ROCM)
    // The pointwise backend already registers relu/sigmoid/tanh/silu for
    // every dtype; its coverage is a superset of what the DNN library
    // offers here (fp64/bf16 activation included), and the elementwise
    // kernels are faster for this memory-bound surface.
    m.impl("relu_", relu_inplace_kernel_cudnn);
#else
    // The pointwise backend covers the activation surface without a DNN
    // library dependency.
#endif
    m.impl("threshold_backward", threshold_backward_kernel);
    m.impl("silu_mul", silu_mul_cuda);
    m.impl("fused_swiglu", fused_swiglu_cuda);
    m.impl("silu_and_mul", silu_and_mul_cuda);
}

} // namespace cuda
} // namespace tensorplay
