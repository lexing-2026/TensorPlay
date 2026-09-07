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

#endif  // USE_CUDNN

template <typename T>
inline void run_threshold_backward_iter(TensorIteratorBase& iter, T threshold) {
    gpu_kernel(iter, [threshold] __host__ __device__(T output_value, T grad_value) -> T {
        return output_value > threshold ? grad_value : T(0);
    });
}

Tensor threshold_backward_kernel(const Tensor& grad_output, const Tensor& output, Scalar threshold) {
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

// --- Native softmax ---
//
// Row softmax over an arbitrary dimension without a DNN-library dependency.
// The softmax dimension is the middle of the (outer, softmax_size, inner)
// view: one thread block walks one row whose consecutive elements sit
// `inner_size` elements apart.  Two passes (max, then sum of exponentials)
// keep the numerics of the library path; reduced-precision inputs compute
// in fp32, matching the upcast the DNN descriptor path applies.

template <typename scalar_t, typename compute_t>
__global__ void softmax_dim_kernel(
    scalar_t* out, const scalar_t* in, int64_t rows, int64_t softmax_size,
    int64_t inner_size, bool log_mode) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) return;
  const int64_t outer = row / inner_size;
  const int64_t inner = row % inner_size;
  const scalar_t* row_in =
      in + outer * softmax_size * inner_size + inner;
  scalar_t* row_out = out + outer * softmax_size * inner_size + inner;

  __shared__ compute_t tile[1024];
  const int tid = static_cast<int>(threadIdx.x);

  compute_t thread_max = -std::numeric_limits<compute_t>::infinity();
  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    const compute_t v = static_cast<compute_t>(row_in[j * inner_size]);
    thread_max = (v > thread_max) ? v : thread_max;
  }
  tile[tid] = thread_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) tile[tid] = (tile[tid + s] > tile[tid]) ? tile[tid + s] : tile[tid];
    __syncthreads();
  }
  const compute_t row_max = tile[0];

  compute_t thread_sum = compute_t(0);
  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    thread_sum += std::exp(static_cast<compute_t>(row_in[j * inner_size]) - row_max);
  }
  tile[tid] = thread_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) tile[tid] += tile[tid + s];
    __syncthreads();
  }
  const compute_t denom = std::log(tile[0]);

  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    const compute_t v =
        static_cast<compute_t>(row_in[j * inner_size]) - row_max - denom;
    row_out[j * inner_size] =
        static_cast<scalar_t>(log_mode ? v : std::exp(v));
  }
}

// --- Wave (warp) softmax for rows along the fast dimension ---
//
// One wave owns one row (or two rows for short rows): every lane keeps a
// strided register slice, and the max / sum reductions run as butterfly
// shuffles with no shared-memory round trip.  The logical wave width is
// clamped to the next power of two of the row length, so a hardware wave can
// hold two independent logical waves when rows are short.  Templates are
// instantiated for both 32- and 64-lane hardware waves and selected from the
// device attribute at launch time.

namespace {

inline int softmax_log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

inline int softmax_wave_size() {
  static int wave = []() {
    int dev = 0, lanes = 32;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&lanes, cudaDevAttrWarpSize, dev);
    return lanes > 0 ? lanes : 32;
  }();
  return wave;
}

template <typename acc_t, int WAVE_SIZE, int WAVE_BATCH, bool kIsMax>
__device__ __forceinline__ void wave_butterfly_reduce(acc_t* values) {
  const unsigned long long mask =
      WAVE_SIZE == 64 ? 0xffffffffffffffffull : 0xffffffffull;
#pragma unroll
  for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WAVE_BATCH; ++i) {
      const acc_t other =
          __shfl_xor_sync(mask, values[i], offset, WAVE_SIZE);
      values[i] = kIsMax ? (other > values[i] ? other : values[i])
                         : (values[i] + other);
    }
  }
}

template <typename scalar_t, typename acc_t, int LOG2_ELEMENTS, bool LOG_MODE,
          int HW_WAVE>
__global__ void softmax_wave_forward(scalar_t* dst, const scalar_t* src,
                                     int batch_count, int stride,
                                     int element_count) {
  constexpr int kNextPow2 = 1 << LOG2_ELEMENTS;
  constexpr int kWAVE = (kNextPow2 < HW_WAVE) ? kNextPow2 : HW_WAVE;
  constexpr int kIterations = kNextPow2 / kWAVE;
  constexpr int kBatchesPerWave = (kNextPow2 <= 128) ? 2 : 1;

  const int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchesPerWave;
  int local_batches = batch_count - first_batch;
  if (local_batches > kBatchesPerWave) local_batches = kBatchesPerWave;

  const int local_idx = static_cast<int>(threadIdx.x);
  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  acc_t elems[kBatchesPerWave][kIterations];
  for (int i = 0; i < kBatchesPerWave; ++i) {
    const int count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int element_index = local_idx + it * kWAVE;
      elems[i][it] =
          element_index < count
              ? static_cast<acc_t>(src[i * element_count + it * kWAVE])
              : -std::numeric_limits<acc_t>::infinity();
    }
  }

  acc_t max_value[kBatchesPerWave];
#pragma unroll
  for (int i = 0; i < kBatchesPerWave; ++i) {
    max_value[i] = elems[i][0];
#pragma unroll
    for (int it = 1; it < kIterations; ++it) {
      max_value[i] =
          max_value[i] > elems[i][it] ? max_value[i] : elems[i][it];
    }
  }
  wave_butterfly_reduce<acc_t, kWAVE, kBatchesPerWave, true>(max_value);

  acc_t sum[kBatchesPerWave];
#pragma unroll
  for (int i = 0; i < kBatchesPerWave; ++i) {
    sum[i] = acc_t(0);
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      if (LOG_MODE) {
        sum[i] += std::exp(elems[i][it] - max_value[i]);
      } else {
        elems[i][it] = std::exp(elems[i][it] - max_value[i]);
        sum[i] += elems[i][it];
      }
    }
  }
  wave_butterfly_reduce<acc_t, kWAVE, kBatchesPerWave, false>(sum);

#pragma unroll
  for (int i = 0; i < kBatchesPerWave; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int element_index = local_idx + it * kWAVE;
      if (element_index < element_count) {
        if (LOG_MODE) {
          // Elements stay raw: subtract the max and the log normalizer.
          const acc_t v =
              elems[i][it] - max_value[i] - std::log(sum[i]);
          dst[i * element_count + it * kWAVE] = static_cast<scalar_t>(v);
        } else {
          // Elements hold exp(x - max) from the accumulation pass.
          dst[i * element_count + it * kWAVE] =
              static_cast<scalar_t>(elems[i][it] / sum[i]);
        }
      }
    }
  }
}

template <typename scalar_t, typename acc_t, bool LOG_MODE>
void launch_wave_softmax(scalar_t* dst, const scalar_t* src, int64_t batch_count,
                         int64_t element_count, cudaStream_t stream) {
  const int log2_elements = softmax_log2_ceil(static_cast<int>(element_count));
  const int next_pow2 = 1 << log2_elements;
  const int hw_wave = softmax_wave_size();
  const int wave_size = next_pow2 < hw_wave ? next_pow2 : hw_wave;
  const int batches_per_wave = (next_pow2 <= 128) ? 2 : 1;
  constexpr int kThreadsPerBlock = 128;
  const int warps_per_block = kThreadsPerBlock / wave_size;
  const int batches_per_block = warps_per_block * batches_per_wave;
  const int64_t blocks =
      (batch_count + batches_per_block - 1) / batches_per_block;
  dim3 threads(static_cast<unsigned>(wave_size),
               static_cast<unsigned>(warps_per_block), 1);

#define TP_LAUNCH_WAVE_SOFTMAX(L2E)                                          \
  do {                                                                       \
    if (hw_wave == 64) {                                                     \
      softmax_wave_forward<scalar_t, acc_t, L2E, LOG_MODE, 64>               \
          <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(           \
              dst, src, static_cast<int>(batch_count),                       \
              static_cast<int>(element_count),                               \
              static_cast<int>(element_count));                              \
    } else {                                                                 \
      softmax_wave_forward<scalar_t, acc_t, L2E, LOG_MODE, 32>               \
          <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(           \
              dst, src, static_cast<int>(batch_count),                       \
              static_cast<int>(element_count),                               \
              static_cast<int>(element_count));                              \
    }                                                                        \
  } while (0)

  switch (log2_elements) {
    case 0: TP_LAUNCH_WAVE_SOFTMAX(0); break;
    case 1: TP_LAUNCH_WAVE_SOFTMAX(1); break;
    case 2: TP_LAUNCH_WAVE_SOFTMAX(2); break;
    case 3: TP_LAUNCH_WAVE_SOFTMAX(3); break;
    case 4: TP_LAUNCH_WAVE_SOFTMAX(4); break;
    case 5: TP_LAUNCH_WAVE_SOFTMAX(5); break;
    case 6: TP_LAUNCH_WAVE_SOFTMAX(6); break;
    case 7: TP_LAUNCH_WAVE_SOFTMAX(7); break;
    case 8: TP_LAUNCH_WAVE_SOFTMAX(8); break;
    case 9: TP_LAUNCH_WAVE_SOFTMAX(9); break;
    case 10: TP_LAUNCH_WAVE_SOFTMAX(10); break;
    default: TP_LAUNCH_WAVE_SOFTMAX(11); break;
  }
#undef TP_LAUNCH_WAVE_SOFTMAX
  CUDA_CHECK(cudaGetLastError());
}

// The wave kernel covers rows laid out along the fastest dimension with a
// bounded row length; anything else (strided rows, very long rows, huge
// batches) stays on the block kernel below.
template <typename scalar_t, typename acc_t, bool LOG_MODE>
bool try_wave_softmax(const Tensor& self, Tensor& result, int64_t softmax_size,
                      int64_t rows) {
  constexpr int64_t kMaxRowBytes = 8192;
  if (softmax_size <= 0 || softmax_size > 2048) return false;
  if (softmax_size * static_cast<int64_t>(sizeof(scalar_t)) > kMaxRowBytes)
    return false;
  if (rows * softmax_size > static_cast<int64_t>(INT32_MAX)) return false;
  if (!self.is_contiguous() || !result.is_contiguous()) return false;
  launch_wave_softmax<scalar_t, acc_t, LOG_MODE>(
      result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), rows,
      softmax_size, getCurrentCUDAStream().stream());
  return true;
}

}  // namespace

template <typename scalar_t, typename compute_t>
void softmax_dim_dispatch(const Tensor& self, Tensor& result, int64_t dim,
                          bool log_mode) {
  // One thread block per (outer, inner) row; the kernel splits blockIdx
  // back into the outer/inner pair.
  int64_t rows = 1;
  for (int64_t i = 0; i < dim; ++i) rows *= self.size(i);
  int64_t inner_size = 1;
  for (int64_t i = dim + 1; i < self.dim(); ++i) inner_size *= self.size(i);
  const int64_t softmax_size = self.size(dim);
  if (inner_size == 1) {
    // Fast dimension: wave-resident kernel, one hardware wave per row.
    const bool wave = log_mode
        ? try_wave_softmax<scalar_t, compute_t, true>(self, result,
                                                      softmax_size, rows)
        : try_wave_softmax<scalar_t, compute_t, false>(self, result,
                                                       softmax_size, rows);
    if (wave) return;
  }
  constexpr int kThreads = 256;
  softmax_dim_kernel<scalar_t, compute_t>
      <<<static_cast<unsigned>(rows * inner_size), kThreads, 0,
         getCurrentCUDAStream().stream()>>>(
          result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(),
          rows * inner_size, softmax_size, inner_size, log_mode);
  CUDA_CHECK(cudaGetLastError());
}


Tensor softmax_native_impl(const Tensor& self, int64_t dim, bool log_mode) {
  int64_t dim_idx = dim < 0 ? dim + self.dim() : dim;
  Tensor result = Tensor::empty(
      static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
      self.device());
  switch (self.dtype()) {
    case DType::Float32:
      softmax_dim_dispatch<float, float>(self, result, dim_idx, log_mode);
      break;
    case DType::Float64:
      softmax_dim_dispatch<double, double>(self, result, dim_idx, log_mode);
      break;
    case DType::Float16:
      softmax_dim_dispatch<Half, float>(self, result, dim_idx, log_mode);
      break;
    case DType::BFloat16:
      softmax_dim_dispatch<BFloat16, float>(self, result, dim_idx, log_mode);
      break;
    default:
      TP_THROW(NotImplementedError,
               "softmax: unsupported dtype on this GPU backend");
  }
  return result;
}

Tensor softmax_kernel_native(const Tensor& self, int64_t dim, DType dtype) {
  (void)dtype;
  return softmax_native_impl(self, dim, false);
}

Tensor log_softmax_kernel_native(const Tensor& self, int64_t dim, DType dtype) {
  (void)dtype;
  return softmax_native_impl(self, dim, true);
}

namespace {

// One block per (outer, inner) row, matching the forward layout.  The row
// reduction streams twice: once for the shared factor, once for the write.
//   softmax:  grad_in = out * (grad - <grad, out>_dim)
//   log:      grad_in = grad - exp(out) * <grad>_dim
template <typename scalar_t, typename compute_t, bool LOG_MODE>
__global__ void softmax_backward_data_kernel(
    scalar_t* grad_in, const scalar_t* grad, const scalar_t* out,
    int64_t rows, int64_t softmax_size, int64_t inner_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) return;
  const int64_t outer = row / inner_size;
  const int64_t inner = row % inner_size;
  const int64_t base = outer * softmax_size * inner_size + inner;
  const scalar_t* row_grad = grad + base;
  const scalar_t* row_out = out + base;
  scalar_t* row_in = grad_in + base;

  __shared__ compute_t tile[1024];
  const int tid = static_cast<int>(threadIdx.x);

  compute_t thread_sum = compute_t(0);
  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    thread_sum += static_cast<compute_t>(row_grad[j * inner_size]) *
                  (LOG_MODE ? compute_t(1)
                            : static_cast<compute_t>(row_out[j * inner_size]));
  }
  tile[tid] = thread_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) tile[tid] += tile[tid + s];
    __syncthreads();
  }
  const compute_t factor = tile[0];

  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    const compute_t g = static_cast<compute_t>(row_grad[j * inner_size]);
    const compute_t o = static_cast<compute_t>(row_out[j * inner_size]);
    compute_t v;
    if constexpr (LOG_MODE) {
      v = g - std::exp(o) * factor;
    } else {
      v = o * (g - factor);
    }
    row_in[j * inner_size] = static_cast<scalar_t>(v);
  }
}

// grad_output drives the result dtype; reduced-width inputs accumulate and
// compute in float.  grad_output may carry float32 for a half input (the
// half_to_float forward path), in which case the result casts back to the
// input dtype.
Tensor softmax_backward_native_impl(const Tensor& grad_output,
                                    const Tensor& output, int64_t dim,
                                    DType input_dtype, bool log_mode) {
  Tensor g = grad_output.dim() == 0 ? grad_output.view({1}) : grad_output;
  Tensor o = output.dim() == 0 ? output.view({1}) : output;
  const int64_t nd = g.dim();
  const int64_t d = dim < 0 ? dim + nd : dim;
  if (d < 0 || d >= nd) {
    TP_THROW(IndexError,
             "dim must be non-negative and less than input dimensions");
  }
  DType result_dtype = g.dtype();
  if (result_dtype != input_dtype && result_dtype == DType::Float32 &&
      input_dtype == DType::Float16) {
    result_dtype = DType::Float16;
  }
  if (!isFloatingType(result_dtype)) {
    TP_THROW(TypeError, "unsupported dtype for softmax backward");
  }

  Tensor result =
      Tensor::empty(static_cast<std::vector<int64_t>>(g.shape()), result_dtype,
                    g.device());
  if (g.numel() == 0) return result;

  Tensor gc = g.contiguous();
  Tensor oc = o.contiguous().to(gc.dtype());
  Tensor result_work = result_dtype == gc.dtype()
                           ? result
                           : Tensor::empty(
                                 static_cast<std::vector<int64_t>>(g.shape()),
                                 gc.dtype(), g.device());

  int64_t outer = 1;
  for (int64_t i = 0; i < d; ++i) outer *= gc.size(i);
  int64_t inner = 1;
  for (int64_t i = d + 1; i < gc.dim(); ++i) inner *= gc.size(i);
  const int64_t dim_size = gc.size(d);
  const int64_t rows = outer * inner;
  const int threads = dim_size < 256 ? 32 : 256;

  #define TP_SOFTMAX_BWD_LAUNCH(ctype, acc)                                \
  if (log_mode) {                                                          \
    constexpr bool LOG_MODE = true;                                        \
    softmax_backward_data_kernel<ctype, acc, LOG_MODE>                     \
        <<<static_cast<unsigned>(rows), threads, 0,                        \
           getCurrentCUDAStream().stream()>>>(                             \
            result_work.data_ptr<ctype>(), gc.data_ptr<ctype>(),           \
            oc.data_ptr<ctype>(), rows, dim_size, inner);                  \
  } else {                                                                 \
    constexpr bool LOG_MODE = false;                                       \
    softmax_backward_data_kernel<ctype, acc, LOG_MODE>                     \
        <<<static_cast<unsigned>(rows), threads, 0,                        \
           getCurrentCUDAStream().stream()>>>(                             \
            result_work.data_ptr<ctype>(), gc.data_ptr<ctype>(),           \
            oc.data_ptr<ctype>(), rows, dim_size, inner);                  \
  }

  switch (gc.dtype()) {
    case DType::Float32:
      TP_SOFTMAX_BWD_LAUNCH(float, float)
      break;
    case DType::Float64:
      TP_SOFTMAX_BWD_LAUNCH(double, double)
      break;
    case DType::Float16:
      TP_SOFTMAX_BWD_LAUNCH(Half, float)
      break;
    case DType::BFloat16:
      TP_SOFTMAX_BWD_LAUNCH(BFloat16, float)
      break;
    default:
      TP_THROW(TypeError, "unsupported dtype for softmax backward");
  }
  #undef TP_SOFTMAX_BWD_LAUNCH
  CUDA_CHECK(cudaGetLastError());

  if (result_work.data_ptr() != result.data_ptr()) {
    result.copy_(result_work);
  }
  return result;
}

}  // namespace

Tensor _softmax_backward_data_cuda(const Tensor& grad_output,
                                   const Tensor& output, int64_t dim,
                                   DType input_dtype) {
  return softmax_backward_native_impl(grad_output, output, dim, input_dtype,
                                      /*log_mode=*/false);
}

Tensor& _softmax_backward_data_out_cuda(const Tensor& grad_output,
                                        const Tensor& output, int64_t dim,
                                        DType input_dtype, Tensor& grad_input) {
  grad_input = softmax_backward_native_impl(grad_output, output, dim,
                                            input_dtype, /*log_mode=*/false);
  return grad_input;
}

Tensor _log_softmax_backward_data_cuda(const Tensor& grad_output,
                                       const Tensor& output, int64_t dim,
                                       DType input_dtype) {
  return softmax_backward_native_impl(grad_output, output, dim, input_dtype,
                                      /*log_mode=*/true);
}

Tensor& _log_softmax_backward_data_out_cuda(const Tensor& grad_output,
                                            const Tensor& output, int64_t dim,
                                            DType input_dtype,
                                            Tensor& grad_input) {
  grad_input = softmax_backward_native_impl(grad_output, output, dim,
                                            input_dtype, /*log_mode=*/true);
  return grad_input;
}

Tensor& _softmax_out_cuda(const Tensor& self, int64_t dim, bool half_to_float,
                          Tensor& out) {
  if (half_to_float) {
    TP_THROW(RuntimeError,
             "softmax with half to float conversion is not supported on this backend");
  }
  out = softmax_native_impl(self, dim, false);
  return out;
}

Tensor& _log_softmax_out_cuda(const Tensor& self, int64_t dim,
                              bool half_to_float, Tensor& out) {
  if (half_to_float) {
    TP_THROW(RuntimeError,
             "log_softmax with half to float conversion is not supported on this backend");
  }
  out = softmax_native_impl(self, dim, true);
  return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ActivationKernels) {
    m.impl("_softmax.out", _softmax_out_cuda);
    m.impl("_log_softmax.out", _log_softmax_out_cuda);
    m.impl("_softmax_backward_data", _softmax_backward_data_cuda);
    m.impl("_softmax_backward_data.out", _softmax_backward_data_out_cuda);
    m.impl("_log_softmax_backward_data", _log_softmax_backward_data_cuda);
    m.impl("_log_softmax_backward_data.out", _log_softmax_backward_data_out_cuda);
#if defined(USE_CUDNN) && !defined(USE_ROCM)
    m.impl("relu", relu_kernel_cudnn);
    m.impl("relu_", relu_inplace_kernel_cudnn);
    m.impl("sigmoid", sigmoid_kernel_cudnn);
    m.impl("tanh", tanh_kernel_cudnn);
    m.impl("silu", silu_kernel_cudnn);
    // m.impl("elu", elu_kernel_cudnn); // Not registered in native_functions yet
    m.impl("softmax", softmax_kernel_cudnn);
    m.impl("log_softmax", log_softmax_kernel_cudnn);
#elif defined(USE_CUDNN) && defined(USE_ROCM)
    // The pointwise backend already registers relu/sigmoid/tanh/silu for
    // every dtype; its coverage is a superset of what the DNN library offers
    // here (fp64/bf16 activation and bf16 softmax return not-implemented),
    // and the elementwise kernels are faster for this memory-bound surface.
    m.impl("relu_", relu_inplace_kernel_cudnn);
    m.impl("softmax", softmax_kernel_native);
    m.impl("log_softmax", log_softmax_kernel_native);
#else
    // Native kernels cover the activation/softmax surface without a DNN
    // library dependency.
    m.impl("softmax", softmax_kernel_native);
    m.impl("log_softmax", log_softmax_kernel_native);
#endif
    m.impl("threshold_backward", threshold_backward_kernel);
    m.impl("silu_mul", silu_mul_cuda);
    m.impl("fused_swiglu", fused_swiglu_cuda);
    m.impl("silu_and_mul", silu_and_mul_cuda);
}

} // namespace cuda
} // namespace tensorplay
