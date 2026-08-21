#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "Allocator.h"
#include "CUDNNUtils.h"
#include "Scalar.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDABroadcast.cuh"

#include <cuda_runtime.h>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace tensorplay {
namespace cuda {

// ATen alignment: scalars are converted to the opmath type of T so reduced
// floating types (Half/BFloat16) receive float32 scalars, matching aten
// native/cuda binary kernels.
template <typename T> struct BinaryOpMath { using type = T; };
template <> struct BinaryOpMath<tensorplay::Half> { using type = float; };
template <> struct BinaryOpMath<tensorplay::BFloat16> { using type = float; };

template <typename T>
inline typename BinaryOpMath<T>::type scalar_to_opmath(const Scalar& s) {
    return s.to<typename BinaryOpMath<T>::type>();
}

// Grid-stride wrapper so any launch config is correct (ATen elementwise style)
#define TP_CUDA_GRIDSTRIDE(i) \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    int64_t tp_stride = static_cast<int64_t>(blockDim.x) * gridDim.x; \
    for (; i < n; i += tp_stride)

// Forward declarations for the scalar fallback used by the fused kernel.
Tensor add_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha);
Tensor mul_scalar_kernel(const Tensor& self, Scalar other);
Tensor& relu_inplace_kernel_cudnn(Tensor& self);

// --- Utils ---
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

#define MAX_DIMS 8

static bool is_channels_last_4d(
    const Tensor& tensor,
    const std::vector<int64_t>& logical_shape) {
    if (tensor.dim() != 4 || logical_shape.size() != 4) return false;
    for (size_t dim = 0; dim < 4; ++dim) {
        if (tensor.size(static_cast<int64_t>(dim)) != logical_shape[dim]) {
            return false;
        }
    }
    const int64_t c = logical_shape[1];
    const int64_t h = logical_shape[2];
    const int64_t w = logical_shape[3];
    return tensor.stride(0) == c * h * w &&
           tensor.stride(1) == 1 &&
           tensor.stride(2) == w * c &&
           tensor.stride(3) == c;
}

static Tensor empty_channels_last_4d(
    const std::vector<int64_t>& logical_shape,
    DType dtype,
    const Device& device) {
    Tensor result = Tensor::empty(logical_shape, dtype, device);
    if (logical_shape.size() != 4) return result;
    const int64_t c = logical_shape[1];
    const int64_t h = logical_shape[2];
    const int64_t w = logical_shape[3];
    return result.as_strided(
        logical_shape, {c * h * w, 1, w * c, c});
}

// --- Kernels ---

// DIV Kernel with broadcasting
template <typename T>
__global__ void div_broadcast_kernel(int64_t n, 
                                     const T* a, TensorDesc a_desc,
                                     const T* b, TensorDesc b_desc,
                                     T* y, TensorDesc y_desc) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        int64_t a_off = get_offset(i, a_desc, y_desc);
        int64_t b_off = get_offset(i, b_desc, y_desc);
        y[i] = static_cast<T>(static_cast<M>(a[a_off]) / static_cast<M>(b[b_off]));
    }
}

template <typename T>
__global__ void add_broadcast_kernel(int64_t n, 
                                     const T* a, TensorDesc a_desc,
                                     const T* b, TensorDesc b_desc,
                                     T* y, TensorDesc y_desc, typename BinaryOpMath<T>::type alpha) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        int64_t a_off = get_offset(i, a_desc, y_desc);
        int64_t b_off = get_offset(i, b_desc, y_desc);
        y[i] = static_cast<T>(static_cast<M>(a[a_off]) + alpha * static_cast<M>(b[b_off]));
    }
}

template <typename T>
__global__ void sub_broadcast_kernel(int64_t n, 
                                     const T* a, TensorDesc a_desc,
                                     const T* b, TensorDesc b_desc,
                                     T* y, TensorDesc y_desc, typename BinaryOpMath<T>::type alpha) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        int64_t a_off = get_offset(i, a_desc, y_desc);
        int64_t b_off = get_offset(i, b_desc, y_desc);
        y[i] = static_cast<T>(static_cast<M>(a[a_off]) - alpha * static_cast<M>(b[b_off]));
    }
}

template <typename T, bool Divide>
__global__ void addc_broadcast_kernel(
    int64_t n,
    const T* a, TensorDesc a_desc,
    const T* b, TensorDesc b_desc,
    const T* c, TensorDesc c_desc,
    T* y, TensorDesc y_desc, typename BinaryOpMath<T>::type alpha) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        const int64_t a_off = get_offset(i, a_desc, y_desc);
        const int64_t b_off = get_offset(i, b_desc, y_desc);
        const int64_t c_off = get_offset(i, c_desc, y_desc);
        if constexpr (Divide) {
            y[i] = static_cast<T>(static_cast<M>(a[a_off]) + alpha * (static_cast<M>(b[b_off]) / static_cast<M>(c[c_off])));
        } else {
            y[i] = static_cast<T>(static_cast<M>(a[a_off]) + alpha * static_cast<M>(b[b_off]) * static_cast<M>(c[c_off]));
        }
    }
}

template <typename T>
__global__ void mul_broadcast_kernel(int64_t n, 
                                     const T* a, TensorDesc a_desc,
                                     const T* b, TensorDesc b_desc,
                                     T* y, TensorDesc y_desc) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        int64_t a_off = get_offset(i, a_desc, y_desc);
        int64_t b_off = get_offset(i, b_desc, y_desc);
        y[i] = static_cast<T>(static_cast<M>(a[a_off]) * static_cast<M>(b[b_off]));
    }
}

template <typename T>
__global__ void fused_mul_add_kernel_cuda_impl(int64_t n,
                                               const T* a,
                                               const T* b,
                                               const T* c,
                                               T* y) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        y[i] = static_cast<T>(static_cast<M>(a[i]) * static_cast<M>(b[i]) + static_cast<M>(c[i]));
    }
}

__global__ void fused_mul_add_scalar_kernel_cuda_impl(int64_t n,
                                                       const float* a,
                                                       float b,
                                                       float c,
                                                       float* y) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b + c;
}

// Tensor-Scalar Kernels
template <typename T>
__global__ void add_scalar_kernel_cuda_impl(int64_t n, const T* a, typename BinaryOpMath<T>::type b, T* y, typename BinaryOpMath<T>::type alpha) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        y[i] = static_cast<T>(static_cast<M>(a[i]) + alpha * static_cast<M>(b));
    }
}

template <typename T>
__global__ void sub_scalar_kernel_cuda_impl(int64_t n, const T* a, typename BinaryOpMath<T>::type b, T* y, typename BinaryOpMath<T>::type alpha) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        y[i] = static_cast<T>(static_cast<M>(a[i]) - alpha * static_cast<M>(b));
    }
}

template <typename T>
__global__ void mul_scalar_kernel_cuda_impl(int64_t n, const T* a, typename BinaryOpMath<T>::type b, T* y) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        y[i] = static_cast<T>(static_cast<M>(a[i]) * static_cast<M>(b));
    }
}

template <typename T>
__global__ void div_scalar_kernel_cuda_impl(int64_t n, const T* a, typename BinaryOpMath<T>::type b, T* y) {
    using M = typename BinaryOpMath<T>::type;
    TP_CUDA_GRIDSTRIDE(i) {
        y[i] = static_cast<T>(static_cast<M>(a[i]) / static_cast<M>(b));
    }
}

// --- Dispatchers ---

void get_grid_block(int64_t n, dim3& grid, dim3& block) {
    block.x = 256;
    grid.x = (n + 255) / 256;
}

#ifdef USE_CUDNN
// Helper for cuDNN binary op
void cudnn_binary_op(const Tensor& a, const Tensor& b, Tensor& c, cudnnOpTensorOp_t op, double alpha1, double alpha2, double beta) {
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    cudnnTensorDescriptor_t aDesc = createTensorDescriptor(a);
    cudnnTensorDescriptor_t bDesc = createTensorDescriptor(b);
    cudnnTensorDescriptor_t cDesc = createTensorDescriptor(c);
    
    cudnnOpTensorDescriptor_t opDesc;
    CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&opDesc));
    
    cudnnDataType_t compType = (a.dtype() == DType::Float64) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    // CUDNN_PROPAGATE_NAN is standard.
    CUDNN_CHECK(cudnnSetOpTensorDescriptor(opDesc, op, compType, CUDNN_PROPAGATE_NAN));
    
    float a1_f = (float)alpha1;
    float a2_f = (float)alpha2;
    float b_f = (float)beta;
    double a1_d = alpha1;
    double a2_d = alpha2;
    double b_d = beta;
    
    void* alpha1_p = (compType == CUDNN_DATA_DOUBLE) ? (void*)&a1_d : (void*)&a1_f;
    void* alpha2_p = (compType == CUDNN_DATA_DOUBLE) ? (void*)&a2_d : (void*)&a2_f;
    void* beta_p = (compType == CUDNN_DATA_DOUBLE) ? (void*)&b_d : (void*)&b_f;
    
    cudnnStatus_t status = cudnnOpTensor(handle, opDesc, 
        alpha1_p, aDesc, a.data_ptr(),
        alpha2_p, bDesc, b.data_ptr(),
        beta_p, cDesc, c.data_ptr());

    if (status != CUDNN_STATUS_SUCCESS) {
         // Cleanup before throw
         cudnnDestroyOpTensorDescriptor(opDesc);
         cudnnDestroyTensorDescriptor(aDesc);
         cudnnDestroyTensorDescriptor(bDesc);
         cudnnDestroyTensorDescriptor(cDesc);
         TP_THROW(RuntimeError, std::string("cuDNN Error in cudnnOpTensor: ") + cudnnGetErrorString(status));
    }
        
    CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(opDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(aDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cDesc));
}
#endif

// ADD
Tensor add_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    int64_t n = result.numel();
    if (n == 0) return result;

    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor a = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    Tensor b = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

    TensorDesc a_desc = make_desc(a, out_shape.size());
    TensorDesc b_desc = make_desc(b, out_shape.size());
    TensorDesc y_desc = make_desc(result, out_shape.size());
    
    switch (result_dtype) {
        case DType::Float32:
            add_broadcast_kernel<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, result.data_ptr<float>(), y_desc, alpha.to<float>());
            break;
        case DType::Int32:
            add_broadcast_kernel<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, result.data_ptr<int>(), y_desc, alpha.to<int>());
            break;
        case DType::Int64:
            add_broadcast_kernel<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, result.data_ptr<int64_t>(), y_desc, alpha.to<int64_t>());
            break;
        case DType::Float16:
            add_broadcast_kernel<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::Half>(), a_desc, b.data_ptr<tensorplay::Half>(), b_desc, result.data_ptr<tensorplay::Half>(), y_desc, alpha.to<float>());
            break;
        case DType::BFloat16:
            add_broadcast_kernel<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::BFloat16>(), a_desc, b.data_ptr<tensorplay::BFloat16>(), b_desc, result.data_ptr<tensorplay::BFloat16>(), y_desc, alpha.to<float>());
            break;
        case DType::Float64:
            add_broadcast_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, result.data_ptr<double>(), y_desc, alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA add: unsupported dtype");
    }
    return result;
}

__global__ void add_relu_same_shape_kernel(
    int64_t n, const float* self, const float* other, float* result) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float value = self[i] + other[i];
        result[i] = value < 0.0f ? 0.0f : value;
    }
}

// TorchInductor's pointwise epilogue fusion writes relu(add(...)) directly
// to the final output, including when the add operands broadcast.
__global__ void add_relu_broadcast_kernel(
    int64_t n,
    const float* self,
    TensorDesc self_desc,
    const float* other,
    TensorDesc other_desc,
    float* result,
    TensorDesc result_desc) {
    TP_CUDA_GRIDSTRIDE(i) {
        const int64_t self_offset = get_offset(i, self_desc, result_desc);
        const int64_t other_offset = get_offset(i, other_desc, result_desc);
        const float value = self[self_offset] + other[other_offset];
        const int64_t result_offset = get_output_offset(i, result_desc);
        result[result_offset] = value < 0.0f ? 0.0f : value;
    }
}

Tensor add_relu_cuda(const Tensor& self, const Tensor& other) {
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.shape() == other.shape() && self.is_contiguous() &&
        other.is_contiguous()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32, self.device());
        const int64_t n = result.numel();
        if (n == 0) return result;
        dim3 grid, block;
        get_grid_block(n, grid, block);
        add_relu_same_shape_kernel<<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n, self.data_ptr<float>(), other.data_ptr<float>(), result.data_ptr<float>());
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32) {
        const std::vector<int64_t> output_shape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(self.shape()),
            static_cast<std::vector<int64_t>>(other.shape()));
        const bool output_channels_last =
            is_channels_last_4d(self, output_shape) ||
            is_channels_last_4d(other, output_shape);
        Tensor result = output_channels_last
            ? empty_channels_last_4d(output_shape, DType::Float32, self.device())
            : Tensor::empty(output_shape, DType::Float32, self.device());
        const int64_t n = result.numel();
        if (n == 0) return result;
        dim3 grid, block;
        get_grid_block(n, grid, block);
        const TensorDesc self_desc = make_desc(self, output_shape.size());
        const TensorDesc other_desc = make_desc(other, output_shape.size());
        const TensorDesc result_desc = make_desc(result, output_shape.size());
        add_relu_broadcast_kernel<<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n,
            self.data_ptr<float>(),
            self_desc,
            other.data_ptr<float>(),
            other_desc,
            result.data_ptr<float>(),
            result_desc);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    Tensor result = add_kernel(self, other, Scalar(1));
    return relu_inplace_kernel_cudnn(result);
}

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
    int64_t n = self.numel();
    if (n == 0) return self;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    // For inplace, we cast other to self.dtype()
    Tensor b = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());

    TensorDesc a_desc = make_desc(self, self.dim());
    TensorDesc b_desc = make_desc(b, self.dim());
    TensorDesc y_desc = make_desc(self, self.dim());
    
    switch (self.dtype()) {
        case DType::Float32:
            add_broadcast_kernel<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, self.data_ptr<float>(), y_desc, alpha.to<float>());
            break;
        case DType::Int32:
            add_broadcast_kernel<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, self.data_ptr<int>(), y_desc, alpha.to<int>());
            break;
        case DType::Int64:
            add_broadcast_kernel<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, self.data_ptr<int64_t>(), y_desc, alpha.to<int64_t>());
            break;
        case DType::Float16:
            add_broadcast_kernel<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::Half>(), a_desc, b.data_ptr<tensorplay::Half>(), b_desc, self.data_ptr<tensorplay::Half>(), y_desc, alpha.to<float>());
            break;
        case DType::BFloat16:
            add_broadcast_kernel<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::BFloat16>(), a_desc, b.data_ptr<tensorplay::BFloat16>(), b_desc, self.data_ptr<tensorplay::BFloat16>(), y_desc, alpha.to<float>());
            break;
        case DType::Float64:
            add_broadcast_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, self.data_ptr<double>(), y_desc, alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA add_: unsupported dtype");
    }
    return self;
}

// SUB
Tensor sub_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    int64_t n = result.numel();
    if (n == 0) return result;

    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor a = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    Tensor b = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

    TensorDesc a_desc = make_desc(a, out_shape.size());
    TensorDesc b_desc = make_desc(b, out_shape.size());
    TensorDesc y_desc = make_desc(result, out_shape.size());
    
    switch (result_dtype) {
        case DType::Float32:
            sub_broadcast_kernel<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, result.data_ptr<float>(), y_desc, alpha.to<float>());
            break;
        case DType::Int32:
            sub_broadcast_kernel<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, result.data_ptr<int>(), y_desc, alpha.to<int>());
            break;
        case DType::Int64:
            sub_broadcast_kernel<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, result.data_ptr<int64_t>(), y_desc, alpha.to<int64_t>());
            break;
        case DType::Float16:
            sub_broadcast_kernel<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::Half>(), a_desc, b.data_ptr<tensorplay::Half>(), b_desc, result.data_ptr<tensorplay::Half>(), y_desc, alpha.to<float>());
            break;
        case DType::BFloat16:
            sub_broadcast_kernel<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::BFloat16>(), a_desc, b.data_ptr<tensorplay::BFloat16>(), b_desc, result.data_ptr<tensorplay::BFloat16>(), y_desc, alpha.to<float>());
            break;
        case DType::Float64:
            sub_broadcast_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, result.data_ptr<double>(), y_desc, alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA sub: unsupported dtype");
    }
    return result;
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
    int64_t n = self.numel();
    if (n == 0) return self;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor b = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());

    TensorDesc a_desc = make_desc(self, self.dim());
    TensorDesc b_desc = make_desc(b, self.dim());
    TensorDesc y_desc = make_desc(self, self.dim());
    
    switch (self.dtype()) {
        case DType::Float32:
            sub_broadcast_kernel<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, self.data_ptr<float>(), y_desc, alpha.to<float>());
            break;
        case DType::Int32:
            sub_broadcast_kernel<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, self.data_ptr<int>(), y_desc, alpha.to<int>());
            break;
        case DType::Int64:
            sub_broadcast_kernel<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, self.data_ptr<int64_t>(), y_desc, alpha.to<int64_t>());
            break;
        case DType::Float16:
            sub_broadcast_kernel<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::Half>(), a_desc, b.data_ptr<tensorplay::Half>(), b_desc, self.data_ptr<tensorplay::Half>(), y_desc, alpha.to<float>());
            break;
        case DType::BFloat16:
            sub_broadcast_kernel<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::BFloat16>(), a_desc, b.data_ptr<tensorplay::BFloat16>(), b_desc, self.data_ptr<tensorplay::BFloat16>(), y_desc, alpha.to<float>());
            break;
        case DType::Float64:
            sub_broadcast_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, self.data_ptr<double>(), y_desc, alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA sub_: unsupported dtype");
    }
    return self;
}

// MUL
Tensor mul_kernel(const Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    int64_t n = result.numel();
    if (n == 0) return result;

    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor a = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    Tensor b = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

    TensorDesc a_desc = make_desc(a, out_shape.size());
    TensorDesc b_desc = make_desc(b, out_shape.size());
    TensorDesc y_desc = make_desc(result, out_shape.size());
    
    switch (result_dtype) {
        case DType::Float32:
            mul_broadcast_kernel<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, result.data_ptr<float>(), y_desc);
            break;
        case DType::Int32:
            mul_broadcast_kernel<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, result.data_ptr<int>(), y_desc);
            break;
        case DType::Int64:
            mul_broadcast_kernel<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, result.data_ptr<int64_t>(), y_desc);
            break;
        case DType::Float16:
            mul_broadcast_kernel<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::Half>(), a_desc, b.data_ptr<tensorplay::Half>(), b_desc, result.data_ptr<tensorplay::Half>(), y_desc);
            break;
        case DType::BFloat16:
            mul_broadcast_kernel<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::BFloat16>(), a_desc, b.data_ptr<tensorplay::BFloat16>(), b_desc, result.data_ptr<tensorplay::BFloat16>(), y_desc);
            break;
        case DType::Float64:
            mul_broadcast_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, result.data_ptr<double>(), y_desc);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA mul: unsupported dtype");
    }
    return result;
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
    int64_t n = self.numel();
    if (n == 0) return self;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor b = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());

    TensorDesc a_desc = make_desc(self, self.dim());
    TensorDesc b_desc = make_desc(b, self.dim());
    TensorDesc y_desc = make_desc(self, self.dim());
    
    switch (self.dtype()) {
        case DType::Float32:
            mul_broadcast_kernel<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, self.data_ptr<float>(), y_desc);
            break;
        case DType::Int32:
            mul_broadcast_kernel<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, self.data_ptr<int>(), y_desc);
            break;
        case DType::Int64:
            mul_broadcast_kernel<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, self.data_ptr<int64_t>(), y_desc);
            break;
        case DType::Float16:
            mul_broadcast_kernel<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::Half>(), a_desc, b.data_ptr<tensorplay::Half>(), b_desc, self.data_ptr<tensorplay::Half>(), y_desc);
            break;
        case DType::BFloat16:
            mul_broadcast_kernel<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::BFloat16>(), a_desc, b.data_ptr<tensorplay::BFloat16>(), b_desc, self.data_ptr<tensorplay::BFloat16>(), y_desc);
            break;
        case DType::Float64:
            mul_broadcast_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, self.data_ptr<double>(), y_desc);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA mul_: unsupported dtype");
    }
    return self;
}

Tensor fused_mul_add_kernel(const Tensor& self, const Tensor& other, const Tensor& addend) {
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        addend.dtype() == DType::Float32 && self.is_contiguous() &&
        other.is_contiguous() && addend.is_contiguous() &&
        self.shape() == other.shape() && self.shape() == addend.shape()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32, self.device());
        int64_t n = self.numel();
        if (n == 0) return result;
        dim3 grid, block;
        get_grid_block(n, grid, block);
        fused_mul_add_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n, self.data_ptr<float>(), other.data_ptr<float>(), addend.data_ptr<float>(),
            result.data_ptr<float>());
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    return add_kernel(mul_kernel(self, other), addend, Scalar(1));
}

Tensor fused_mul_add_scalar_kernel(const Tensor& self, Scalar other, Scalar addend) {
    if (self.dtype() == DType::Float32 && self.is_contiguous()) {
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), DType::Float32, self.device());
        const int64_t n = self.numel();
        if (n == 0) return result;
        dim3 grid, block;
        get_grid_block(n, grid, block);
        fused_mul_add_scalar_kernel_cuda_impl<<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n, self.data_ptr<float>(), static_cast<float>(other.toDouble()),
            static_cast<float>(addend.toDouble()), result.data_ptr<float>());
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    return add_scalar_kernel(mul_scalar_kernel(self, other), addend, Scalar(1));
}

// DIV
Tensor div_kernel(const Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    if (isIntegralType(result_dtype)) result_dtype = DType::Float32; // Div promotes to float
    
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    int64_t n = result.numel();
    if (n == 0) return result;

    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor a = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    Tensor b = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

    TensorDesc a_desc = make_desc(a, out_shape.size());
    TensorDesc b_desc = make_desc(b, out_shape.size());
    TensorDesc y_desc = make_desc(result, out_shape.size());
    
    switch (result_dtype) {
        case DType::Float32:
            div_broadcast_kernel<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, result.data_ptr<float>(), y_desc);
            break;
        case DType::Float16:
            div_broadcast_kernel<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::Half>(), a_desc, b.data_ptr<tensorplay::Half>(), b_desc, result.data_ptr<tensorplay::Half>(), y_desc);
            break;
        case DType::BFloat16:
            div_broadcast_kernel<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::BFloat16>(), a_desc, b.data_ptr<tensorplay::BFloat16>(), b_desc, result.data_ptr<tensorplay::BFloat16>(), y_desc);
            break;
        case DType::Float64:
            div_broadcast_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, result.data_ptr<double>(), y_desc);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA div: unsupported dtype");
    }
    return result;
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
    int64_t n = self.numel();
    if (n == 0) return self;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    // Inplace div might change dtype if self is int (e.g. 5/2 = 2 or 2.5?)
    // In PyTorch, in-place div on int tensor performs floor division or cast?
    // "RuntimeError: result type Float can't be cast to the desired output type Long" usually.
    // For now, let's assume we do standard div and cast back.
    
    Tensor b = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());

    TensorDesc a_desc = make_desc(self, self.dim());
    TensorDesc b_desc = make_desc(b, self.dim());
    TensorDesc y_desc = make_desc(self, self.dim());
    
    switch (self.dtype()) {
        case DType::Float32:
            div_broadcast_kernel<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, self.data_ptr<float>(), y_desc);
            break;
        case DType::Int32:
            div_broadcast_kernel<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, self.data_ptr<int>(), y_desc);
            break;
        case DType::Int64:
            div_broadcast_kernel<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, self.data_ptr<int64_t>(), y_desc);
            break;
        case DType::Float16:
            div_broadcast_kernel<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::Half>(), a_desc, b.data_ptr<tensorplay::Half>(), b_desc, self.data_ptr<tensorplay::Half>(), y_desc);
            break;
        case DType::BFloat16:
            div_broadcast_kernel<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::BFloat16>(), a_desc, b.data_ptr<tensorplay::BFloat16>(), b_desc, self.data_ptr<tensorplay::BFloat16>(), y_desc);
            break;
        case DType::Float64:
            div_broadcast_kernel<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, self.data_ptr<double>(), y_desc);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA div_: unsupported dtype");
    }
    return self;
}


Tensor add_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
    DType result_dtype = self.dtype();
    // PyTorch's scalar promotion is value/type aware: a Python float does not
    // widen an already floating tensor (including fp16/bf16).  Only integral
    // tensors need promotion when they meet a floating scalar.
    if (!isFloatingType(result_dtype) &&
        (other.isFloatingPoint() || alpha.isFloatingPoint())) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor a = (self.dtype() == result_dtype) ? self_contig : self_contig.to(result_dtype);
    
    switch (result_dtype) {
        case DType::Float32:
            add_scalar_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<float>(), other.to<float>(), result.data_ptr<float>(), alpha.to<float>());
            break;
        case DType::Int32:
            add_scalar_kernel_cuda_impl<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int>(), other.to<int>(), result.data_ptr<int>(), alpha.to<int>());
            break;
        case DType::Int64:
            add_scalar_kernel_cuda_impl<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int64_t>(), other.to<int64_t>(), result.data_ptr<int64_t>(), alpha.to<int64_t>());
            break;
        case DType::Float16:
            add_scalar_kernel_cuda_impl<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::Half>(), other.to<float>(), result.data_ptr<tensorplay::Half>(), alpha.to<float>());
            break;
        case DType::BFloat16:
            add_scalar_kernel_cuda_impl<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::BFloat16>(), other.to<float>(), result.data_ptr<tensorplay::BFloat16>(), alpha.to<float>());
            break;
        case DType::Float64:
            add_scalar_kernel_cuda_impl<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<double>(), other.to<double>(), result.data_ptr<double>(), alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA add_scalar: unsupported dtype");
    }
    return result;
}

Tensor& add_scalar_inplace_kernel(Tensor& self, Scalar other, Scalar alpha) {
    int64_t n = self.numel();
    if (n == 0) return self;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    if (!self.is_contiguous()) {
         TP_THROW(NotImplementedError, "CUDA add_scalar_: non-contiguous input not supported yet (requires strided kernel)");
    }
    
    switch (self.dtype()) {
        case DType::Float32:
            add_scalar_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<float>(), other.to<float>(), self.data_ptr<float>(), alpha.to<float>());
            break;
        case DType::Int32:
            add_scalar_kernel_cuda_impl<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int>(), other.to<int>(), self.data_ptr<int>(), alpha.to<int>());
            break;
        case DType::Int64:
            add_scalar_kernel_cuda_impl<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int64_t>(), other.to<int64_t>(), self.data_ptr<int64_t>(), alpha.to<int64_t>());
            break;
        case DType::Float16:
            add_scalar_kernel_cuda_impl<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::Half>(), other.to<float>(), self.data_ptr<tensorplay::Half>(), alpha.to<float>());
            break;
        case DType::BFloat16:
            add_scalar_kernel_cuda_impl<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::BFloat16>(), other.to<float>(), self.data_ptr<tensorplay::BFloat16>(), alpha.to<float>());
            break;
        case DType::Float64:
            add_scalar_kernel_cuda_impl<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<double>(), other.to<double>(), self.data_ptr<double>(), alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA add_scalar_: unsupported dtype");
    }
    return self;
}

Tensor sub_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
    DType result_dtype = self.dtype();
    if (!isFloatingType(result_dtype) &&
        (other.isFloatingPoint() || alpha.isFloatingPoint())) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor a = (self.dtype() == result_dtype) ? self_contig : self_contig.to(result_dtype);
    
    switch (result_dtype) {
        case DType::Float32:
            sub_scalar_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<float>(), other.to<float>(), result.data_ptr<float>(), alpha.to<float>());
            break;
        case DType::Int32:
            sub_scalar_kernel_cuda_impl<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int>(), other.to<int>(), result.data_ptr<int>(), alpha.to<int>());
            break;
        case DType::Int64:
            sub_scalar_kernel_cuda_impl<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int64_t>(), other.to<int64_t>(), result.data_ptr<int64_t>(), alpha.to<int64_t>());
            break;
        case DType::Float16:
            sub_scalar_kernel_cuda_impl<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::Half>(), other.to<float>(), result.data_ptr<tensorplay::Half>(), alpha.to<float>());
            break;
        case DType::BFloat16:
            sub_scalar_kernel_cuda_impl<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::BFloat16>(), other.to<float>(), result.data_ptr<tensorplay::BFloat16>(), alpha.to<float>());
            break;
        case DType::Float64:
            sub_scalar_kernel_cuda_impl<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<double>(), other.to<double>(), result.data_ptr<double>(), alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA sub_scalar: unsupported dtype");
    }
    return result;
}

Tensor& sub_scalar_inplace_kernel(Tensor& self, Scalar other, Scalar alpha) {
    int64_t n = self.numel();
    if (n == 0) return self;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    if (!self.is_contiguous()) {
         TP_THROW(NotImplementedError, "CUDA sub_scalar_: non-contiguous input not supported yet");
    }
    
    switch (self.dtype()) {
        case DType::Float32:
            sub_scalar_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<float>(), other.to<float>(), self.data_ptr<float>(), alpha.to<float>());
            break;
        case DType::Int32:
            sub_scalar_kernel_cuda_impl<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int>(), other.to<int>(), self.data_ptr<int>(), alpha.to<int>());
            break;
        case DType::Int64:
            sub_scalar_kernel_cuda_impl<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int64_t>(), other.to<int64_t>(), self.data_ptr<int64_t>(), alpha.to<int64_t>());
            break;
        case DType::Float16:
            sub_scalar_kernel_cuda_impl<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::Half>(), other.to<float>(), self.data_ptr<tensorplay::Half>(), alpha.to<float>());
            break;
        case DType::BFloat16:
            sub_scalar_kernel_cuda_impl<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::BFloat16>(), other.to<float>(), self.data_ptr<tensorplay::BFloat16>(), alpha.to<float>());
            break;
        case DType::Float64:
            sub_scalar_kernel_cuda_impl<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<double>(), other.to<double>(), self.data_ptr<double>(), alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA sub_scalar_: unsupported dtype");
    }
    return self;
}

Tensor mul_scalar_kernel(const Tensor& self, Scalar other) {
    DType result_dtype = self.dtype();
    if (!isFloatingType(result_dtype) && other.isFloatingPoint()) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor a = (self.dtype() == result_dtype) ? self_contig : self_contig.to(result_dtype);
    
    switch (result_dtype) {
        case DType::Float32:
            mul_scalar_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<float>(), other.to<float>(), result.data_ptr<float>());
            break;
        case DType::Int32:
            mul_scalar_kernel_cuda_impl<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int>(), other.to<int>(), result.data_ptr<int>());
            break;
        case DType::Int64:
            mul_scalar_kernel_cuda_impl<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<int64_t>(), other.to<int64_t>(), result.data_ptr<int64_t>());
            break;
        case DType::Float16:
            mul_scalar_kernel_cuda_impl<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::Half>(), other.to<float>(), result.data_ptr<tensorplay::Half>());
            break;
        case DType::BFloat16:
            mul_scalar_kernel_cuda_impl<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::BFloat16>(), other.to<float>(), result.data_ptr<tensorplay::BFloat16>());
            break;
        case DType::Float64:
            mul_scalar_kernel_cuda_impl<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<double>(), other.to<double>(), result.data_ptr<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA mul_scalar: unsupported dtype");
    }
    return result;
}

Tensor& mul_scalar_inplace_kernel(Tensor& self, Scalar other) {
    int64_t n = self.numel();
    if (n == 0) return self;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    if (!self.is_contiguous()) {
         TP_THROW(NotImplementedError, "CUDA mul_scalar_: non-contiguous input not supported yet");
    }
    
    switch (self.dtype()) {
        case DType::Float32:
            mul_scalar_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<float>(), other.to<float>(), self.data_ptr<float>());
            break;
        case DType::Int32:
            mul_scalar_kernel_cuda_impl<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int>(), other.to<int>(), self.data_ptr<int>());
            break;
        case DType::Int64:
            mul_scalar_kernel_cuda_impl<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int64_t>(), other.to<int64_t>(), self.data_ptr<int64_t>());
            break;
        case DType::Float16:
            mul_scalar_kernel_cuda_impl<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::Half>(), other.to<float>(), self.data_ptr<tensorplay::Half>());
            break;
        case DType::BFloat16:
            mul_scalar_kernel_cuda_impl<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::BFloat16>(), other.to<float>(), self.data_ptr<tensorplay::BFloat16>());
            break;
        case DType::Float64:
            mul_scalar_kernel_cuda_impl<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<double>(), other.to<double>(), self.data_ptr<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA mul_scalar_: unsupported dtype");
    }
    return self;
}

Tensor div_scalar_kernel(const Tensor& self, Scalar other) {
    DType result_dtype = self.dtype();
    // True division promotes integral tensors, but preserves fp16/bf16/fp32/
    // fp64 just like torch.div(tensor, Python scalar).
    if (!isFloatingType(result_dtype)) result_dtype = DType::Float32;
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor a = (self.dtype() == result_dtype) ? self_contig : self_contig.to(result_dtype);
    
    switch (result_dtype) {
        case DType::Float32:
            div_scalar_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<float>(), other.to<float>(), result.data_ptr<float>());
            break;
        case DType::Float16:
            div_scalar_kernel_cuda_impl<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::Half>(), other.to<float>(), result.data_ptr<tensorplay::Half>());
            break;
        case DType::BFloat16:
            div_scalar_kernel_cuda_impl<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<tensorplay::BFloat16>(), other.to<float>(), result.data_ptr<tensorplay::BFloat16>());
            break;
        case DType::Float64:
            div_scalar_kernel_cuda_impl<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a.data_ptr<double>(), other.to<double>(), result.data_ptr<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA div_scalar: unsupported dtype");
    }
    return result;
}

Tensor& div_scalar_inplace_kernel(Tensor& self, Scalar other) {
    int64_t n = self.numel();
    if (n == 0) return self;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    if (!self.is_contiguous()) {
         TP_THROW(NotImplementedError, "CUDA div_scalar_: non-contiguous input not supported yet");
    }
    
    // Inplace division on integer tensor?
    // PyTorch: "RuntimeError: result type Float can't be cast to the desired output type Long"
    // Unless floor_divide.
    // Here we implement standard C++ division which for int is floor/trunc.
    // If float, it's float div.
    
    switch (self.dtype()) {
        case DType::Float32:
            div_scalar_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<float>(), other.to<float>(), self.data_ptr<float>());
            break;
        case DType::Int32:
            div_scalar_kernel_cuda_impl<int><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int>(), other.to<int>(), self.data_ptr<int>());
            break;
        case DType::Int64:
            div_scalar_kernel_cuda_impl<int64_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<int64_t>(), other.to<int64_t>(), self.data_ptr<int64_t>());
            break;
        case DType::Float16:
            div_scalar_kernel_cuda_impl<tensorplay::Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::Half>(), other.to<float>(), self.data_ptr<tensorplay::Half>());
            break;
        case DType::BFloat16:
            div_scalar_kernel_cuda_impl<tensorplay::BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<tensorplay::BFloat16>(), other.to<float>(), self.data_ptr<tensorplay::BFloat16>());
            break;
        case DType::Float64:
            div_scalar_kernel_cuda_impl<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<double>(), other.to<double>(), self.data_ptr<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA div_scalar_: unsupported dtype");
    }
    return self;
}

template <bool Divide>
Tensor addc_cuda_impl(const Tensor& self, const Tensor& tensor1,
                      const Tensor& tensor2, Scalar value) {
    if constexpr (Divide) {
        if (isIntegralType(tensor1.dtype(), true) &&
            isIntegralType(tensor2.dtype(), true)) {
            TP_THROW(RuntimeError, "Integer division with addcdiv is not supported");
        }
    }

    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(tensor1.shape()),
        static_cast<std::vector<int64_t>>(tensor2.shape()));
    DType result_dtype = promoteTypes(
        promoteTypes(self.dtype(), tensor1.dtype()), tensor2.dtype());
    if constexpr (Divide) {
        if (isIntegralType(result_dtype)) result_dtype = DType::Float32;
    }
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    const int64_t n = result.numel();
    if (n == 0) return result;
    dim3 grid, block;
    get_grid_block(n, grid, block);

    Tensor a = self.dtype() == result_dtype ? self : self.to(result_dtype);
    Tensor b = tensor1.dtype() == result_dtype ? tensor1 : tensor1.to(result_dtype);
    Tensor c = tensor2.dtype() == result_dtype ? tensor2 : tensor2.to(result_dtype);
    TensorDesc a_desc = make_desc(a, out_shape.size());
    TensorDesc b_desc = make_desc(b, out_shape.size());
    TensorDesc c_desc = make_desc(c, out_shape.size());
    TensorDesc y_desc = make_desc(result, out_shape.size());

    #define ADDC_CASE(ctype, name) \
        case DType::name: \
            addc_broadcast_kernel<ctype, Divide><<<grid, block, 0, getCurrentCUDAStream().stream()>>>( \
                n, a.data_ptr<ctype>(), a_desc, b.data_ptr<ctype>(), b_desc, \
                c.data_ptr<ctype>(), c_desc, result.data_ptr<ctype>(), y_desc, scalar_to_opmath<ctype>(value)); \
            break;
    switch (result_dtype) {
        ADDC_CASE(float, Float32)
        ADDC_CASE(double, Float64)
        ADDC_CASE(tensorplay::Half, Float16)
        ADDC_CASE(tensorplay::BFloat16, BFloat16)
        ADDC_CASE(int, Int32)
        ADDC_CASE(int64_t, Int64)
        default: TP_THROW(NotImplementedError, "CUDA addcmul/addcdiv: unsupported dtype");
    }
    #undef ADDC_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

template <bool Divide>
Tensor& addc_cuda_inplace_impl(Tensor& self, const Tensor& tensor1,
                               const Tensor& tensor2, Scalar value) {
    if constexpr (Divide) {
        if (isIntegralType(tensor1.dtype(), true) &&
            isIntegralType(tensor2.dtype(), true)) {
            TP_THROW(RuntimeError, "Integer division with addcdiv is not supported");
        }
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(tensor1.shape()),
        static_cast<std::vector<int64_t>>(tensor2.shape()));
    if (out_shape != static_cast<std::vector<int64_t>>(self.shape())) {
        TP_THROW(RuntimeError, "addcmul_/addcdiv_: output shape does not match self");
    }
    const int64_t n = self.numel();
    if (n == 0) return self;
    dim3 grid, block;
    get_grid_block(n, grid, block);

    Tensor b = tensor1.dtype() == self.dtype() ? tensor1 : tensor1.to(self.dtype());
    Tensor c = tensor2.dtype() == self.dtype() ? tensor2 : tensor2.to(self.dtype());
    TensorDesc a_desc = make_desc(self, out_shape.size());
    TensorDesc b_desc = make_desc(b, out_shape.size());
    TensorDesc c_desc = make_desc(c, out_shape.size());
    TensorDesc y_desc = make_desc(self, out_shape.size());

    #define ADDC_INPLACE_CASE(ctype, name) \
        case DType::name: \
            addc_broadcast_kernel<ctype, Divide><<<grid, block, 0, getCurrentCUDAStream().stream()>>>( \
                n, self.data_ptr<ctype>(), a_desc, b.data_ptr<ctype>(), b_desc, \
                c.data_ptr<ctype>(), c_desc, self.data_ptr<ctype>(), y_desc, scalar_to_opmath<ctype>(value)); \
            break;
    switch (self.dtype()) {
        ADDC_INPLACE_CASE(float, Float32)
        ADDC_INPLACE_CASE(double, Float64)
        ADDC_INPLACE_CASE(tensorplay::Half, Float16)
        ADDC_INPLACE_CASE(tensorplay::BFloat16, BFloat16)
        ADDC_INPLACE_CASE(int, Int32)
        ADDC_INPLACE_CASE(int64_t, Int64)
        default: TP_THROW(NotImplementedError, "CUDA addcmul_/addcdiv_: unsupported dtype");
    }
    #undef ADDC_INPLACE_CASE
    CUDA_CHECK(cudaGetLastError());
    return self;
}

Tensor addcmul_cuda(const Tensor& self, const Tensor& tensor1,
                    const Tensor& tensor2, Scalar value) {
    return addc_cuda_impl<false>(self, tensor1, tensor2, value);
}

Tensor& addcmul_inplace_cuda(Tensor& self, const Tensor& tensor1,
                             const Tensor& tensor2, Scalar value) {
    return addc_cuda_inplace_impl<false>(self, tensor1, tensor2, value);
}

Tensor addcdiv_cuda(const Tensor& self, const Tensor& tensor1,
                    const Tensor& tensor2, Scalar value) {
    return addc_cuda_impl<true>(self, tensor1, tensor2, value);
}

Tensor& addcdiv_inplace_cuda(Tensor& self, const Tensor& tensor1,
                             const Tensor& tensor2, Scalar value) {
    return addc_cuda_inplace_impl<true>(self, tensor1, tensor2, value);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ArithmeticKernels) {
    m.impl("add.Tensor", add_kernel);
    m.impl("add_relu", add_relu_cuda);
    m.impl("add_.Tensor", add_inplace_kernel);
    m.impl("add.Scalar", add_scalar_kernel);
    m.impl("add_.Scalar", add_scalar_inplace_kernel);
    
    m.impl("sub.Tensor", sub_kernel);
    m.impl("sub_.Tensor", sub_inplace_kernel);
    m.impl("sub.Scalar", sub_scalar_kernel);
    m.impl("sub_.Scalar", sub_scalar_inplace_kernel);
    
    m.impl("mul.Tensor", mul_kernel);
    m.impl("fused_mul_add", fused_mul_add_kernel);
    m.impl("fused_mul_add.Scalar", fused_mul_add_scalar_kernel);
    m.impl("mul_.Tensor", mul_inplace_kernel);
    m.impl("mul.Scalar", mul_scalar_kernel);
    m.impl("mul_.Scalar", mul_scalar_inplace_kernel);
    
    m.impl("div.Tensor", div_kernel);
    m.impl("addcmul", addcmul_cuda);
    m.impl("addcmul_", addcmul_inplace_cuda);
    m.impl("addcdiv", addcdiv_cuda);
    m.impl("addcdiv_", addcdiv_inplace_cuda);
    m.impl("div_.Tensor", div_inplace_kernel);
    m.impl("div.Scalar", div_scalar_kernel);
    m.impl("div_.Scalar", div_scalar_inplace_kernel);
}

} // namespace cuda
} // namespace tensorplay
