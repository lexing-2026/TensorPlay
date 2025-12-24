#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "Allocator.h"
#include "CUDNNUtils.h"
#include "Utils.h"
#include "TypePromotion.h"
#include <cuda_runtime.h>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace tensorplay {
namespace cuda {

// --- Utils ---
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

#define MAX_DIMS 8

struct TensorDesc {
    int64_t shape[MAX_DIMS];
    int64_t strides[MAX_DIMS];
    int ndim;
};

// Pad shapes/strides to output ndim (left padding with 1s and 0 strides for broadcast or just 1s and dummy strides)
// Actually for broadcasting, we need to align dimensions from the right.
// But broadcast_shapes aligns them.
// We assume we have the output shape. We need to "view" inputs as having that rank.
// If input has rank K and output has rank N (N >= K).
// Input is padded on the left with 1s.
// Strides for those 1s should be 0 (conceptually) or just 0 so they don't contribute to offset?
// If dim is 1, offset contribution is 0 anyway.
TensorDesc make_desc(const Tensor& t, int ndim) {
    TensorDesc desc;
    desc.ndim = ndim;
    int t_ndim = t.dim();
    int diff = ndim - t_ndim;
    
    for (int i = 0; i < ndim; ++i) {
        if (i < diff) {
            desc.shape[i] = 1;
            desc.strides[i] = 0; 
        } else {
            desc.shape[i] = t.shape()[i - diff];
            desc.strides[i] = t.strides()[i - diff];
        }
    }
    return desc;
}

TensorDesc make_desc_from_shape(const std::vector<int64_t>& shape) {
    TensorDesc desc;
    desc.ndim = shape.size();
    int64_t stride = 1;
    for (int i = desc.ndim - 1; i >= 0; --i) {
        desc.shape[i] = shape[i];
        desc.strides[i] = stride;
        stride *= shape[i];
    }
    return desc;
}

__device__ int64_t get_offset(int64_t idx, const TensorDesc& desc, const TensorDesc& out_desc) {
    int64_t offset = 0;
    // We compute coordinates from idx based on out_desc, then map to input offset
    for (int i = out_desc.ndim - 1; i >= 0; --i) {
        int64_t coord = idx % out_desc.shape[i];
        idx /= out_desc.shape[i];
        
        if (desc.shape[i] != 1) {
            offset += coord * desc.strides[i];
        }
    }
    return offset;
}

// --- Kernels ---

// DIV Kernel with broadcasting
template <typename T>
__global__ void div_broadcast_kernel(int64_t n, 
                                     const T* a, TensorDesc a_desc,
                                     const T* b, TensorDesc b_desc,
                                     T* y, TensorDesc y_desc) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t a_off = get_offset(i, a_desc, y_desc);
        int64_t b_off = get_offset(i, b_desc, y_desc);
        y[i] = a[a_off] / b[b_off];
    }
}

template <typename T>
__global__ void add_broadcast_kernel(int64_t n, 
                                     const T* a, TensorDesc a_desc,
                                     const T* b, TensorDesc b_desc,
                                     T* y, TensorDesc y_desc, T alpha) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t a_off = get_offset(i, a_desc, y_desc);
        int64_t b_off = get_offset(i, b_desc, y_desc);
        y[i] = a[a_off] + alpha * b[b_off];
    }
}

template <typename T>
__global__ void sub_broadcast_kernel(int64_t n, 
                                     const T* a, TensorDesc a_desc,
                                     const T* b, TensorDesc b_desc,
                                     T* y, TensorDesc y_desc, T alpha) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t a_off = get_offset(i, a_desc, y_desc);
        int64_t b_off = get_offset(i, b_desc, y_desc);
        y[i] = a[a_off] - alpha * b[b_off];
    }
}

template <typename T>
__global__ void mul_broadcast_kernel(int64_t n, 
                                     const T* a, TensorDesc a_desc,
                                     const T* b, TensorDesc b_desc,
                                     T* y, TensorDesc y_desc) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t a_off = get_offset(i, a_desc, y_desc);
        int64_t b_off = get_offset(i, b_desc, y_desc);
        y[i] = a[a_off] * b[b_off];
    }
}

// Tensor-Scalar Kernels
template <typename T>
__global__ void add_scalar_kernel_cuda_impl(int n, const T* a, T b, T* y, T alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + alpha * b;
}

template <typename T>
__global__ void sub_scalar_kernel_cuda_impl(int n, const T* a, T b, T* y, T alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] - alpha * b;
}

template <typename T>
__global__ void mul_scalar_kernel_cuda_impl(int n, const T* a, T b, T* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b;
}

template <typename T>
__global__ void div_scalar_kernel_cuda_impl(int n, const T* a, T b, T* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b;
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
            add_broadcast_kernel<float><<<grid, block>>>(n, a.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, result.data_ptr<float>(), y_desc, alpha.to<float>());
            break;
        case DType::Int32:
            add_broadcast_kernel<int><<<grid, block>>>(n, a.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, result.data_ptr<int>(), y_desc, alpha.to<int>());
            break;
        case DType::Int64:
            add_broadcast_kernel<int64_t><<<grid, block>>>(n, a.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, result.data_ptr<int64_t>(), y_desc, alpha.to<int64_t>());
            break;
        case DType::Float64:
            add_broadcast_kernel<double><<<grid, block>>>(n, a.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, result.data_ptr<double>(), y_desc, alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA add: unsupported dtype");
    }
    return result;
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
            add_broadcast_kernel<float><<<grid, block>>>(n, self.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, self.data_ptr<float>(), y_desc, alpha.to<float>());
            break;
        case DType::Int32:
            add_broadcast_kernel<int><<<grid, block>>>(n, self.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, self.data_ptr<int>(), y_desc, alpha.to<int>());
            break;
        case DType::Int64:
            add_broadcast_kernel<int64_t><<<grid, block>>>(n, self.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, self.data_ptr<int64_t>(), y_desc, alpha.to<int64_t>());
            break;
        case DType::Float64:
            add_broadcast_kernel<double><<<grid, block>>>(n, self.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, self.data_ptr<double>(), y_desc, alpha.to<double>());
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
            sub_broadcast_kernel<float><<<grid, block>>>(n, a.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, result.data_ptr<float>(), y_desc, alpha.to<float>());
            break;
        case DType::Int32:
            sub_broadcast_kernel<int><<<grid, block>>>(n, a.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, result.data_ptr<int>(), y_desc, alpha.to<int>());
            break;
        case DType::Int64:
            sub_broadcast_kernel<int64_t><<<grid, block>>>(n, a.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, result.data_ptr<int64_t>(), y_desc, alpha.to<int64_t>());
            break;
        case DType::Float64:
            sub_broadcast_kernel<double><<<grid, block>>>(n, a.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, result.data_ptr<double>(), y_desc, alpha.to<double>());
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
            sub_broadcast_kernel<float><<<grid, block>>>(n, self.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, self.data_ptr<float>(), y_desc, alpha.to<float>());
            break;
        case DType::Int32:
            sub_broadcast_kernel<int><<<grid, block>>>(n, self.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, self.data_ptr<int>(), y_desc, alpha.to<int>());
            break;
        case DType::Int64:
            sub_broadcast_kernel<int64_t><<<grid, block>>>(n, self.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, self.data_ptr<int64_t>(), y_desc, alpha.to<int64_t>());
            break;
        case DType::Float64:
            sub_broadcast_kernel<double><<<grid, block>>>(n, self.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, self.data_ptr<double>(), y_desc, alpha.to<double>());
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
            mul_broadcast_kernel<float><<<grid, block>>>(n, a.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, result.data_ptr<float>(), y_desc);
            break;
        case DType::Int32:
            mul_broadcast_kernel<int><<<grid, block>>>(n, a.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, result.data_ptr<int>(), y_desc);
            break;
        case DType::Int64:
            mul_broadcast_kernel<int64_t><<<grid, block>>>(n, a.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, result.data_ptr<int64_t>(), y_desc);
            break;
        case DType::Float64:
            mul_broadcast_kernel<double><<<grid, block>>>(n, a.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, result.data_ptr<double>(), y_desc);
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
            mul_broadcast_kernel<float><<<grid, block>>>(n, self.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, self.data_ptr<float>(), y_desc);
            break;
        case DType::Int32:
            mul_broadcast_kernel<int><<<grid, block>>>(n, self.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, self.data_ptr<int>(), y_desc);
            break;
        case DType::Int64:
            mul_broadcast_kernel<int64_t><<<grid, block>>>(n, self.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, self.data_ptr<int64_t>(), y_desc);
            break;
        case DType::Float64:
            mul_broadcast_kernel<double><<<grid, block>>>(n, self.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, self.data_ptr<double>(), y_desc);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA mul_: unsupported dtype");
    }
    return self;
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
            div_broadcast_kernel<float><<<grid, block>>>(n, a.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, result.data_ptr<float>(), y_desc);
            break;
        case DType::Float64:
            div_broadcast_kernel<double><<<grid, block>>>(n, a.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, result.data_ptr<double>(), y_desc);
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
            div_broadcast_kernel<float><<<grid, block>>>(n, self.data_ptr<float>(), a_desc, b.data_ptr<float>(), b_desc, self.data_ptr<float>(), y_desc);
            break;
        case DType::Int32:
            div_broadcast_kernel<int><<<grid, block>>>(n, self.data_ptr<int>(), a_desc, b.data_ptr<int>(), b_desc, self.data_ptr<int>(), y_desc);
            break;
        case DType::Int64:
            div_broadcast_kernel<int64_t><<<grid, block>>>(n, self.data_ptr<int64_t>(), a_desc, b.data_ptr<int64_t>(), b_desc, self.data_ptr<int64_t>(), y_desc);
            break;
        case DType::Float64:
            div_broadcast_kernel<double><<<grid, block>>>(n, self.data_ptr<double>(), a_desc, b.data_ptr<double>(), b_desc, self.data_ptr<double>(), y_desc);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA div_: unsupported dtype");
    }
    return self;
}


Tensor add_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
    DType result_dtype = self.dtype();
    if (other.isFloatingPoint() || alpha.isFloatingPoint()) {
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
            add_scalar_kernel_cuda_impl<float><<<grid, block>>>(n, a.data_ptr<float>(), other.to<float>(), result.data_ptr<float>(), alpha.to<float>());
            break;
        case DType::Int32:
            add_scalar_kernel_cuda_impl<int><<<grid, block>>>(n, a.data_ptr<int>(), other.to<int>(), result.data_ptr<int>(), alpha.to<int>());
            break;
        case DType::Int64:
            add_scalar_kernel_cuda_impl<int64_t><<<grid, block>>>(n, a.data_ptr<int64_t>(), other.to<int64_t>(), result.data_ptr<int64_t>(), alpha.to<int64_t>());
            break;
        case DType::Float64:
            add_scalar_kernel_cuda_impl<double><<<grid, block>>>(n, a.data_ptr<double>(), other.to<double>(), result.data_ptr<double>(), alpha.to<double>());
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
            add_scalar_kernel_cuda_impl<float><<<grid, block>>>(n, self.data_ptr<float>(), other.to<float>(), self.data_ptr<float>(), alpha.to<float>());
            break;
        case DType::Int32:
            add_scalar_kernel_cuda_impl<int><<<grid, block>>>(n, self.data_ptr<int>(), other.to<int>(), self.data_ptr<int>(), alpha.to<int>());
            break;
        case DType::Int64:
            add_scalar_kernel_cuda_impl<int64_t><<<grid, block>>>(n, self.data_ptr<int64_t>(), other.to<int64_t>(), self.data_ptr<int64_t>(), alpha.to<int64_t>());
            break;
        case DType::Float64:
            add_scalar_kernel_cuda_impl<double><<<grid, block>>>(n, self.data_ptr<double>(), other.to<double>(), self.data_ptr<double>(), alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA add_scalar_: unsupported dtype");
    }
    return self;
}

Tensor sub_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
    DType result_dtype = self.dtype();
    if (other.isFloatingPoint() || alpha.isFloatingPoint()) {
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
            sub_scalar_kernel_cuda_impl<float><<<grid, block>>>(n, a.data_ptr<float>(), other.to<float>(), result.data_ptr<float>(), alpha.to<float>());
            break;
        case DType::Int32:
            sub_scalar_kernel_cuda_impl<int><<<grid, block>>>(n, a.data_ptr<int>(), other.to<int>(), result.data_ptr<int>(), alpha.to<int>());
            break;
        case DType::Int64:
            sub_scalar_kernel_cuda_impl<int64_t><<<grid, block>>>(n, a.data_ptr<int64_t>(), other.to<int64_t>(), result.data_ptr<int64_t>(), alpha.to<int64_t>());
            break;
        case DType::Float64:
            sub_scalar_kernel_cuda_impl<double><<<grid, block>>>(n, a.data_ptr<double>(), other.to<double>(), result.data_ptr<double>(), alpha.to<double>());
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
            sub_scalar_kernel_cuda_impl<float><<<grid, block>>>(n, self.data_ptr<float>(), other.to<float>(), self.data_ptr<float>(), alpha.to<float>());
            break;
        case DType::Int32:
            sub_scalar_kernel_cuda_impl<int><<<grid, block>>>(n, self.data_ptr<int>(), other.to<int>(), self.data_ptr<int>(), alpha.to<int>());
            break;
        case DType::Int64:
            sub_scalar_kernel_cuda_impl<int64_t><<<grid, block>>>(n, self.data_ptr<int64_t>(), other.to<int64_t>(), self.data_ptr<int64_t>(), alpha.to<int64_t>());
            break;
        case DType::Float64:
            sub_scalar_kernel_cuda_impl<double><<<grid, block>>>(n, self.data_ptr<double>(), other.to<double>(), self.data_ptr<double>(), alpha.to<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA sub_scalar_: unsupported dtype");
    }
    return self;
}

Tensor mul_scalar_kernel(const Tensor& self, Scalar other) {
    DType result_dtype = self.dtype();
    if (other.isFloatingPoint()) {
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
            mul_scalar_kernel_cuda_impl<float><<<grid, block>>>(n, a.data_ptr<float>(), other.to<float>(), result.data_ptr<float>());
            break;
        case DType::Int32:
            mul_scalar_kernel_cuda_impl<int><<<grid, block>>>(n, a.data_ptr<int>(), other.to<int>(), result.data_ptr<int>());
            break;
        case DType::Int64:
            mul_scalar_kernel_cuda_impl<int64_t><<<grid, block>>>(n, a.data_ptr<int64_t>(), other.to<int64_t>(), result.data_ptr<int64_t>());
            break;
        case DType::Float64:
            mul_scalar_kernel_cuda_impl<double><<<grid, block>>>(n, a.data_ptr<double>(), other.to<double>(), result.data_ptr<double>());
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
            mul_scalar_kernel_cuda_impl<float><<<grid, block>>>(n, self.data_ptr<float>(), other.to<float>(), self.data_ptr<float>());
            break;
        case DType::Int32:
            mul_scalar_kernel_cuda_impl<int><<<grid, block>>>(n, self.data_ptr<int>(), other.to<int>(), self.data_ptr<int>());
            break;
        case DType::Int64:
            mul_scalar_kernel_cuda_impl<int64_t><<<grid, block>>>(n, self.data_ptr<int64_t>(), other.to<int64_t>(), self.data_ptr<int64_t>());
            break;
        case DType::Float64:
            mul_scalar_kernel_cuda_impl<double><<<grid, block>>>(n, self.data_ptr<double>(), other.to<double>(), self.data_ptr<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA mul_scalar_: unsupported dtype");
    }
    return self;
}

Tensor div_scalar_kernel(const Tensor& self, Scalar other) {
    DType result_dtype = self.dtype();
    result_dtype = promoteTypes(result_dtype, DType::Float32); // Always promote to float
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 grid, block; get_grid_block(n, grid, block);
    
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor a = (self.dtype() == result_dtype) ? self_contig : self_contig.to(result_dtype);
    
    switch (result_dtype) {
        case DType::Float32:
            div_scalar_kernel_cuda_impl<float><<<grid, block>>>(n, a.data_ptr<float>(), other.to<float>(), result.data_ptr<float>());
            break;
        case DType::Float64:
            div_scalar_kernel_cuda_impl<double><<<grid, block>>>(n, a.data_ptr<double>(), other.to<double>(), result.data_ptr<double>());
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
            div_scalar_kernel_cuda_impl<float><<<grid, block>>>(n, self.data_ptr<float>(), other.to<float>(), self.data_ptr<float>());
            break;
        case DType::Int32:
            div_scalar_kernel_cuda_impl<int><<<grid, block>>>(n, self.data_ptr<int>(), other.to<int>(), self.data_ptr<int>());
            break;
        case DType::Int64:
            div_scalar_kernel_cuda_impl<int64_t><<<grid, block>>>(n, self.data_ptr<int64_t>(), other.to<int64_t>(), self.data_ptr<int64_t>());
            break;
        case DType::Float64:
            div_scalar_kernel_cuda_impl<double><<<grid, block>>>(n, self.data_ptr<double>(), other.to<double>(), self.data_ptr<double>());
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA div_scalar_: unsupported dtype");
    }
    return self;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ArithmeticKernels) {
    m.impl("add.Tensor", add_kernel);
    m.impl("add_.Tensor", add_inplace_kernel);
    m.impl("add.Scalar", add_scalar_kernel);
    m.impl("add_.Scalar", add_scalar_inplace_kernel);
    
    m.impl("sub.Tensor", sub_kernel);
    m.impl("sub_.Tensor", sub_inplace_kernel);
    m.impl("sub.Scalar", sub_scalar_kernel);
    m.impl("sub_.Scalar", sub_scalar_inplace_kernel);
    
    m.impl("mul.Tensor", mul_kernel);
    m.impl("mul_.Tensor", mul_inplace_kernel);
    m.impl("mul.Scalar", mul_scalar_kernel);
    m.impl("mul_.Scalar", mul_scalar_inplace_kernel);
    
    m.impl("div.Tensor", div_kernel);
    m.impl("div_.Tensor", div_inplace_kernel);
    m.impl("div.Scalar", div_scalar_kernel);
    m.impl("div_.Scalar", div_scalar_inplace_kernel);
}

} // namespace cuda
} // namespace tensorplay
