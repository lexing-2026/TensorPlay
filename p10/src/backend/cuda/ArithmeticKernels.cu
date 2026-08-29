#include "Tensor.h"
#include "SparseKernels.h"
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
#include "CUDAComplex.cuh"
#include <thrust/complex.h>

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

// native/cuda binary kernels.
template <typename T> struct BinaryOpMath { using type = T; };
template <> struct BinaryOpMath<tensorplay::Half> { using type = float; };
template <> struct BinaryOpMath<tensorplay::BFloat16> { using type = float; };

template <typename T>
inline typename BinaryOpMath<T>::type scalar_to_opmath(const Scalar& s) {
    return s.to<typename BinaryOpMath<T>::type>();
}

// --- complex (thrust::complex on interleaved storage) -----------------------
// Same-shape contiguous inputs take the flat kernel; everything else goes
// through the TensorDesc broadcast kernel. `alpha` rides in the functor.
template <typename T, typename OpF>
static void run_cplx_binary(const Tensor& a, const Tensor& b, Tensor& y,
                            OpF op) {
    auto stream = getCurrentCUDAStream().stream();
    const int64_t n = y.numel();
    bool same = a.is_contiguous() && b.is_contiguous() && y.is_contiguous() &&
                a.dim() == static_cast<int64_t>(y.dim()) &&
                b.dim() == static_cast<int64_t>(y.dim());
    if (same) {
        for (int64_t d = 0; d < static_cast<int64_t>(y.dim()); ++d) {
            if (a.size(d) != y.size(d) || b.size(d) != y.size(d)) {
                same = false;
                break;
            }
        }
    }
    if (same) {
        cuda::cplx::launch_binary<T>(n, a.data_ptr(), b.data_ptr(),
                                     y.data_ptr(), op, stream);
    } else {
        TensorDesc ad = make_desc(a, static_cast<size_t>(y.dim()));
        TensorDesc bd = make_desc(b, static_cast<size_t>(y.dim()));
        TensorDesc yd = make_desc(y, static_cast<size_t>(y.dim()));
        cuda::cplx::launch_binary_broadcast<T>(n, a.data_ptr(), ad,
                                               b.data_ptr(), bd, y.data_ptr(),
                                               yd, op, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename T> inline thrust::complex<T> s2c(const Scalar& s);
template <> inline thrust::complex<float> s2c<float>(const Scalar& s) {
    return cuda::cplx::to_c64(s);
}
template <> inline thrust::complex<double> s2c<double>(const Scalar& s) {
    return cuda::cplx::to_c128(s);
}

template <typename T>
static void run_cplx_add_scalar(Tensor& x, const Scalar& other,
                                const Scalar& alpha, Tensor& y) {
    cuda::cplx::add_scalar_kernel_impl<T>
        <<<cuda::cplx::default_grid(x.numel()), cuda::cplx::default_block(),
           0, getCurrentCUDAStream().stream()>>>(
            x.numel(),
            static_cast<const thrust::complex<T>*>(x.data_ptr()),
            s2c<T>(other),
            s2c<T>(alpha),
            static_cast<thrust::complex<T>*>(y.data_ptr()));
    CUDA_CHECK(cudaGetLastError());
}
template <typename T>
static void run_cplx_sub_scalar(Tensor& x, const Scalar& other,
                                const Scalar& alpha, Tensor& y) {
    cuda::cplx::sub_scalar_kernel_impl<T>
        <<<cuda::cplx::default_grid(x.numel()), cuda::cplx::default_block(),
           0, getCurrentCUDAStream().stream()>>>(
            x.numel(),
            static_cast<const thrust::complex<T>*>(x.data_ptr()),
            s2c<T>(other),
            s2c<T>(alpha),
            static_cast<thrust::complex<T>*>(y.data_ptr()));
    CUDA_CHECK(cudaGetLastError());
}
template <typename T>
static void run_cplx_mul_scalar(Tensor& x, const Scalar& other, Tensor& y) {
    cuda::cplx::mul_scalar_kernel_impl<T>
        <<<cuda::cplx::default_grid(x.numel()), cuda::cplx::default_block(),
           0, getCurrentCUDAStream().stream()>>>(
            x.numel(),
            static_cast<const thrust::complex<T>*>(x.data_ptr()),
            s2c<T>(other),
            static_cast<thrust::complex<T>*>(y.data_ptr()));
    CUDA_CHECK(cudaGetLastError());
}
template <typename T>
static void run_cplx_div_scalar(Tensor& x, const Scalar& other, Tensor& y) {
    cuda::cplx::div_scalar_kernel_impl<T>
        <<<cuda::cplx::default_grid(x.numel()), cuda::cplx::default_block(),
           0, getCurrentCUDAStream().stream()>>>(
            x.numel(),
            static_cast<const thrust::complex<T>*>(x.data_ptr()),
            s2c<T>(other),
            static_cast<thrust::complex<T>*>(y.data_ptr()));
    CUDA_CHECK(cudaGetLastError());
}

#define TP_CUDA_GRIDSTRIDE(i) \
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    int64_t tp_stride = static_cast<int64_t>(blockDim.x) * gridDim.x; \
    for (; i < n; i += tp_stride)

// Forward declarations for the scalar fallback used by the fused kernel.
Tensor add_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha);
Tensor mul_scalar_kernel(const Tensor& self, Scalar other);
#ifdef USE_CUDNN
Tensor& relu_inplace_kernel_cudnn(Tensor& self);
#else
Tensor relu_kernel_cuda(const Tensor& self);
#endif

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

// --- Vectorized same-shape fast path ---
// (get_offset == identity), so the general broadcast machinery is skipped in

template <typename T, int VecSize>
struct alignas(VecSize * sizeof(T)) TPVecPack { T v[VecSize]; };

struct BinaryAddVecOp { template <typename M> __device__ M operator()(M x, M y, M a) const { return x + a * y; } };
struct BinarySubVecOp { template <typename M> __device__ M operator()(M x, M y, M a) const { return x - a * y; } };
struct BinaryMulVecOp { template <typename M> __device__ M operator()(M x, M y, M) const { return x * y; } };
struct BinaryDivVecOp { template <typename M> __device__ M operator()(M x, M y, M) const { return x / y; } };

template <typename T, int VecSize, typename Op>
__global__ void binary_same_shape_vectorized_kernel(
    int64_t n, const T* __restrict__ a, const T* __restrict__ b,
    T* __restrict__ y, typename BinaryOpMath<T>::type alpha, Op op) {
    using M = typename BinaryOpMath<T>::type;
    const int64_t vec_n = n / VecSize;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < vec_n; i += stride) {
        TPVecPack<T, VecSize> pa = *reinterpret_cast<const TPVecPack<T, VecSize>*>(a + i * VecSize);
        TPVecPack<T, VecSize> pb = *reinterpret_cast<const TPVecPack<T, VecSize>*>(b + i * VecSize);
        TPVecPack<T, VecSize> po;
#pragma unroll
        for (int v = 0; v < VecSize; ++v)
            po.v[v] = static_cast<T>(op(static_cast<M>(pa.v[v]), static_cast<M>(pb.v[v]), static_cast<M>(alpha)));
        *reinterpret_cast<TPVecPack<T, VecSize>*>(y + i * VecSize) = po;
    }
    for (int64_t j = vec_n * VecSize + i; j < n; j += stride) {
        y[j] = static_cast<T>(op(static_cast<M>(a[j]), static_cast<M>(b[j]), static_cast<M>(alpha)));
    }
}

inline int arith_device_sms() {
    static int sms = []() {
        int dev = 0, count = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev);
        return count > 0 ? count : 1;
    }();
    return sms;
}

template <typename T, typename Op>
inline bool launch_binary_vec(
    int64_t n, const Tensor& a, const Tensor& b, Tensor& y,
    typename BinaryOpMath<T>::type alpha, Op op, cudaStream_t stream) {
    constexpr int kVec = 4;
    const T* pa = a.data_ptr<T>();
    const T* pb = b.data_ptr<T>();
    T* py = y.data_ptr<T>();
    const uintptr_t align_mask = sizeof(T) * kVec - 1;
    if ((reinterpret_cast<uintptr_t>(pa) | reinterpret_cast<uintptr_t>(pb) |
         reinterpret_cast<uintptr_t>(py)) & align_mask) return false;

    dim3 block(128);
    const int64_t vec_n = n / kVec;
    const int64_t want = (vec_n + block.x - 1) / block.x;
    const int64_t cap = static_cast<int64_t>(arith_device_sms()) * 4;
    dim3 grid(static_cast<unsigned>(want < 1 ? 1 : (want > cap ? cap : want)));
    binary_same_shape_vectorized_kernel<T, kVec, Op><<<grid, block, 0, stream>>>(
        n, pa, pb, py, alpha, op);
    return true;
}

// Returns true when the vectorized kernel was launched.  Contiguous operands
// with full numel overlap make get_offset(i) == i for every input.
template <typename Op>
inline bool try_binary_vectorized(
    int64_t n, const Tensor& a, const Tensor& b, Tensor& y,
    const Scalar& alpha, Op op) {
    constexpr int64_t kMinElems = 4096;
    if (n < kMinElems || n % 4 != 0) return false;
    if (!a.is_contiguous() || !b.is_contiguous() || !y.is_contiguous()) return false;
    if (a.numel() != n || b.numel() != n) return false;

    const auto stream = getCurrentCUDAStream().stream();
    switch (y.dtype()) {
        case DType::Float32: return launch_binary_vec<float>(n, a, b, y, alpha.to<float>(), op, stream);
        case DType::Float64: return launch_binary_vec<double>(n, a, b, y, alpha.to<double>(), op, stream);
        case DType::Float16: return launch_binary_vec<tensorplay::Half>(n, a, b, y, alpha.to<float>(), op, stream);
        case DType::BFloat16: return launch_binary_vec<tensorplay::BFloat16>(n, a, b, y, alpha.to<float>(), op, stream);
        case DType::Int32: return launch_binary_vec<int>(n, a, b, y, alpha.to<int>(), op, stream);
        case DType::Int64: return launch_binary_vec<int64_t>(n, a, b, y, alpha.to<int64_t>(), op, stream);
        default: return false;
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

    if (try_binary_vectorized(n, a, b, result, alpha, BinaryAddVecOp{})) {
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

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
        case DType::ComplexFloat:
            run_cplx_binary<float>(a, b, result,
                                   cuda::cplx::AddAlphaOp<float>{s2c<float>(alpha)});
            break;
        case DType::ComplexDouble:
            run_cplx_binary<double>(a, b, result,
                                    cuda::cplx::AddAlphaOp<double>{s2c<double>(alpha)});
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
#ifdef USE_CUDNN
    return relu_inplace_kernel_cudnn(result);
#else
    return relu_kernel_cuda(result);
#endif
}

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
    if (other.is_sparse()) {
        return add_sparse_to_dense_cuda(self, other, alpha);
    }
    int64_t n = self.numel();
    if (n == 0) return self;

    // For inplace, we cast other to self.dtype()
    Tensor b = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());

    // the out-of-place add); the broadcast machinery below is only needed
    // when shapes/strides actually differ.
    if (try_binary_vectorized(n, self, b, self, alpha, BinaryAddVecOp{})) {
        return self;
    }

    dim3 grid, block; get_grid_block(n, grid, block);

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
        case DType::ComplexFloat:
            run_cplx_binary<float>(self, b, self,
                                   cuda::cplx::AddAlphaOp<float>{s2c<float>(alpha)});
            break;
        case DType::ComplexDouble:
            run_cplx_binary<double>(self, b, self,
                                    cuda::cplx::AddAlphaOp<double>{s2c<double>(alpha)});
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

    if (try_binary_vectorized(n, a, b, result, alpha, BinarySubVecOp{})) {
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

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
        case DType::ComplexFloat:
            run_cplx_binary<float>(a, b, result,
                                   cuda::cplx::SubAlphaOp<float>{s2c<float>(alpha)});
            break;
        case DType::ComplexDouble:
            run_cplx_binary<double>(a, b, result,
                                    cuda::cplx::SubAlphaOp<double>{s2c<double>(alpha)});
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA sub: unsupported dtype");
    }
    return result;
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
    int64_t n = self.numel();
    if (n == 0) return self;

    Tensor b = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());

    if (try_binary_vectorized(n, self, b, self, alpha, BinarySubVecOp{})) {
        return self;
    }

    dim3 grid, block; get_grid_block(n, grid, block);

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
        case DType::ComplexFloat:
            run_cplx_binary<float>(self, b, self,
                                   cuda::cplx::SubAlphaOp<float>{s2c<float>(alpha)});
            break;
        case DType::ComplexDouble:
            run_cplx_binary<double>(self, b, self,
                                    cuda::cplx::SubAlphaOp<double>{s2c<double>(alpha)});
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

    if (try_binary_vectorized(n, a, b, result, Scalar(1), BinaryMulVecOp{})) {
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

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
        case DType::ComplexFloat:
            run_cplx_binary<float>(a, b, result, cuda::cplx::MulOp{});
            break;
        case DType::ComplexDouble:
            run_cplx_binary<double>(a, b, result, cuda::cplx::MulOp{});
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA mul: unsupported dtype");
    }
    return result;
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
    int64_t n = self.numel();
    if (n == 0) return self;

    Tensor b = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());

    if (try_binary_vectorized(n, self, b, self, Scalar(1), BinaryMulVecOp{})) {
        return self;
    }

    dim3 grid, block; get_grid_block(n, grid, block);

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
        case DType::ComplexFloat:
            run_cplx_binary<float>(self, b, self, cuda::cplx::MulOp{});
            break;
        case DType::ComplexDouble:
            run_cplx_binary<double>(self, b, self, cuda::cplx::MulOp{});
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

    if (try_binary_vectorized(n, a, b, result, Scalar(1), BinaryDivVecOp{})) {
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

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
        case DType::ComplexFloat:
            run_cplx_binary<float>(a, b, result, cuda::cplx::DivOp{});
            break;
        case DType::ComplexDouble:
            run_cplx_binary<double>(a, b, result, cuda::cplx::DivOp{});
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA div: unsupported dtype");
    }
    return result;
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
    int64_t n = self.numel();
    if (n == 0) return self;

    // Inplace div might change dtype if self is int (e.g. 5/2 = 2 or 2.5?)
    // "RuntimeError: result type Float can't be cast to the desired output type Long" usually.
    // For now, let's assume we do standard div and cast back.

    Tensor b = (other.dtype() == self.dtype()) ? other : other.to(self.dtype());

    if (try_binary_vectorized(n, self, b, self, Scalar(1), BinaryDivVecOp{})) {
        return self;
    }

    dim3 grid, block; get_grid_block(n, grid, block);

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
        case DType::ComplexFloat:
            run_cplx_binary<float>(self, b, self, cuda::cplx::DivOp{});
            break;
        case DType::ComplexDouble:
            run_cplx_binary<double>(self, b, self, cuda::cplx::DivOp{});
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA div_: unsupported dtype");
    }
    return self;
}


Tensor add_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
    DType result_dtype = cuda::cplx::scalar_result_dtype(
        self.dtype(), other, &alpha);
    
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
        case DType::ComplexFloat:
            run_cplx_add_scalar<float>(a, other, alpha, result);
            break;
        case DType::ComplexDouble:
            run_cplx_add_scalar<double>(a, other, alpha, result);
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
        case DType::ComplexFloat:
            run_cplx_add_scalar<float>(self, other, alpha, self);
            break;
        case DType::ComplexDouble:
            run_cplx_add_scalar<double>(self, other, alpha, self);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA add_scalar_: unsupported dtype");
    }
    return self;
}

Tensor sub_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
    DType result_dtype = cuda::cplx::scalar_result_dtype(
        self.dtype(), other, &alpha);
    
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
        case DType::ComplexFloat:
            run_cplx_sub_scalar<float>(a, other, alpha, result);
            break;
        case DType::ComplexDouble:
            run_cplx_sub_scalar<double>(a, other, alpha, result);
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
        case DType::ComplexFloat:
            run_cplx_sub_scalar<float>(self, other, alpha, self);
            break;
        case DType::ComplexDouble:
            run_cplx_sub_scalar<double>(self, other, alpha, self);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA sub_scalar_: unsupported dtype");
    }
    return self;
}

Tensor mul_scalar_kernel(const Tensor& self, Scalar other) {
    DType result_dtype = cuda::cplx::scalar_result_dtype(self.dtype(), other);
    
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
        case DType::ComplexFloat:
            run_cplx_mul_scalar<float>(a, other, result);
            break;
        case DType::ComplexDouble:
            run_cplx_mul_scalar<double>(a, other, result);
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
        case DType::ComplexFloat:
            run_cplx_mul_scalar<float>(self, other, self);
            break;
        case DType::ComplexDouble:
            run_cplx_mul_scalar<double>(self, other, self);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA mul_scalar_: unsupported dtype");
    }
    return self;
}

Tensor div_scalar_kernel(const Tensor& self, Scalar other) {
    DType result_dtype = self.dtype();
    // True division promotes integral tensors to Float32 (ComplexFloat for a
    if (!isFloatingOrComplexType(result_dtype)) {
        result_dtype = other.isComplex() ? DType::ComplexFloat : DType::Float32;
    } else if (!isComplexType(result_dtype) && other.isComplex()) {
        result_dtype = promoteTypes(toComplexType(result_dtype), other.dtype());
    }
    
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
        case DType::ComplexFloat:
            run_cplx_div_scalar<float>(a, other, result);
            break;
        case DType::ComplexDouble:
            run_cplx_div_scalar<double>(a, other, result);
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
        case DType::ComplexFloat:
            run_cplx_div_scalar<float>(self, other, self);
            break;
        case DType::ComplexDouble:
            run_cplx_div_scalar<double>(self, other, self);
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
