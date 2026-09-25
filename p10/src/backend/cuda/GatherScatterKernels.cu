#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include "GPUPrimitives.cuh"
#include "Complex.h"
#include "Atomic.cuh"

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

constexpr int kThreads = 256;

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim - ndim, ")");
    }
    return dim;
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

template <typename T, typename IndexT>
__global__ void gather_kernel(int64_t n, int64_t idx_dim_size, int64_t idx_inner,
                              int64_t self_dim_size, int64_t self_inner,
                              const T* s, const IndexT* ip, T* d) {
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; flat < n; flat += stride) {
        int64_t rem = flat;
        int64_t outer_off = rem / (idx_dim_size * idx_inner);
        rem -= outer_off * idx_dim_size * idx_inner;
        int64_t t = rem % idx_inner;
        int64_t idx = ip[flat];
        if (idx < 0) idx += self_dim_size;
        d[flat] = s[(outer_off * self_dim_size + idx) * self_inner + t];
    }
}

template <typename IndexT>
void launch_gather_kernel_for_index(
        int64_t n, int64_t idx_dim_size, int64_t idx_inner,
        int64_t self_dim_size, int64_t self_inner, const Tensor& self,
        const Tensor& index, Tensor& result, cudaStream_t stream) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
#define TP_GA_CASE(ctype, name) \
    case DType::name: \
        gather_kernel<ctype, IndexT><<<blocks, kThreads, 0, stream>>>( \
            n, idx_dim_size, idx_inner, self_dim_size, self_inner, \
            self.data_ptr<ctype>(), index.data_ptr<IndexT>(), \
            result.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_GA_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_GA_CASE)
        case DType::ComplexHalf:
            gather_kernel<tensorplay::complex<Half>, IndexT><<<
                blocks, kThreads, 0, stream>>>(
                n, idx_dim_size, idx_inner, self_dim_size, self_inner,
                static_cast<const tensorplay::complex<Half>*>(self.data_ptr()),
                index.data_ptr<IndexT>(),
                static_cast<tensorplay::complex<Half>*>(result.data_ptr()));
            break;
        case DType::ComplexFloat:
            gather_kernel<tensorplay::complex<float>, IndexT><<<
                blocks, kThreads, 0, stream>>>(
                n, idx_dim_size, idx_inner, self_dim_size, self_inner,
                static_cast<const tensorplay::complex<float>*>(self.data_ptr()),
                index.data_ptr<IndexT>(),
                static_cast<tensorplay::complex<float>*>(result.data_ptr()));
            break;
        case DType::ComplexDouble:
            gather_kernel<tensorplay::complex<double>, IndexT><<<
                blocks, kThreads, 0, stream>>>(
                n, idx_dim_size, idx_inner, self_dim_size, self_inner,
                static_cast<const tensorplay::complex<double>*>(self.data_ptr()),
                index.data_ptr<IndexT>(),
                static_cast<tensorplay::complex<double>*>(result.data_ptr()));
            break;
        case DType::BComplex32:
            gather_kernel<tensorplay::complex<BFloat16>, IndexT><<<
                blocks, kThreads, 0, stream>>>(
                n, idx_dim_size, idx_inner, self_dim_size, self_inner,
                static_cast<const tensorplay::complex<BFloat16>*>(self.data_ptr()),
                index.data_ptr<IndexT>(),
                static_cast<tensorplay::complex<BFloat16>*>(result.data_ptr()));
            break;
        default: TP_THROW(TypeError, "gather: unsupported dtype");
    }
#undef TP_GA_CASE
}

__device__ __forceinline__ void atomic_add_rel(int64_t* addr, int64_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(uint8_t* addr, uint8_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(int8_t* addr, int8_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(int16_t* addr, int16_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(uint16_t* addr, uint16_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(uint32_t* addr, uint32_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(uint64_t* addr, uint64_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(int32_t* addr, int32_t v) { gpuAtomicAdd(addr, v); }
__device__ __forceinline__ void atomic_add_rel(float* addr, float v) { gpuAtomicAdd(addr, v); }
__device__ __forceinline__ void atomic_add_rel(double* addr, double v) { gpuAtomicAdd(addr, v); }
__device__ __forceinline__ void atomic_add_rel(Half* addr, Half v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(BFloat16* addr, BFloat16 v) {
    gpuAtomicAdd(addr, v);
}

template <typename T>
__device__ __forceinline__ void atomic_add_rel(tensorplay::complex<T>* addr,
                                               tensorplay::complex<T> v) {
    atomic_add_rel(reinterpret_cast<T*>(addr), v.real());
    atomic_add_rel(reinterpret_cast<T*>(addr) + 1, v.imag());
}

template <typename T>
__device__ __forceinline__ void indexed_atomic_add(T* addr, T v) {
    if constexpr (std::is_same_v<T, bool>) {
        gpuAtomicAdd(addr, v);
    } else {
        atomic_add_rel(addr, v);
    }
}

template <typename T>
inline constexpr bool scatter_add_supported_v =
    std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
    std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
    std::is_same_v<T, int64_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, Half> || std::is_same_v<T, BFloat16> ||
    std::is_same_v<T, bool> ||
    std::is_same_v<T, tensorplay::complex<float>> ||
    std::is_same_v<T, tensorplay::complex<double>> ||
    std::is_same_v<T, tensorplay::complex<Half>> ||
    std::is_same_v<T, tensorplay::complex<BFloat16>>;

template <typename T, bool Add, typename IndexT>
__global__ void scatter_kernel(int64_t total_idx, int64_t idx_dim_size, int64_t idx_inner,
                               int64_t self_dim_size, int64_t self_inner,
                               T* d, const IndexT* ip, const T* vp) {
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; flat < total_idx; flat += stride) {
        int64_t rem = flat;
        int64_t outer_off = rem / (idx_dim_size * idx_inner);
        rem -= outer_off * idx_dim_size * idx_inner;
        int64_t t = rem % idx_inner;
        int64_t idx = ip[flat];
        if (idx < 0) idx += self_dim_size;
        int64_t dst = (outer_off * self_dim_size + idx) * self_inner + t;
        if constexpr (Add) {
            if constexpr (scatter_add_supported_v<T>) {
                indexed_atomic_add(&d[dst], vp[flat]);
            }
        }
        else d[dst] = vp[flat];
    }
}

template <typename IndexT, bool Add>
void launch_scatter_for_index(
        int64_t total_idx, int64_t idx_dim_size, int64_t idx_inner,
        int64_t self_dim_size, int64_t self_inner, Tensor& result,
        const Tensor& index, const Tensor& source, cudaStream_t stream) {
#define TP_SC_CASE(ctype, name) \
    case DType::name: \
        scatter_kernel<ctype, Add, IndexT><<< \
            (total_idx + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total_idx, idx_dim_size, idx_inner, self_dim_size, self_inner, \
            static_cast<ctype*>(result.data_ptr()), index.data_ptr<IndexT>(), \
            static_cast<const ctype*>(source.data_ptr())); \
        break;
    switch (result.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SC_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_SC_CASE)
        TP_SC_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_SC_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_SC_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_SC_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "scatter: unsupported dtype");
    }
#undef TP_SC_CASE
}

template <typename T, typename IndexT>
__global__ void index_add_kernel(int64_t total, int64_t inner, int64_t row,
                                 int64_t n_idx,
                                 T* d, const IndexT* ip, const T* sp) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < total; t += stride) {
        if constexpr (scatter_add_supported_v<T> ||
                      std::is_same_v<T, tensorplay::complex<float>> ||
                      std::is_same_v<T, tensorplay::complex<double>> ||
                      std::is_same_v<T, tensorplay::complex<Half>> ||
                      std::is_same_v<T, tensorplay::complex<BFloat16>>) {
            int64_t source_slice = t / inner;
            int64_t c = t % inner;
            int64_t k = source_slice % n_idx;
            int64_t o = source_slice / n_idx;
            int64_t iv = ip[k];
            if (iv < 0) iv += row;
            indexed_atomic_add(&d[(o * row + iv) * inner + c], sp[t]);
        }
    }
}

template <typename IndexT>
void launch_index_add_for_index(
        int64_t total, int64_t inner, int64_t row, int64_t n_idx,
        const Tensor& result, const Tensor& index, const Tensor& source,
        cudaStream_t stream) {
#define TP_IADD_CASE(ctype, name) \
    case DType::name: \
        index_add_kernel<ctype, IndexT><<< \
            (total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total, inner, row, n_idx, \
            static_cast<ctype*>(result.data_ptr()), \
            index.data_ptr<IndexT>(), \
            static_cast<const ctype*>(source.data_ptr())); \
        break;
    switch (result.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IADD_CASE)
        TP_IADD_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IADD_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IADD_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IADD_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default:
            TP_THROW(NotImplementedError, "index_add on CUDA does not support this dtype");
    }
#undef TP_IADD_CASE
}

template <bool Add>
Tensor scatter_base_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                         const Tensor& src) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as output tensor");
    }
    Tensor idx_c = (index.dtype() == DType::Int64 || index.dtype() == DType::Int32)
        ? index.contiguous() : index.to(DType::Int64).contiguous();
    std::vector<int64_t> idx_shape(static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    if (src.numel() == 1) {
        src_b = src.expand(idx_shape).contiguous();
    } else {
        std::vector<int64_t> bshape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(src.shape()), idx_shape);
        if (bshape != idx_shape) {
            TP_THROW(RuntimeError, "scatter: src shape must broadcast to the index shape");
        }
        src_b = src.expand(idx_shape).contiguous();
    }
    if (src_b.dtype() != self.dtype()) {
        src_b = src_b.to(self.dtype());
    }
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
    int64_t idx_outer = 1;
    for (int64_t i = 0; i < dim; ++i) idx_outer *= idx_c.size(i);
    int64_t idx_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) idx_inner *= idx_c.size(i);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    int64_t idx_dim_size = idx_c.size(dim);
    int64_t total_idx = idx_c.numel();
    int64_t self_dim_size = self.size(dim);
    if (total_idx == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
    if (Add) {
        switch (self.dtype()) {
            case DType::UInt8: case DType::Int8: case DType::Int16:
            case DType::Int32: case DType::Int64:
            case DType::UInt16: case DType::UInt32: case DType::UInt64:
            case DType::Float32: case DType::Float64:
            case DType::Float16: case DType::BFloat16: case DType::Bool:
            case DType::ComplexHalf: case DType::ComplexFloat:
            case DType::ComplexDouble: case DType::BComplex32:
                break;
            default:
                TP_THROW(NotImplementedError,
                         "scatter_add on CUDA does not support this dtype");
        }
    }
    if (idx_c.dtype() == DType::Int32) {
        launch_scatter_for_index<int32_t, Add>(
            total_idx, idx_dim_size, idx_inner, self_dim_size, inner,
            result, idx_c, src_b, stream);
    } else {
        launch_scatter_for_index<int64_t, Add>(
            total_idx, idx_dim_size, idx_inner, self_dim_size, inner,
            result, idx_c, src_b, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

}

Tensor gather_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as input tensor");
    }
    for (int64_t i = 0; i < nd; ++i) {
        if (i != dim && index.size(i) > self.size(i)) {
            TP_THROW(IndexError, "Size does not match at dimension ", i,
                     " (input: ", self.size(i), ", index: ", index.size(i), ")");
        }
    }
    Tensor idx_c = (index.dtype() == DType::Int64 || index.dtype() == DType::Int32)
        ? index.contiguous() : index.to(DType::Int64).contiguous();
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(idx_c.shape()), self.dtype(), self.device());
    int64_t idx_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) idx_inner *= idx_c.size(i);
    int64_t self_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) self_inner *= self.size(i);
    int64_t idx_dim_size = idx_c.size(dim);
    int64_t n = result.numel();
    int64_t self_dim_size = self.size(dim);
    if (n == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
    if (idx_c.dtype() == DType::Int32) {
        launch_gather_kernel_for_index<int32_t>(
            n, idx_dim_size, idx_inner, self_dim_size, self_inner,
            self_c, idx_c, result, stream);
    } else {
        launch_gather_kernel_for_index<int64_t>(
            n, idx_dim_size, idx_inner, self_dim_size, self_inner,
            self_c, idx_c, result, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor scatter_add_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    globalContext().alertNotDeterministic("scatter_add_cuda");
    return scatter_base_cuda<true>(self, dim, index, src);
}
Tensor scatter_src_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_cuda<false>(self, dim, index, src);
}
Tensor scatter_value_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
    Tensor full = Tensor::full({}, value, self.dtype(), self.device());
    return scatter_base_cuda<false>(self, dim, index, full);
}

static Tensor& scatter_base_inplace_cuda(Tensor& self, int64_t dim, const Tensor& index,
                                         const Tensor& src, bool add) {
    if (add) {
        globalContext().alertNotDeterministic("scatter_add_");
    }
    if (!self.is_contiguous()) {
        Tensor out = add ? scatter_base_cuda<true>(self, dim, index, src)
                         : scatter_base_cuda<false>(self, dim, index, src);
        self.copy_(out);
        return self;
    }
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as output tensor");
    }
    Tensor idx_c = (index.dtype() == DType::Int64 || index.dtype() == DType::Int32)
        ? index.contiguous() : index.to(DType::Int64).contiguous();
    std::vector<int64_t> idx_shape(static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    if (src.numel() == 1) {
        src_b = src.expand(idx_shape).contiguous();
    } else {
        std::vector<int64_t> bshape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(src.shape()), idx_shape);
        if (bshape != idx_shape) {
            TP_THROW(RuntimeError, "scatter_: src shape must broadcast to the index shape");
        }
        src_b = src.expand(idx_shape).contiguous();
    }
    if (src_b.dtype() != self.dtype()) {
        src_b = src_b.to(self.dtype());
    }
    Tensor& result = self;
    int64_t idx_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) idx_inner *= idx_c.size(i);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    int64_t idx_dim_size = idx_c.size(dim);
    int64_t total_idx = idx_c.numel();
    int64_t self_dim_size = self.size(dim);
    if (total_idx == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
    if (add) {
        switch (self.dtype()) {
            case DType::UInt8: case DType::Int8: case DType::Int16:
            case DType::Int32: case DType::Int64:
            case DType::UInt16: case DType::UInt32: case DType::UInt64:
            case DType::Float32: case DType::Float64:
            case DType::Float16: case DType::BFloat16: case DType::Bool:
            case DType::ComplexHalf: case DType::ComplexFloat:
            case DType::ComplexDouble: case DType::BComplex32:
                break;
            default:
                TP_THROW(NotImplementedError,
                         "scatter_add_ on CUDA does not support this dtype");
        }
        if (idx_c.dtype() == DType::Int32) {
            launch_scatter_for_index<int32_t, true>(
                total_idx, idx_dim_size, idx_inner, self_dim_size, inner,
                result, idx_c, src_b, stream);
        } else {
            launch_scatter_for_index<int64_t, true>(
                total_idx, idx_dim_size, idx_inner, self_dim_size, inner,
                result, idx_c, src_b, stream);
        }
    } else {
        if (idx_c.dtype() == DType::Int32) {
            launch_scatter_for_index<int32_t, false>(
                total_idx, idx_dim_size, idx_inner, self_dim_size, inner,
                result, idx_c, src_b, stream);
        } else {
            launch_scatter_for_index<int64_t, false>(
                total_idx, idx_dim_size, idx_inner, self_dim_size, inner,
                result, idx_c, src_b, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor& scatter_inplace_src_cuda(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_inplace_cuda(self, dim, index, src, false);
}

Tensor& scatter_inplace_value_cuda(Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
    Tensor full = Tensor::full({}, value, self.dtype(), self.device());
    return scatter_base_inplace_cuda(self, dim, index, full, false);
}

Tensor& scatter_add_inplace_cuda(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_inplace_cuda(self, dim, index, src, true);
}

Tensor index_add_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64 || index.dtype() == DType::Int32)
        ? index.contiguous() : index.to(DType::Int64).contiguous();
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return result;
    int64_t row = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    Tensor source_c = source.contiguous();
    if (source_c.dim() != nd) {
        TP_THROW(RuntimeError, "index_add: source must have same number of dims as input");
    }
    if (source_c.size(dim) != n_idx) {
        TP_THROW(RuntimeError, "index_add: source size along dim must equal index length");
    }
    int64_t total = outer * n_idx * inner;
    auto stream = getCurrentCUDAStream().stream();
    if (idx.dtype() == DType::Int32) {
        launch_index_add_for_index<int32_t>(
            total, inner, row, n_idx, result, idx, source_c, stream);
    } else {
        launch_index_add_for_index<int64_t>(
            total, inner, row, n_idx, result, idx, source_c, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, GatherScatterKernels) {
    m.impl("gather", gather_cuda);
    m.impl("scatter_add", scatter_add_cuda);
    m.impl("scatter.src", scatter_src_cuda);
    m.impl("scatter.value", scatter_value_cuda);
    m.impl("scatter_.src", scatter_inplace_src_cuda);
    m.impl("scatter_.value", scatter_inplace_value_cuda);
    m.impl("scatter_add_", scatter_add_inplace_cuda);
    m.impl("index_add", index_add_cuda);
}

}
}
