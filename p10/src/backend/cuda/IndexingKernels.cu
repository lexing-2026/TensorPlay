// High-throughput indexing and masking kernels.
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
#include "CUDALoops.cuh"

// Narrow floating-point atomics operate on the containing 32-bit word and
// replace the selected half with an atomic compare-and-swap. The overloads
// stay at global scope so qualified scalar types resolve without ambiguity.
#include "Atomic.cuh"

#include <cassert>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>

// Index kernels validate their index values on the device itself: an
// out-of-range value faults the launch instead of reading or writing out of
// bounds.  The check is deliberately active in release builds, so invalid
// indexing input cannot corrupt memory silently.  Negative values wrap into
// range afterwards, matching advanced-indexing semantics.
#define TP_INDEX_RANGE_GUARD(iv, row)                 \
    if ((iv) < -(row) || (iv) >= (row)) { __trap(); } \
    if ((iv) < 0) { (iv) += (row); }
#include <string>
#include <type_traits>
#include "OutWrite.h"

namespace {
inline std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                             const std::vector<int64_t>& b) {
    // Broadcast dimensions from the trailing axis; size-one axes stretch.
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t x = i < a.size() ? a[a.size() - 1 - i] : 1;
        const int64_t y = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (x != y && x != 1 && y != 1) {
            TP_THROW(RuntimeError,
                     "The size of tensor a (", x,
                     ") must match the size of tensor b (", y,
                     ") at non-singleton dimension ", rank - 1 - i);
        }
        out[rank - 1 - i] = std::max(x, y);
    }
    return out;
}
} // namespace

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

// Keep-predicate for lower and upper triangular masks.
template <typename T, bool Lower>
__global__ void triangular_mask_kernel(int64_t batch_rows, int64_t rows, int64_t cols,
                                       const T* in, T* out, int64_t diagonal) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < batch_rows; t += stride) {
        int64_t bi = t / rows, r = t % rows;
        const T* sp = in + bi * rows * cols + r * cols;
        T* dp = out + bi * rows * cols + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            bool keep = Lower ? (c <= r + diagonal) : (c >= r + diagonal);
            dp[c] = keep ? sp[c] : static_cast<T>(0);
        }
    }
}

// Gather with separate result and source trailing extents.
template <typename T, typename IndexT>
__global__ void gather_kernel(int64_t n, int64_t idx_dim_size, int64_t idx_inner,
                              int64_t self_dim_size, int64_t self_inner,
                              const T* s, const IndexT* ip, T* d) {
    // The result follows the index shape; source and index trailing extents
    // can differ on axes other than the selected one.
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; flat < n; flat += stride) {
        int64_t rem = flat;
        int64_t outer_off = rem / (idx_dim_size * idx_inner); rem -= outer_off * idx_dim_size * idx_inner;
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

// Boolean accumulation is logical OR, realized through the byte CAS loop in
// the integer-family atomic.
__device__ __forceinline__ void atomic_add_rel(bool* addr, bool v) {
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

// Scatter and scatter-add use elementwise indexed writes. Add mode uses
// atomic accumulation and is intentionally unordered for colliding indices.
template <typename T, bool Add, typename IndexT>
__global__ void scatter_kernel(int64_t total_idx, int64_t idx_dim_size, int64_t idx_inner,
                               int64_t self_dim_size, int64_t self_inner,
                               T* d, const IndexT* ip, const T* vp) {
    // One thread handles one indexed element. Colliding additions serialize
    // through atomics.
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
    // One thread per (source position, inner column): adds sv into the
    // selected destination slice.
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

// Index-select row gather.
template <typename T, typename IndexT>
__global__ void index_select_kernel(int64_t total_out_elems, int64_t n_idx, int64_t inner,
                                    int64_t row, const T* s, const IndexT* ip, T* d) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < total_out_elems; i += stride) {
        int64_t t = i / inner;      // (o * n_idx + k)
        int64_t c = i % inner;
        int64_t k = t % n_idx;
        int64_t iv = ip[k];
        TP_INDEX_RANGE_GUARD(iv, row);
        d[i] = s[(t / n_idx * row + iv) * inner + c];
    }
}

template <typename T, typename IndexT>
__global__ void index_select_slice_kernel(int64_t n_slices, int64_t n_idx,
                                          int64_t inner, int64_t row,
                                          const T* s, const IndexT* ip, T* d) {
    const int64_t slice_stride = static_cast<int64_t>(gridDim.x);
    for (int64_t slice = static_cast<int64_t>(blockIdx.x); slice < n_slices;
         slice += slice_stride) {
        const int64_t outer_index = slice / n_idx;
        const int64_t index_position = slice % n_idx;
        int64_t source_index = ip[index_position];
        TP_INDEX_RANGE_GUARD(source_index, row);
        const T* source = s + (outer_index * row + source_index) * inner;
        T* destination = d + slice * inner;
        for (int64_t c = threadIdx.x; c < inner; c += blockDim.x) {
            destination[c] = source[c];
        }
    }
}

template <typename IndexT>
void launch_index_select_for_index(
        int64_t total, int64_t n_idx, int64_t inner, int64_t row,
        int64_t outer, const Tensor& self, const Tensor& index, Tensor& result,
        int slice_threads, cudaStream_t stream) {
#define TP_IS_CASE(ctype, name) \
    case DType::name: { \
        if (inner >= 64) { \
            const int64_t slices = outer * n_idx; \
            const int64_t blocks = std::min<int64_t>(slices, 4096); \
            index_select_slice_kernel<ctype, IndexT><<< \
                static_cast<unsigned>(blocks), slice_threads, 0, stream>>>( \
                slices, n_idx, inner, row, \
                static_cast<const ctype*>(self.data_ptr()), \
                index.data_ptr<IndexT>(), static_cast<ctype*>(result.data_ptr())); \
        } else { \
            index_select_kernel<ctype, IndexT><<< \
                (total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                total, n_idx, inner, row, \
                static_cast<const ctype*>(self.data_ptr()), \
                index.data_ptr<IndexT>(), static_cast<ctype*>(result.data_ptr())); \
        } \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IS_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_IS_CASE)
        TP_IS_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IS_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IS_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IS_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "index_select: unsupported dtype");
    }
#undef TP_IS_CASE
}

template <typename T>
__global__ void index_copy_kernel(int64_t n_idx_x_inner, int64_t inner, int64_t row,
                                  T* d, const int64_t* ip, const T* sp) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < n_idx_x_inner; t += stride) {
        int64_t k = t / inner, c = t % inner;
        int64_t iv = ip[k];
        TP_INDEX_RANGE_GUARD(iv, row);
        d[iv * inner + c] = sp[t];
    }
}

template <typename T>
__global__ void index_fill_kernel(int64_t total, int64_t inner, int64_t row,
                                  T* d, const int64_t* ip, T v) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < total; t += stride) {
        int64_t k = t / inner, c = t % inner;
        int64_t iv = ip[k];
        TP_INDEX_RANGE_GUARD(iv, row);
        d[iv * inner + c] = v;
    }
}

template <typename T>
inline void run_nonzero_mark_iter(TensorIteratorBase& iter) {
    gpu_kernel(iter, [] __host__ __device__(T value) -> int64_t {
        return static_cast<bool>(value != T(0)) ? int64_t(1) : int64_t(0);
    });
}

template <typename T>
__global__ void nonzero_fill_kernel(int64_t n, int64_t ndim, const T* x,
                                    const int64_t* sizes, const int64_t* positions,
                                    int64_t* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (!(x[i] != T(0))) continue;
        const int64_t slot = positions[i] - 1;
        int64_t rem = i;
        for (int64_t d2 = ndim - 1; d2 >= 0; --d2) {
            out[slot * ndim + d2] = rem % sizes[d2];
            rem /= sizes[d2];
        }
    }
}

}

// ---------------------------------------------------------------------------
// masked_fill / masked_fill_
// ---------------------------------------------------------------------------

template <typename T>
inline void run_masked_fill_iter(TensorIteratorBase& iter, T value) {
    gpu_kernel(iter, [value] __host__ __device__(T self_value, bool mask_value) -> T {
        return mask_value ? value : self_value;
    });
}

inline void dispatch_masked_fill_iter(TensorIteratorBase& iter,
                                      DType dtype, const Scalar& value) {
#define TP_MF_ITER_CASE(ctype, name) \
    case DType::name: \
        run_masked_fill_iter<ctype>(iter, value.to<ctype>()); \
        break;
    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MF_ITER_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_MF_ITER_CASE)
        case DType::ComplexHalf:
            run_masked_fill_iter<tensorplay::complex<Half>>(
                iter, value.to<tensorplay::complex<Half>>());
            break;
        case DType::ComplexFloat:
            run_masked_fill_iter<tensorplay::complex<float>>(
                iter, value.to<tensorplay::complex<float>>());
            break;
        case DType::ComplexDouble:
            run_masked_fill_iter<tensorplay::complex<double>>(
                iter, value.to<tensorplay::complex<double>>());
            break;
        case DType::BComplex32:
            run_masked_fill_iter<tensorplay::complex<BFloat16>>(
                iter, value.to<tensorplay::complex<BFloat16>>());
            break;
        default: TP_THROW(TypeError, "masked_fill: unsupported dtype");
    }
#undef TP_MF_ITER_CASE
}

Tensor masked_fill_cuda(const Tensor& self, const Tensor& mask, const Scalar& value) {
    // Broadcast the mask and source once, then apply the replacement in one pass.
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_fill only supports boolean masks");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(mask.shape()));
    Tensor result = Tensor::empty(out_shape, self.dtype(), self.device());
    TensorIterator iter = TensorIteratorConfig()
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .add_output(result)
        .add_const_input(self)
        .add_const_input(mask)
        .build();
    dispatch_masked_fill_iter(iter, self.dtype(), value);
    return result;
}

Tensor& masked_fill__cuda(Tensor& self, const Tensor& mask, const Scalar& value) {
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_fill only supports boolean masks");
    }
    TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .add_output(self)
        .add_const_input(self)
        .add_const_input(mask)
        .build();
    dispatch_masked_fill_iter(iter, self.dtype(), value);
    return self;
}

Tensor& masked_fill_tensor__cuda(Tensor& self, const Tensor& mask, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "masked_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    return masked_fill__cuda(self, mask, value.item());
}

Tensor masked_fill_tensor_cuda(const Tensor& self, const Tensor& mask, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "masked_fill only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    return masked_fill_cuda(self, mask, value.item());
}

// ---------------------------------------------------------------------------
// tril / triu.
// ---------------------------------------------------------------------------

Tensor tril_cuda(const Tensor& self, int64_t diagonal);
Tensor triu_cuda(const Tensor& self, int64_t diagonal);

namespace {
template <bool Lower>
Tensor triangular_mask_entry(const Tensor& self, int64_t diagonal) {
    int64_t ndim = self.dim();
    if (ndim < 2) TP_THROW(RuntimeError, "tril/triu requires tensor with at least 2 dimensions");
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t rows = self.size(ndim - 2);
    int64_t cols = self.size(ndim - 1);
    int64_t batch = self.numel() / (rows * cols);
    if (batch == 0 || rows == 0 || cols == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
    int64_t work = batch * rows;
#define TP_TRI_CASE(ctype, name) \
    case DType::name: \
        triangular_mask_kernel<ctype, Lower><<<(work + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            work, rows, cols, self_c.data_ptr<ctype>(), result.data_ptr<ctype>(), diagonal); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_TRI_CASE)
        default: TP_THROW(TypeError, "tril/triu: unsupported dtype");
    }
#undef TP_TRI_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}
} // anonymous namespace

Tensor tril_cuda(const Tensor& self, int64_t diagonal) { return triangular_mask_entry<true>(self, diagonal); }
Tensor triu_cuda(const Tensor& self, int64_t diagonal) { return triangular_mask_entry<false>(self, diagonal); }

// ---------------------------------------------------------------------------
// gather.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// scatter / scatter_add.
// ---------------------------------------------------------------------------

namespace {
enum class ScatterMode { Assign, Add };

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
} // anonymous namespace

Tensor scatter_add_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
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

// scatter, written directly into self instead of a clone.  Non-contiguous self
// falls back to the out-of-place
// kernel plus a copy back so any layout works.
static Tensor& scatter_base_inplace_cuda(Tensor& self, int64_t dim, const Tensor& index,
                                         const Tensor& src, bool add) {
    if (add) {
        // Accumulates with atomicAdd (no deterministic variant implemented).
        globalContext().alertNotDeterministic("scatter_add_");
    }
    if (!self.is_contiguous()) {
        // scatter_base_cuda<Add> already starts from a clone of self, so its
        // result is exactly what scatter_/scatter_add_ should leave in self.
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
    return scatter_base_inplace_cuda(self, dim, index, src, /*add=*/false);
}

Tensor& scatter_inplace_value_cuda(Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
    Tensor full = Tensor::full({}, value, self.dtype(), self.device());
    return scatter_base_inplace_cuda(self, dim, index, full, /*add=*/false);
}

Tensor& scatter_add_inplace_cuda(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_inplace_cuda(self, dim, index, src, /*add=*/true);
}

// ---------------------------------------------------------------------------
// index_select.
// ---------------------------------------------------------------------------

Tensor index_select_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != 1) TP_THROW(IndexError, "index_select(): index should be a vector");
    Tensor idx = (index.dtype() == DType::Int64 || index.dtype() == DType::Int32)
        ? index.contiguous() : index.to(DType::Int64).contiguous();
    int64_t n_idx = idx.numel();
    int64_t row = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape(static_cast<std::vector<int64_t>>(self.shape()));
    out_shape[dim] = n_idx;
    Tensor result = Tensor::empty(out_shape, self.dtype(), self.device());
    int64_t total = result.numel();
    if (total == 0) return result;
    Tensor self_c = self.contiguous();
    auto stream = getCurrentCUDAStream().stream();
    const int slice_threads = inner >= 1024 ? 512 : kThreads;
    if (idx.dtype() == DType::Int32) {
        launch_index_select_for_index<int32_t>(
            total, n_idx, inner, row, outer, self_c, idx, result,
            slice_threads, stream);
    } else {
        launch_index_select_for_index<int64_t>(
            total, n_idx, inner, row, outer, self_c, idx, result,
            slice_threads, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ---------------------------------------------------------------------------
// index_add with atomic accumulation.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// index_copy / index_fill.
// ---------------------------------------------------------------------------

Tensor index_copy_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return result;
    int64_t row = self.size(dim);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    int64_t total = n_idx * inner;
    Tensor source_c = source.contiguous();
    auto stream = getCurrentCUDAStream().stream();
#define TP_IC_CASE(ctype, name) \
    case DType::name: \
        index_copy_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total, inner, row, static_cast<ctype*>(result.data_ptr()), \
            idx.data_ptr<int64_t>(), static_cast<const ctype*>(source_c.data_ptr())); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IC_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_IC_CASE)
        TP_IC_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IC_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IC_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IC_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "index_copy: unsupported dtype");
    }
#undef TP_IC_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor index_fill_scalar_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value);

Tensor index_fill_tensor_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "index_fill only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    Scalar v = value.item();
    return index_fill_scalar_cuda(self, dim, index, v);
}

Tensor index_fill_scalar_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return result;
    int64_t row = self.size(dim);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    int64_t total = n_idx * inner;
    auto stream = getCurrentCUDAStream().stream();
#define TP_IF_CASE(ctype, name) \
    case DType::name: { \
        ctype v = value.to<ctype>(); \
        index_fill_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total, inner, row, static_cast<ctype*>(result.data_ptr()), \
            idx.data_ptr<int64_t>(), v); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IF_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_IF_CASE)
        TP_IF_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IF_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IF_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IF_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "index_fill: unsupported dtype");
    }
#undef TP_IF_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor& index_fill_scalar__cuda(Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
    // Fill a clone, then copy it back through the existing in-place path.
    self.copy_(index_fill_scalar_cuda(self, dim, index, value));
    return self;
}

Tensor& index_fill_tensor__cuda(Tensor& self, int64_t dim, const Tensor& index, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "index_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    return index_fill_scalar__cuda(self, dim, index, value.item());
}

// ---------------------------------------------------------------------------
// nonzero (device flag/prefix pass followed by an ordered coordinate pass).
// ---------------------------------------------------------------------------

Tensor nonzero_cuda(const Tensor& self) {
    Tensor self_c = self.contiguous();
    int64_t nd = self.dim();
    int64_t n = self_c.numel();
    // Empty input: no matches. Launching with a 0-block grid is a CUDA error,
    if (n == 0) {
        return Tensor::zeros({0, nd}, DType::Int64, self.device());
    }
    TP_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int>::max()),
             "nonzero: input is too large for device scan");
    Tensor flags = Tensor::empty({n}, DType::Int64, self.device());
    Tensor self_flat = self_c.reshape({n});
    TensorIterator flag_iter = TensorIteratorConfig()
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .add_output(flags)
        .add_const_input(self_flat)
        .build();
#define TP_NZC_CASE(ctype, name) \
    case DType::name: \
        run_nonzero_mark_iter<ctype>(flag_iter); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NZC_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_NZC_CASE)
        case DType::ComplexHalf:
            run_nonzero_mark_iter<tensorplay::complex<Half>>(flag_iter);
            break;
        case DType::ComplexFloat:
            run_nonzero_mark_iter<tensorplay::complex<float>>(flag_iter);
            break;
        case DType::ComplexDouble:
            run_nonzero_mark_iter<tensorplay::complex<double>>(flag_iter);
            break;
        case DType::BComplex32:
            run_nonzero_mark_iter<tensorplay::complex<BFloat16>>(flag_iter);
            break;
        default: TP_THROW(TypeError, "nonzero: unsupported dtype");
    }
#undef TP_NZC_CASE
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    Tensor positions = Tensor::empty({n}, DType::Int64, self.device());
    size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_bytes, flags.data_ptr<int64_t>(),
        positions.data_ptr<int64_t>(), static_cast<int>(n), stream));
    Tensor scan_storage = Tensor::empty(
        {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)},
        DType::UInt8, self.device());
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scan_storage.data_ptr(), scan_bytes, flags.data_ptr<int64_t>(),
        positions.data_ptr<int64_t>(), static_cast<int>(n), stream));
    int64_t count_host = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &count_host, positions.data_ptr<int64_t>() + n - 1, sizeof(int64_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Tensor result = Tensor::zeros({count_host, nd}, DType::Int64, self.device());
    if (count_host == 0) return result;
    // sizes live on the host; stage them on-device for the fill kernel
    std::vector<int64_t> h_sizes(static_cast<std::vector<int64_t>>(self_c.shape()));
    Tensor sizes_d = Tensor::empty({nd}, DType::Int64, self.device());
    CUDA_CHECK(cudaMemcpy(sizes_d.data_ptr<int64_t>(), h_sizes.data(), nd * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
#define TP_NZF_CASE(ctype, name) \
    case DType::name: \
        nonzero_fill_kernel<ctype><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            n, nd, self_c.data_ptr<ctype>(), sizes_d.data_ptr<int64_t>(), \
            positions.data_ptr<int64_t>(), result.data_ptr<int64_t>()); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NZF_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_NZF_CASE)
        case DType::ComplexHalf:
            nonzero_fill_kernel<tensorplay::complex<Half>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, nd, static_cast<const tensorplay::complex<Half>*>(self_c.data_ptr()),
                sizes_d.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
                result.data_ptr<int64_t>());
            break;
        case DType::ComplexFloat:
            nonzero_fill_kernel<tensorplay::complex<float>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, nd, static_cast<const tensorplay::complex<float>*>(self_c.data_ptr()),
                sizes_d.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
                result.data_ptr<int64_t>());
            break;
        case DType::ComplexDouble:
            nonzero_fill_kernel<tensorplay::complex<double>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, nd, static_cast<const tensorplay::complex<double>*>(self_c.data_ptr()),
                sizes_d.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
                result.data_ptr<int64_t>());
            break;
        case DType::BComplex32:
            nonzero_fill_kernel<tensorplay::complex<BFloat16>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, nd, static_cast<const tensorplay::complex<BFloat16>*>(self_c.data_ptr()),
                sizes_d.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
                result.data_ptr<int64_t>());
            break;
        default: TP_THROW(TypeError, "nonzero: unsupported dtype");
    }
#undef TP_NZF_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ---------------------------------------------------------------------------
// take (reshape -> index_select -> reshape).
// ---------------------------------------------------------------------------

Tensor take_cuda(const Tensor& self, const Tensor& index) {
    Tensor flat = self.reshape({self.numel()});
    return index_select_cuda(flat, 0, index.reshape({index.numel()}))
        .reshape(static_cast<std::vector<int64_t>>(index.shape()));
}

// ---------------------------------------------------------------------------
// masked_scatter (source values are consumed in mask order).
// ---------------------------------------------------------------------------

template <typename T>
inline void run_masked_scatter_iter(TensorIteratorBase& iter,
                                    const T* source) {
    gpu_kernel(iter, [source] __host__ __device__(
        T self_value, bool mask_value, int64_t source_offset) -> T {
        return mask_value ? source[source_offset] : self_value;
    });
}

__global__ void masked_scatter_size_check(
    const int64_t* source_offsets, const bool* mask, int64_t last,
    int64_t source_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int64_t selected =
            source_offsets[last] + (mask[last] ? int64_t(1) : int64_t(0));
        assert(selected <= source_size);
    }
}

Tensor masked_scatter_cuda(const Tensor& self, const Tensor& mask, const Tensor& source) {
    if (source.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "masked_scatter: self and source must have the same dtype");
    }
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_scatter: mask must be bool");
    }
    if (mask.device() != self.device() || source.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "masked_scatter: self, mask, and source must be on the same device");
    }

    Tensor self_iter = self.dim() == 0 ? self.unsqueeze(0) : self;
    Tensor mask_temp = mask.dim() == 0 ? mask.unsqueeze(0) : mask;
    Tensor m_full = mask_temp.expand(
        static_cast<std::vector<int64_t>>(self_iter.shape())).contiguous();
    Tensor src = source.contiguous();
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
    Tensor result_iter = result.dim() == 0 ? result.unsqueeze(0) : result;
    const int64_t n = result_iter.numel();
    if (n == 0) return result;

    TP_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int>::max()),
             "masked_scatter: input is too large for device scan");
    Tensor mask_flat = m_full.reshape({n});
    Tensor flags = Tensor::empty({n}, DType::Int64, self.device());
    Tensor source_offsets = Tensor::empty({n}, DType::Int64, self.device());
    TensorIterator flag_iter = TensorIteratorConfig()
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .add_output(flags)
        .add_const_input(mask_flat)
        .build();
    gpu_kernel(flag_iter, [] __host__ __device__(bool value) -> int64_t {
        return value ? int64_t(1) : int64_t(0);
    });

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, flags.data_ptr<int64_t>(),
        source_offsets.data_ptr<int64_t>(), static_cast<int>(n), stream));
    Tensor scan_storage = Tensor::empty(
        {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)},
        DType::UInt8, self.device());
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_storage.data_ptr(), scan_bytes, flags.data_ptr<int64_t>(),
        source_offsets.data_ptr<int64_t>(), static_cast<int>(n), stream));

    masked_scatter_size_check<<<1, 1, 0, stream>>>(
        source_offsets.data_ptr<int64_t>(), m_full.data_ptr<bool>(), n - 1,
        src.numel());
    CUDA_CHECK(cudaGetLastError());

    Tensor source_offsets_view = source_offsets.reshape(
        static_cast<std::vector<int64_t>>(result_iter.shape()));
    TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(result_iter)
        .add_input(result_iter)
        .add_const_input(m_full)
        .add_input(source_offsets_view)
        .build();

#define TP_MS_CASE(ctype, name) \
    case DType::name: \
        run_masked_scatter_iter<ctype>(iter, src.data_ptr<ctype>()); \
        break;
    switch (result.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MS_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_MS_CASE)
        case DType::ComplexHalf:
            run_masked_scatter_iter<tensorplay::complex<Half>>(
                iter, static_cast<const tensorplay::complex<Half>*>(src.data_ptr()));
            break;
        case DType::ComplexFloat:
            run_masked_scatter_iter<tensorplay::complex<float>>(
                iter, static_cast<const tensorplay::complex<float>*>(src.data_ptr()));
            break;
        case DType::ComplexDouble:
            run_masked_scatter_iter<tensorplay::complex<double>>(
                iter, static_cast<const tensorplay::complex<double>*>(src.data_ptr()));
            break;
        case DType::BComplex32:
            run_masked_scatter_iter<tensorplay::complex<BFloat16>>(
                iter, static_cast<const tensorplay::complex<BFloat16>*>(src.data_ptr()));
            break;
        default: TP_THROW(TypeError, "masked_scatter: unsupported dtype");
    }
#undef TP_MS_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}


Tensor scatter_reduce_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                           const Tensor& src, const std::string& reduce,
                           bool include_self);
Tensor index_reduce_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                         const Tensor& source, const std::string& reduce,
                         bool include_self);
Tensor scatter_reduce_backward_self_cuda(const Tensor& grad,
                                         const Tensor& self, int64_t dim,
                                         const Tensor& index,
                                         const Tensor& src,
                                         const std::string& reduce,
                                         bool include_self);
Tensor scatter_reduce_backward_src_cuda(const Tensor& grad,
                                        const Tensor& self, int64_t dim,
                                        const Tensor& index,
                                        const Tensor& src,
                                        const std::string& reduce,
                                        bool include_self);
Tensor index_reduce_backward_self_cuda(const Tensor& grad,
                                       const Tensor& self, int64_t dim,
                                       const Tensor& index,
                                       const Tensor& source,
                                       const std::string& reduce,
                                       bool include_self);
Tensor index_reduce_backward_src_cuda(const Tensor& grad,
                                      const Tensor& self, int64_t dim,
                                      const Tensor& index,
                                      const Tensor& source,
                                      const std::string& reduce,
                                      bool include_self);
Tensor& interop_tril_out_cuda(const Tensor& self, int64_t diagonal, Tensor& out) {
        write_out(out, tril_cuda(self, diagonal));
        return out;

}

Tensor& interop_triu_out_cuda(const Tensor& self, int64_t diagonal, Tensor& out) {
        write_out(out, triu_cuda(self, diagonal));
        return out;

}


Tensor& interop_masked_fill__Scalar_cuda(Tensor& self, const Tensor& mask, const Scalar& value) {
        return masked_fill__cuda(self, mask, value);
    
}

TENSORPLAY_LIBRARY_IMPL(CUDA, IndexingKernels) {
    m.impl("masked_fill", masked_fill_cuda);
    m.impl("masked_fill_", masked_fill__cuda);
    m.impl("masked_fill.Tensor", masked_fill_tensor_cuda);
    m.impl("masked_fill_.Tensor", masked_fill_tensor__cuda);
    m.impl("tril", tril_cuda);
    m.impl("triu", triu_cuda);
    m.impl("gather", gather_cuda);
    m.impl("scatter_add", scatter_add_cuda);
    m.impl("scatter.src", scatter_src_cuda);
    m.impl("scatter.value", scatter_value_cuda);
    m.impl("scatter_.src", scatter_inplace_src_cuda);
    m.impl("scatter_.value", scatter_inplace_value_cuda);
    m.impl("scatter_add_", scatter_add_inplace_cuda);
    m.impl("index_select", index_select_cuda);
    m.impl("index_add", index_add_cuda);
    m.impl("index_copy", index_copy_cuda);
    m.impl("index_fill.Tensor", index_fill_tensor_cuda);
    m.impl("index_fill.Scalar", index_fill_scalar_cuda);
    m.impl("index_fill_.Tensor", index_fill_tensor__cuda);
    m.impl("index_fill_.Scalar", index_fill_scalar__cuda);
    m.impl("nonzero", nonzero_cuda);
    m.impl("take", take_cuda);
    m.impl("masked_scatter", masked_scatter_cuda);

    // out-variants: run the value kernel, then transfer into the caller's
    // buffer.  masked_fill_.Scalar routes through the tensor-overload kernel.
    m.impl("tril.out", interop_tril_out_cuda);
    m.impl("triu.out", interop_triu_out_cuda);
    m.impl("masked_fill_.Scalar", interop_masked_fill__Scalar_cuda);
}

} // namespace cuda

} // namespace tensorplay
