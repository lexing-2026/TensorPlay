// High-throughput indexing, masking, and scan kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include "GPUPrimitives.cuh"
#include "SortingRadixSelect.cuh"
#include "SortUtils.cuh"
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
#include <tuple>
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

// Per-slice in-place heapsort carrying original positions. Global-memory
// storage supports arbitrary slice sizes without shared-memory limits while
// preserving the stable-order contract. Kept as the fallback for shapes
// beyond the radix path's limits (slice count > 2^21 or numel > INT_MAX).
template <typename T>
__global__ void sort_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                            bool descending, const T* in, T* vals, int64_t* idxs) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        T* vb = vals + o * d_size * inner + in2;
        int64_t* ib = idxs + o * d_size * inner + in2;
        for (int64_t j = 0; j < d_size; ++j) { vb[j * inner] = sp[j * inner]; ib[j * inner] = j; }
        // build heap on (value, index) pairs
        auto less = [&](int64_t a, int64_t b) {
            T va = vb[a * inner], vbv = vb[b * inner];
            bool lt = va < vbv, gt = va > vbv;
            // Ascending needs a MAX-heap (largest at root, extracted to the
            // tail), i.e. the *larger* element must sink on sift_down.
            return descending ? gt : lt;
        };
        auto swap_pair = [&](int64_t a, int64_t b) {
            T tv = vb[a * inner]; vb[a * inner] = vb[b * inner]; vb[b * inner] = tv;
            int64_t ti = ib[a * inner]; ib[a * inner] = ib[b * inner]; ib[b * inner] = ti;
        };
        auto sift_down = [&](int64_t start, int64_t end) {
            int64_t root = start;
            while (2 * root + 1 <= end) {
                int64_t child = 2 * root + 1;
                if (child + 1 <= end && less(child, child + 1)) child++;
                if (less(root, child)) { swap_pair(root, child); root = child; }
                else break;
            }
        };
        for (int64_t st = d_size / 2 - 1; st >= 0; --st) sift_down(st, d_size - 1);
        for (int64_t end = d_size - 1; end > 0; --end) {
            swap_pair(0, end);
            sift_down(0, end - 1);
        }
    }
}

// Radix-sort path: each element is packed into a contiguous (sortable key,
// position) pair, one segmented radix pass orders every slice, then results
// are scattered back to the strided output layout. Radix ordering is stable,
// so equal keys keep their original relative order in both directions.
// Encodings reuse the topk bit-twiddling traits; bool gets a trivial one.
template <typename T>
struct SortRadixTraits : topk_detail::TopKRadixTraits<T> {};

// bool has no topk trait: a single-bit key suffices (false < true).
template <>
struct SortRadixTraits<bool> {
    using key_type = uint32_t;
    static constexpr int bit_count = 1;
    __device__ static inline key_type encode(bool value) { return value ? 1u : 0u; }
    __device__ static inline bool deconvert(key_type value) { return value != 0u; }
};

// Floating encodings fold negative zero onto positive zero before the sign
// flip: the two zero bit patterns compare equal (stable order), not as
// distinct magnitudes.
template <>
struct SortRadixTraits<float> : topk_detail::TopKRadixTraits<float> {
    __device__ static inline key_type encode(float value) {
        uint32_t bits = static_cast<uint32_t>(__float_as_int(value));
        if ((bits & 0x7fffffffu) == 0u) bits = 0u;
        const uint32_t mask = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
        return value == value ? static_cast<uint32_t>(bits ^ mask) : 0xffffffffu;
    }
};

template <>
struct SortRadixTraits<double> : topk_detail::TopKRadixTraits<double> {
    __device__ static inline key_type encode(double value) {
        uint64_t bits = static_cast<uint64_t>(__double_as_longlong(value));
        if ((bits & 0x7fffffffffffffffULL) == 0ULL) bits = 0ULL;
        const uint64_t mask = (bits >> 63) ? 0xffffffffffffffffULL
                                           : 0x8000000000000000ULL;
        return value == value ? static_cast<uint64_t>(bits ^ mask)
                              : 0xffffffffffffffffULL;
    }
};

template <>
struct SortRadixTraits<Half> : topk_detail::TopKRadixTraits<Half> {
    __device__ static inline key_type encode(Half value) {
        uint16_t bits = static_cast<uint16_t>(value.x);
        if ((bits & 0x7fffu) == 0u) bits = 0u;
        const uint16_t mask = (bits & 0x8000u) ? 0xffffu : 0x8000u;
        const float converted = static_cast<float>(value);
        return converted == converted ? static_cast<uint32_t>(bits ^ mask)
                                      : 0xffffu;
    }
};

template <>
struct SortRadixTraits<BFloat16> : topk_detail::TopKRadixTraits<BFloat16> {
    __device__ static inline key_type encode(BFloat16 value) {
        uint16_t bits = static_cast<uint16_t>(value.x);
        if ((bits & 0x7fffu) == 0u) bits = 0u;
        const uint16_t mask = (bits & 0x8000u) ? 0xffffu : 0x8000u;
        const float converted = static_cast<float>(value);
        return converted == converted ? static_cast<uint32_t>(bits ^ mask)
                                      : 0xffffu;
    }
};

// Encoded-key comparators for the warp merge path: plain less/greater over
// the radix key, so the ordering matches the radix path exactly.
struct SortKeyLessOp {
    template <typename K>
    __device__ __forceinline__ bool operator()(K a, K b) const { return a < b; }
};
struct SortKeyGreaterOp {
    template <typename K>
    __device__ __forceinline__ bool operator()(K a, K b) const { return a > b; }
};

// ---------------------------------------------------------------------------
// Warp-per-slice merge sort for short slices.  One warp owns one slice and
// several warps share a block, so short rows run at wave occupancy instead
// of one full block per row; the collective load/store transpose keeps the
// global passes coalesced.  Ordering follows the encoded-key semantics of
// the radix path: keys that compare after every valid value (or NaN) land
// at the end, and stable ordering preserves the input order of ties.
// ---------------------------------------------------------------------------
template <typename T, int SortSize>
__global__ void sort_warp_merge_kernel(
    const T* __restrict__ in, T* __restrict__ vals, int64_t* __restrict__ idxs,
    int64_t slices, int64_t d_size, int64_t inner, bool descending) {
    using Key = typename SortRadixTraits<T>::key_type;
    constexpr int kWarpThreads = 32;
    constexpr int kItemsPerThread = SortSize / kWarpThreads;
    constexpr int kMaxBlockWarps = 16;
    using LoadValues = cub::WarpLoad<
        T, kItemsPerThread, cub::WARP_LOAD_TRANSPOSE>;
    using Sort = cub::WarpMergeSort<
        Key, kItemsPerThread, kWarpThreads, int32_t>;
    using StoreValues = cub::WarpStore<
        T, kItemsPerThread, cub::WARP_STORE_TRANSPOSE>;
    using StoreIndices = cub::WarpStore<
        int64_t, kItemsPerThread, cub::WARP_STORE_TRANSPOSE>;
    __shared__ union {
        typename LoadValues::TempStorage load_values;
        typename Sort::TempStorage sort;
        typename StoreValues::TempStorage store_values;
        typename StoreIndices::TempStorage store_indices;
    } temp_storage[kMaxBlockWarps];

    const int64_t slice = static_cast<int64_t>(blockIdx.x) * blockDim.y +
        threadIdx.y;
    if (slice >= slices) return;
    auto& warp_storage = temp_storage[threadIdx.y];
    const int64_t outer_index = slice / inner;
    const int64_t inner_index = slice - outer_index * inner;
    const int64_t base = outer_index * d_size * inner + inner_index;

    T local_values[kItemsPerThread];
    Key local_keys[kItemsPerThread];
    int32_t local_indices[kItemsPerThread];
    LoadValues(warp_storage.load_values).Load(
        topk_detail::TopKStridedReadAccessor<T>{in + base, inner},
        local_values, static_cast<int>(d_size), static_cast<T>(0));
    __syncwarp();
    #pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const int position = threadIdx.x * kItemsPerThread + item;
        const bool valid = position < d_size;
        local_indices[item] = valid ? static_cast<int32_t>(position) : -1;
        local_keys[item] = valid
            ? SortRadixTraits<T>::encode(local_values[item])
            : std::numeric_limits<Key>::max();
    }
    // The oob default sorts after every valid key under the active
    // comparator: all-ones sorts last ascending, zero sorts last under the
    // descending (greater-than) order.  NaN encodes to the all-ones key,
    // which the radix ordering already places last ascending.
    const Key oob_key = descending
        ? static_cast<Key>(0)
        : std::numeric_limits<Key>::max();
    if (descending) {
        Sort(warp_storage.sort).StableSort(
            local_keys, local_indices, SortKeyGreaterOp{},
            static_cast<int>(d_size), oob_key);
    } else {
        Sort(warp_storage.sort).StableSort(
            local_keys, local_indices, SortKeyLessOp{},
            static_cast<int>(d_size), oob_key);
    }
    #pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        local_values[item] = SortRadixTraits<T>::deconvert(local_keys[item]);
    }
    int64_t out_indices[kItemsPerThread];
    #pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        out_indices[item] = static_cast<int64_t>(local_indices[item]);
    }
    StoreValues(warp_storage.store_values).Store(
        topk_detail::TopKStridedWriteAccessor<T>{vals + base, inner},
        local_values, static_cast<int>(d_size));
    __syncwarp();
    StoreIndices(warp_storage.store_indices).Store(
        topk_detail::TopKStridedWriteAccessor<int64_t>{idxs + base, inner},
        out_indices, static_cast<int>(d_size));
}

// ---------------------------------------------------------------------------
// Block-per-slice radix sort.  One block stages one slice through shared
// memory with cub's collective primitives, so the whole sort costs one
// coalesced read and one coalesced write with no global scatter passes.
// Slices may be strided (any sort dimension); slices shorter than the block
// capacity are padded with keys that always sort to the end, matching the
// NaN-last ordering of the encoded floating keys.
// ---------------------------------------------------------------------------
template <typename T, int BlockThreads, int ItemsPerThread>
__global__ void sort_block_radix_kernel(
    const T* __restrict__ in, T* __restrict__ vals, int64_t* __restrict__ idxs,
    int64_t slices, int64_t d_size, int64_t inner, bool descending) {
    using Key = typename SortRadixTraits<T>::key_type;
    using LoadValues = cub::BlockLoad<T, BlockThreads, ItemsPerThread,
                                      cub::BLOCK_LOAD_TRANSPOSE>;
    using StoreValues = cub::BlockStore<T, BlockThreads, ItemsPerThread,
                                        cub::BLOCK_STORE_TRANSPOSE>;
    using StoreIndices = cub::BlockStore<int64_t, BlockThreads, ItemsPerThread,
                                         cub::BLOCK_STORE_TRANSPOSE>;
    using Sort = cub::BlockRadixSort<Key, BlockThreads, ItemsPerThread, int32_t>;
    __shared__ union {
        typename LoadValues::TempStorage load_values;
        typename Sort::TempStorage sort;
        typename StoreValues::TempStorage store_values;
        typename StoreIndices::TempStorage store_indices;
    } temp_storage;

    const int64_t slice = static_cast<int64_t>(blockIdx.x);
    if (slice >= slices) return;
    const int64_t outer_index = slice / inner;
    const int64_t inner_index = slice - outer_index * inner;
    const int64_t base = outer_index * d_size * inner + inner_index;

    T local_values[ItemsPerThread];
    int32_t local_indices[ItemsPerThread];
    Key local_keys[ItemsPerThread];

    // Keys that always sort to the end of the block regardless of direction.
    const Key end_key = descending ? static_cast<Key>(0)
                                   : std::numeric_limits<Key>::max();
    constexpr int capacity = BlockThreads * ItemsPerThread;
    if (d_size >= capacity) {
        LoadValues(temp_storage.load_values).Load(
            topk_detail::TopKStridedReadAccessor<T>{in + base, inner},
            local_values);
    } else {
        LoadValues(temp_storage.load_values).Load(
            topk_detail::TopKStridedReadAccessor<T>{in + base, inner},
            local_values, static_cast<int>(d_size), static_cast<T>(0));
    }
    __syncthreads();
    #pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int position = threadIdx.x * ItemsPerThread + item;
        const bool valid = position < d_size;
        local_indices[item] = valid ? static_cast<int32_t>(position) : -1;
        local_keys[item] = valid
            ? SortRadixTraits<T>::encode(local_values[item])
            : end_key;
    }
    if (descending) {
        Sort(temp_storage.sort).SortDescending(local_keys, local_indices);
    } else {
        Sort(temp_storage.sort).Sort(local_keys, local_indices);
    }
    __syncthreads();
    #pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        local_values[item] = SortRadixTraits<T>::deconvert(local_keys[item]);
    }
    int64_t out_indices[ItemsPerThread];
    #pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        out_indices[item] = static_cast<int64_t>(local_indices[item]);
    }
    StoreValues(temp_storage.store_values).Store(
        topk_detail::TopKStridedWriteAccessor<T>{vals + base, inner},
        local_values, static_cast<int>(d_size));
    __syncthreads();
    StoreIndices(temp_storage.store_indices).Store(
        topk_detail::TopKStridedWriteAccessor<int64_t>{idxs + base, inner},
        out_indices, static_cast<int>(d_size));
}

// Fixed block capacity: pick the smallest power-of-two bucket covering the
// slice length so the sort performs no wasted digit passes.  Short slices
// run one warp per slice instead: the merge sort amortizes the launch over
// 16 rows per block, which dominates a block-per-row radix pass there.
template <typename T>
void launch_sort_block_radix(
    const Tensor& self_c, Tensor& values, Tensor& indices,
    int64_t slices, int64_t d_size, int64_t inner, bool descending) {
    auto stream = getCurrentCUDAStream().stream();
    if (d_size <= 128) {
        dim3 block(32, 16);
        dim3 grid(static_cast<unsigned>((slices + 15) / 16));
        sort_warp_merge_kernel<T, 128><<<grid, block, 0, stream>>>(
            self_c.data_ptr<T>(), values.data_ptr<T>(),
            indices.data_ptr<int64_t>(), slices, d_size, inner, descending);
        return;
    }
    dim3 grid(static_cast<unsigned>(slices));
    #define TP_SORT_BLOCK_CASE(CAP, IPT)                                     \
        if (d_size <= CAP) {                                                 \
            sort_block_radix_kernel<T, CAP / IPT, IPT>                       \
                <<<grid, CAP / IPT, 0, stream>>>(                            \
                    self_c.data_ptr<T>(), values.data_ptr<T>(),              \
                    indices.data_ptr<int64_t>(), slices, d_size, inner,      \
                    descending);                                             \
            return;                                                          \
        }
    TP_SORT_BLOCK_CASE(256, 4)
    TP_SORT_BLOCK_CASE(512, 8)
    TP_SORT_BLOCK_CASE(1024, 8)
    TP_SORT_BLOCK_CASE(2048, 8)
    TP_SORT_BLOCK_CASE(4096, 8)
    #undef TP_SORT_BLOCK_CASE
}

void sort_block_radix_entry(const Tensor& self_c, Tensor& values, Tensor& indices,
                            int64_t slices, int64_t d_size, int64_t inner,
                            bool descending) {
    switch (self_c.dtype()) {
        #define TP_SORT_BLOCK_TYPE(ctype, name)                              \
        case DType::name:                                                    \
            launch_sort_block_radix<ctype>(                                  \
                self_c, values, indices, slices, d_size, inner, descending); \
            break;
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SORT_BLOCK_TYPE)
        #undef TP_SORT_BLOCK_TYPE
        default: TP_THROW(TypeError, "sort: unsupported dtype");
    }
}

template <typename T>
__global__ void sort_radix_pack_kernel(int64_t n, int64_t d_size, int64_t inner,
                                       const T* in,
                                       typename SortRadixTraits<T>::key_type* keys,
                                       int64_t* pos) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const int64_t slice = i / d_size;
        const int64_t j = i - slice * d_size;
        const int64_t o = slice / inner;
        const int64_t in2 = slice - o * inner;
        const int64_t src = (o * d_size + j) * inner + in2;
        keys[i] = SortRadixTraits<T>::encode(in[src]);
        pos[i] = j;
    }
}

template <typename T>
__global__ void sort_radix_unpack_kernel(int64_t n, int64_t d_size, int64_t inner,
                                         const typename SortRadixTraits<T>::key_type* keys,
                                         const int64_t* pos, T* vals, int64_t* idxs) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const int64_t slice = i / d_size;
        const int64_t j = i - slice * d_size;
        const int64_t o = slice / inner;
        const int64_t in2 = slice - o * inner;
        const int64_t dst = (o * d_size + j) * inner + in2;
        vals[dst] = SortRadixTraits<T>::deconvert(keys[i]);
        idxs[dst] = pos[i];
    }
}

// offsets[s] = s * d_size; end offsets are served by the same buffer shifted
// by one entry since every segment has identical length.
__global__ void sort_radix_fill_offsets_kernel(int n_offsets, int64_t d_size, int* offsets) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_offsets) offsets[i] = static_cast<int>(static_cast<int64_t>(i) * d_size);
}

template <typename T>
void sort_radix_impl(const Tensor& self_c, Tensor& values, Tensor& indices,
                     int64_t d_size, int64_t slices, int64_t inner, bool descending) {
    using Key = typename SortRadixTraits<T>::key_type;
    const int64_t n = self_c.numel();
    const auto device = self_c.device();
    const DType key_dtype = sizeof(Key) == 8 ? DType::UInt64 : DType::UInt32;
    Tensor keys_a = Tensor::empty({n}, key_dtype, device);
    Tensor keys_b = Tensor::empty({n}, key_dtype, device);
    Tensor pos_a = Tensor::empty({n}, DType::Int64, device);
    Tensor pos_b = Tensor::empty({n}, DType::Int64, device);
    Tensor offsets = Tensor::empty({slices + 1}, DType::Int32, device);
    auto stream = getCurrentCUDAStream().stream();
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    const int off_blocks = static_cast<int>((slices + 1 + kThreads - 1) / kThreads);
    sort_radix_pack_kernel<T><<<blocks, kThreads, 0, stream>>>(
        n, d_size, inner, static_cast<const T*>(self_c.data_ptr()),
        keys_a.data_ptr<Key>(), pos_a.data_ptr<int64_t>());
    sort_radix_fill_offsets_kernel<<<off_blocks, kThreads, 0, stream>>>(
        static_cast<int>(slices) + 1, d_size, offsets.data_ptr<int32_t>());
    cub::DoubleBuffer<Key> key_buf(keys_a.data_ptr<Key>(), keys_b.data_ptr<Key>());
    cub::DoubleBuffer<int64_t> pos_buf(pos_a.data_ptr<int64_t>(), pos_b.data_ptr<int64_t>());
    const int* begin_offsets = offsets.data_ptr<int32_t>();
    const int* end_offsets = begin_offsets + 1;
    const int n_items = static_cast<int>(n);
    const int n_segments = static_cast<int>(slices);
    const int bits = SortRadixTraits<T>::bit_count;
    size_t tmp_bytes = 0;
    cudaError_t err = descending
        ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
              nullptr, tmp_bytes, key_buf, pos_buf, n_items, n_segments,
              begin_offsets, end_offsets, 0, bits, stream)
        : cub::DeviceSegmentedRadixSort::SortPairs(
              nullptr, tmp_bytes, key_buf, pos_buf, n_items, n_segments,
              begin_offsets, end_offsets, 0, bits, stream);
    CUDA_CHECK(err);
    Tensor tmp = Tensor::empty({static_cast<int64_t>(std::max<size_t>(tmp_bytes, 1))},
                               DType::UInt8, device);
    err = descending
        ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
              tmp.data_ptr(), tmp_bytes, key_buf, pos_buf, n_items, n_segments,
              begin_offsets, end_offsets, 0, bits, stream)
        : cub::DeviceSegmentedRadixSort::SortPairs(
              tmp.data_ptr(), tmp_bytes, key_buf, pos_buf, n_items, n_segments,
              begin_offsets, end_offsets, 0, bits, stream);
    CUDA_CHECK(err);
    sort_radix_unpack_kernel<T><<<blocks, kThreads, 0, stream>>>(
        n, d_size, inner, key_buf.Current(), pos_buf.Current(),
        static_cast<T*>(values.data_ptr()), indices.data_ptr<int64_t>());
}

void radix_sort_impl(const Tensor& self_c, Tensor& values, Tensor& indices,
                     int64_t /*dim*/, int64_t /*outer*/, int64_t inner, int64_t d_size,
                     int64_t slices, bool descending) {
#define TP_RADIX_CASE(ctype, name) \
    case DType::name: \
        sort_radix_impl<ctype>(self_c, values, indices, d_size, slices, inner, descending); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_RADIX_CASE)
        default: TP_THROW(TypeError, "sort: unsupported dtype");
    }
#undef TP_RADIX_CASE
}

} // anonymous namespace

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

// ---------------------------------------------------------------------------
// sort / argsort (per-slice sort carrying original positions).
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> sort_cuda(const Tensor& self, int64_t dim, bool descending) {
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "sort: expects at least 1 dimension");
    dim = wrap_dim(dim, nd);
    Tensor self_c = self.contiguous();
    int64_t d_size = self_c.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self_c.shape()), dim, outer, inner);
    Tensor values = Tensor::empty(static_cast<std::vector<int64_t>>(self_c.shape()), self_c.dtype(), self_c.device());
    Tensor indices = Tensor::empty(static_cast<std::vector<int64_t>>(self_c.shape()), DType::Int64, self_c.device());
    int64_t slices = outer * inner;
    if (slices == 0 || d_size == 0) return {values, indices};
    auto stream = getCurrentCUDAStream().stream();
    // Radix path: one segmented radix pass orders every slice together.
    // Complexity is O(n * bytes) versus the heapsort fallback's O(n log n)
    // serialized per-slice walk; the fallback remains for tensors beyond
    // the 32-bit size limit of the device primitives.
    // Narrow multi-slice shapes go through the block-per-slice radix sort
    // instead: each slice is ordered entirely in shared memory, which avoids
    // the global scatter/gather of the segmented device pass.  The slice
    // decomposition is (outer, d_size, inner) over the contiguous layout, so
    // a sort dimension that is not the innermost one is served by the same
    // kernel through its strided element accessors when the row stride stays
    // small enough to sit inside the cache hierarchy.  A long stride turns
    // every load and store into its own transaction, so those shapes stage
    // through a permuted contiguous buffer instead: the staging pass and the
    // ordered write-back walk contiguous rows, and the layout fix-up is one
    // strided copy.
    if (self_c.numel() <= std::numeric_limits<int>::max() &&
        d_size >= 2 && d_size <= 4096 && slices > 1) {
        constexpr int64_t kMaxStridedInner = 16;
        if (inner > kMaxStridedInner) {
            std::vector<int64_t> order(static_cast<size_t>(nd));
            for (int64_t d = 0; d < nd; ++d) order[static_cast<size_t>(d)] = d;
            std::swap(order[static_cast<size_t>(dim)],
                      order[static_cast<size_t>(nd - 1)]);
            Tensor staged = self_c.permute(order).contiguous();
            Tensor staged_values = Tensor::empty(
                static_cast<std::vector<int64_t>>(staged.shape()),
                staged.dtype(), staged.device());
            Tensor staged_indices = Tensor::empty(
                static_cast<std::vector<int64_t>>(staged.shape()),
                DType::Int64, staged.device());
            sort_block_radix_entry(staged, staged_values, staged_indices,
                                   slices, d_size, 1, descending);
            std::vector<int64_t> inverse(static_cast<size_t>(nd));
            for (int64_t d = 0; d < nd; ++d) {
                inverse[static_cast<size_t>(order[static_cast<size_t>(d)])] = d;
            }
            values.copy_(staged_values.permute(inverse));
            indices.copy_(staged_indices.permute(inverse));
        } else {
            sort_block_radix_entry(self_c, values, indices,
                                   slices, d_size, inner, descending);
        }
        CUDA_CHECK(cudaGetLastError());
        return {values, indices};
    }
    if (self_c.numel() <= std::numeric_limits<int>::max()) {
        radix_sort_impl(self_c, values, indices, dim, outer, inner, d_size, slices, descending);
        CUDA_CHECK(cudaGetLastError());
        return {values, indices};
    }
#define TP_SORT_CASE(ctype, name) \
    case DType::name: \
        sort_kernel<ctype><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            slices, d_size, inner, descending, self_c.data_ptr<ctype>(), \
            values.data_ptr<ctype>(), indices.data_ptr<int64_t>()); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SORT_CASE)
        default: TP_THROW(TypeError, "sort: unsupported dtype");
    }
#undef TP_SORT_CASE
    CUDA_CHECK(cudaGetLastError());
    return {values, indices};
}

Tensor argsort_cuda(const Tensor& self, int64_t dim, bool descending) {
    // Indices-only variant of the per-slice sort.
    return std::get<1>(sort_cuda(self, dim, descending));
}

// ---------------------------------------------------------------------------
//   flags[i] = (i == 0) || sorted[i] != sorted[i-1]
//   group id = inclusive cumsum(flags) - 1
//   inverse[order[i]] = gid[i]; counts[g] = next boundary - boundary
// ---------------------------------------------------------------------------
namespace {

__global__ void unique_inverse_kernel(int64_t n, const int64_t* __restrict__ order,
                                      const int64_t* __restrict__ gid,
                                      int64_t* __restrict__ inverse) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) inverse[order[i]] = gid[i] - 1;
}

template <typename T>
__global__ void unique_emit_kernel(int64_t n, const T* __restrict__ sorted,
                                   const int64_t* __restrict__ flags,
                                   const int64_t* __restrict__ gid_inclusive,
                                   T* __restrict__ values,
                                   int64_t* __restrict__ starts) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || !flags[i]) return;
    const int64_t g = gid_inclusive[i] - 1;
    values[g] = sorted[i];
    if (starts != nullptr) starts[g] = i;
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> unique_cuda(const Tensor& self, bool sorted,
                                               bool return_inverse,
                                               bool return_counts) {
    Tensor flat = self.contiguous().reshape({self.numel()});
    const int64_t n = flat.numel();

    Tensor values = Tensor::empty({0}, self.dtype(), self.device());
    Tensor inverse = return_inverse
                         ? Tensor::empty(
                               static_cast<std::vector<int64_t>>(self.shape()),
                               DType::Int64, self.device())
                                    : Tensor();
    Tensor counts = return_counts ? Tensor::empty({0}, DType::Int64, self.device())
                                  : Tensor();
    if (n == 0) return std::make_tuple(values, inverse, counts);

    auto [sorted_vals, order] = sort_cuda(flat, 0, false);
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    Tensor flags = Tensor::zeros({n}, DType::Int64, self.device());

    #define UNIQUE_FLAGS_CASE(ctype, name)                                      \
    case DType::name: {                                                         \
        const ctype* sorted = sorted_vals.data_ptr<ctype>();                    \
        gpu_kernel_with_index(flags, [=] GPU_LAMBDA(int64_t i) -> int64_t {     \
            return (i == 0 || sorted[i] != sorted[i - 1]) ? 1 : 0;              \
        });                                                                      \
        break;                                                                   \
    }

    switch (self.dtype()) {
        UNIQUE_FLAGS_CASE(float, Float32)
        UNIQUE_FLAGS_CASE(double, Float64)
        UNIQUE_FLAGS_CASE(int64_t, Int64)
        UNIQUE_FLAGS_CASE(int32_t, Int32)
        UNIQUE_FLAGS_CASE(int16_t, Int16)
        UNIQUE_FLAGS_CASE(int8_t, Int8)
        UNIQUE_FLAGS_CASE(uint8_t, UInt8)
        UNIQUE_FLAGS_CASE(uint16_t, UInt16)
        UNIQUE_FLAGS_CASE(uint32_t, UInt32)
        UNIQUE_FLAGS_CASE(uint64_t, UInt64)
        UNIQUE_FLAGS_CASE(Half, Float16)
        UNIQUE_FLAGS_CASE(BFloat16, BFloat16)
        UNIQUE_FLAGS_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError, "unique: unsupported dtype on CUDA");
    }
    #undef UNIQUE_FLAGS_CASE

    // gid = inclusive cumsum(flags); last element == number of groups.
    Tensor gid = flags.cumsum(0);
    const int64_t num_groups =
        gid.to(Device(DeviceType::CPU)).data_ptr<int64_t>()[n - 1];

    values = Tensor::empty({num_groups}, self.dtype(), self.device());
    if (return_inverse) {
        inverse = Tensor::empty({n}, DType::Int64, self.device());
        unique_inverse_kernel<<<blocks, threads>>>(
            n, order.data_ptr<int64_t>(), gid.data_ptr<int64_t>(),
            inverse.data_ptr<int64_t>());
    }
    Tensor starts;
    int64_t* starts_ptr = nullptr;
    if (return_counts) {
        counts = Tensor::zeros({num_groups}, DType::Int64, self.device());
        starts = Tensor::full({num_groups}, int64_t(-1), DType::Int64,
                              self.device());
        starts_ptr = starts.data_ptr<int64_t>();
    }

    #define UNIQUE_EMIT_CASE(ctype, name)                                      \
        case DType::name:                                                      \
            unique_emit_kernel<ctype><<<blocks, threads>>>(                     \
                n, sorted_vals.data_ptr<ctype>(), flags.data_ptr<int64_t>(),   \
                gid.data_ptr<int64_t>(), values.data_ptr<ctype>(),             \
                starts_ptr);                                                     \
            break;
    switch (self.dtype()) {
        UNIQUE_EMIT_CASE(float, Float32)
        UNIQUE_EMIT_CASE(double, Float64)
        UNIQUE_EMIT_CASE(int64_t, Int64)
        UNIQUE_EMIT_CASE(int32_t, Int32)
        UNIQUE_EMIT_CASE(int16_t, Int16)
        UNIQUE_EMIT_CASE(int8_t, Int8)
        UNIQUE_EMIT_CASE(uint8_t, UInt8)
        UNIQUE_EMIT_CASE(uint16_t, UInt16)
        UNIQUE_EMIT_CASE(uint32_t, UInt32)
        UNIQUE_EMIT_CASE(uint64_t, UInt64)
        UNIQUE_EMIT_CASE(Half, Float16)
        UNIQUE_EMIT_CASE(BFloat16, BFloat16)
        UNIQUE_EMIT_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError, "unique: unsupported dtype on CUDA");
    }
    #undef UNIQUE_EMIT_CASE
    CUDA_CHECK(cudaGetLastError());

    if (return_counts) {
        gpu_kernel_with_index(
            counts, [=] GPU_LAMBDA(int64_t g) -> int64_t {
                const int64_t end = (g + 1 < num_groups) ? starts_ptr[g + 1] : n;
                return end - starts_ptr[g];
            });
    }
    return std::make_tuple(values, inverse, counts);
}

// Two-output flat unique: drops the counts tensor of the three-output path.
std::tuple<Tensor, Tensor> _unique_cuda(const Tensor& self, bool sorted,
                                        bool return_inverse) {
    auto result = unique_cuda(self, sorted, return_inverse, /*return_counts=*/false);
    return std::make_tuple(std::get<0>(result), std::get<1>(result));
}

std::tuple<Tensor, Tensor, Tensor> _unique2_cuda(const Tensor& self, bool sorted,
                                                 bool return_inverse,
                                                 bool return_counts) {
    return unique_cuda(self, sorted, return_inverse, return_counts);
}

// Row equality over the flattened {n, row_len} matrix: two rows match when
// every column pair is equal (a NaN cell never matches another NaN).
template <typename T>
__global__ void unique_row_equal_kernel(int64_t n, int64_t row_len,
                                        const T* rows, int64_t* is_new) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (i == 0) { is_new[0] = 1; continue; }
        const T* cur = rows + i * row_len;
        const T* prev = rows + (i - 1) * row_len;
        int64_t same = 1;
        for (int64_t c = 0; c < row_len; ++c) {
            if (cur[c] != prev[c]) { same = 0; break; }
        }
        is_new[i] = same ? 0 : 1;
    }
}

// Gathers the kept rows into a compact buffer and writes the inverse mapping
// from original row positions to group ids (row order already applied).
template <typename T>
__global__ void unique_row_emit_kernel(int64_t n, int64_t row_len,
                                       const T* rows, const int64_t* order,
                                       const int64_t* gid, T* out,
                                       int64_t* inverse) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const int64_t row = order[i];
        const int64_t g = gid[i] - 1;  // inclusive cumsum -> 0-based group id
        const T* src = rows + i * row_len;
        T* dst = out + g * row_len;
        for (int64_t c = 0; c < row_len; ++c) dst[c] = src[c];
        if (inverse != nullptr) inverse[row] = g;
    }
}

// Dim-wise unique.  Rows (slices along `dim`) are sorted lexicographically by
// `sort_cuda` applied to the transposed matrix — sorting each row-position
// column independently is not row order, so instead the sort runs on a
// {row_len, n} layout along the last axis, which orders rows by their
// first column, and repeated stable passes over remaining columns refine the
// order (LSD across columns; each pass must be stable, guaranteed by the
// tie-breaking index in the radix sort).
std::tuple<Tensor, Tensor, Tensor> unique_dim_cuda_impl(const Tensor& self,
                                                        int64_t dim,
                                                        bool consecutive,
                                                        bool return_inverse,
                                                        bool return_counts) {
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(self.shape());
    const int64_t zero_dims = std::count(sizes.begin(), sizes.end(), 0);
    if (self.size(dim) == 0) {
        TP_CHECK(zero_dims == 1,
                 "Number of zero sized dimensions is more than one, so unique "
                 "cannot be applied");
        Tensor values = Tensor::empty(sizes, self.dtype(), self.device());
        Tensor inverse = Tensor::empty({0}, DType::Int64, self.device());
        Tensor counts = Tensor::empty({0}, DType::Int64, self.device());
        return std::make_tuple(values, inverse, counts);
    }
    TP_CHECK(zero_dims == 0,
             "There are 0 sized dimensions, and they aren't selected, so "
             "unique cannot be applied");

    Tensor input_flat = self.moveaxis(dim, 0).contiguous();
    std::vector<int64_t> front_sizes =
        static_cast<std::vector<int64_t>>(input_flat.shape());
    const int64_t n = front_sizes[0];
    input_flat = input_flat.reshape({n, -1});
    const int64_t row_len = input_flat.size(1);

    Tensor rows_sorted;
    Tensor order;
    if (consecutive) {
        rows_sorted = input_flat;
        order = Tensor::arange(Scalar(int64_t(0)), Scalar(n), Scalar(int64_t(1)),
                               DType::Int64, self.device());
    } else {
        // LSD refinement: stable-sort rows by each column from last to first.
        // The radix sort's tie-breaking on row index keeps each pass stable.
        rows_sorted = input_flat;
        order = Tensor::arange(Scalar(int64_t(0)), Scalar(n), Scalar(int64_t(1)),
                               DType::Int64, self.device());
        for (int64_t c = row_len - 1; c >= 0; --c) {
            // Sort the current rows by column c: gather the column, sort its
            // (key, row) pairs, and reorder the rows through the permutation.
            Tensor col = rows_sorted.slice(1, c, c + 1).reshape({n});
            Tensor col_sorted, col_order;
            std::tie(col_sorted, col_order) = sort_cuda(col, 0, false);
            // col_order indexes rows within the current rows_sorted layout.
            // Apply the same permutation to the row data and original indices.
            order = order.gather(0, col_order);
            Tensor idx = col_order.reshape({n, 1})
                             .expand(std::vector<int64_t>{n, row_len});
            rows_sorted = rows_sorted.gather(0, idx);
        }
    }

    Tensor flags = Tensor::zeros({n}, DType::Int64, self.device());
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    auto stream = getCurrentCUDAStream().stream();

#define UNIQUE_ROW_CASE(ctype, name)                                          \
    case DType::name:                                                          \
        unique_row_equal_kernel<ctype><<<blocks, threads, 0, stream>>>(        \
            n, row_len, rows_sorted.data_ptr<ctype>(),                         \
            flags.data_ptr<int64_t>());                                         \
        break;
    switch (self.dtype()) {
        UNIQUE_ROW_CASE(float, Float32)
        UNIQUE_ROW_CASE(double, Float64)
        UNIQUE_ROW_CASE(int64_t, Int64)
        UNIQUE_ROW_CASE(int32_t, Int32)
        UNIQUE_ROW_CASE(int16_t, Int16)
        UNIQUE_ROW_CASE(int8_t, Int8)
        UNIQUE_ROW_CASE(uint8_t, UInt8)
        UNIQUE_ROW_CASE(uint16_t, UInt16)
        UNIQUE_ROW_CASE(uint32_t, UInt32)
        UNIQUE_ROW_CASE(uint64_t, UInt64)
        UNIQUE_ROW_CASE(Half, Float16)
        UNIQUE_ROW_CASE(BFloat16, BFloat16)
        UNIQUE_ROW_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError,
                     "unique_dim: unsupported dtype on CUDA");
    }
#undef UNIQUE_ROW_CASE

    Tensor gid = flags.cumsum(0);
    const int64_t num_groups =
        gid.to(Device(DeviceType::CPU)).data_ptr<int64_t>()[n - 1];

    Tensor kept_rows = Tensor::empty({num_groups, row_len}, self.dtype(),
                                     self.device());
    Tensor inverse = return_inverse
                         ? Tensor::empty({n}, DType::Int64, self.device())
                         : Tensor();
    int64_t* inverse_ptr = return_inverse ? inverse.data_ptr<int64_t>() : nullptr;

#define UNIQUE_ROW_EMIT_CASE(ctype, name)                                     \
    case DType::name:                                                          \
        unique_row_emit_kernel<ctype><<<blocks, threads, 0, stream>>>(         \
            n, row_len, rows_sorted.data_ptr<ctype>(),                         \
            order.data_ptr<int64_t>(), gid.data_ptr<int64_t>(),                \
            kept_rows.data_ptr<ctype>(), inverse_ptr);                           \
        break;
    switch (self.dtype()) {
        UNIQUE_ROW_EMIT_CASE(float, Float32)
        UNIQUE_ROW_EMIT_CASE(double, Float64)
        UNIQUE_ROW_EMIT_CASE(int64_t, Int64)
        UNIQUE_ROW_EMIT_CASE(int32_t, Int32)
        UNIQUE_ROW_EMIT_CASE(int16_t, Int16)
        UNIQUE_ROW_EMIT_CASE(int8_t, Int8)
        UNIQUE_ROW_EMIT_CASE(uint8_t, UInt8)
        UNIQUE_ROW_EMIT_CASE(uint16_t, UInt16)
        UNIQUE_ROW_EMIT_CASE(uint32_t, UInt32)
        UNIQUE_ROW_EMIT_CASE(uint64_t, UInt64)
        UNIQUE_ROW_EMIT_CASE(Half, Float16)
        UNIQUE_ROW_EMIT_CASE(BFloat16, BFloat16)
        UNIQUE_ROW_EMIT_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError,
                     "unique_dim: unsupported dtype on CUDA");
    }
#undef UNIQUE_ROW_EMIT_CASE

    front_sizes[0] = num_groups;
    Tensor values = kept_rows.reshape(front_sizes).moveaxis(0, dim);

    Tensor counts;
    if (return_counts) {
        // counts[g] = number of positions whose gid equals g+1; resolved via
        // a bincount over the shifted gid buffer.
        Tensor one = Tensor::ones({n}, DType::Int64, self.device());
        Tensor shifted = gid.sub(Scalar(int64_t(1)));
        counts = shifted.bincount(one, num_groups);
    }
    return std::make_tuple(values, inverse, counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_cuda(const Tensor& self,
                                                   int64_t dim, bool sorted,
                                                   bool return_inverse,
                                                   bool return_counts) {
    (void)sorted;
    return unique_dim_cuda_impl(self, dim, /*consecutive=*/false,
                                return_inverse, return_counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_cuda(
        const Tensor& self, int64_t dim, bool return_inverse,
        bool return_counts) {
    return unique_dim_cuda_impl(self, dim, /*consecutive=*/true,
                                return_inverse, return_counts);
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
    m.impl("sort", sort_cuda);
    m.impl("argsort", argsort_cuda);
    m.impl("unique", unique_cuda);
    m.impl("_unique", _unique_cuda);
    m.impl("_unique2", _unique2_cuda);
    m.impl("unique_dim", unique_dim_cuda);
    m.impl("unique_dim_consecutive", unique_dim_consecutive_cuda);
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
