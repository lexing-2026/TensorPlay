#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "GradMode.h"
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

// RAII over the thread-local autograd switch: sections that mutate storage
// directly must not record graph nodes.
struct NoGradGuard {
    bool prev;
    NoGradGuard() : prev(GradMode::is_enabled()) { GradMode::set_enabled(false); }
    ~NoGradGuard() { GradMode::set_enabled(prev); }
};

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

// Contiguous fast path: the outer and row counts ride on the grid, so the
// coordinates never need a per-element division, and each thread walks
// kContigWork columns of its row so several scattered source reads are in
// flight at once.
constexpr int kContigThreads = 128;
constexpr int kContigWork = 8;

template <typename T, typename IndexT>
__global__ void gather_contig_kernel(int64_t idx_dim_size, int64_t idx_inner,
                                     int64_t self_dim_size, int64_t self_inner,
                                     const T* s, const IndexT* ip, T* d) {
    const int64_t outer_off = blockIdx.z;
    const int64_t row_base = (outer_off * idx_dim_size + blockIdx.y) * idx_inner;
    const int64_t t0 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t idxv[kContigWork];
#pragma unroll
    for (int k = 0; k < kContigWork; ++k) {
        const int64_t t = t0 + static_cast<int64_t>(k) * step;
        if (t < idx_inner) idxv[k] = ip[row_base + t];
    }
#pragma unroll
    for (int k = 0; k < kContigWork; ++k) {
        const int64_t t = t0 + static_cast<int64_t>(k) * step;
        if (t >= idx_inner) continue;
        int64_t idx = idxv[k];
        if (idx < 0) idx += self_dim_size;
        d[row_base + t] = s[(outer_off * self_dim_size + idx) * self_inner + t];
    }
}

inline bool scatter_gather_contig_eligible(int64_t outer, int64_t idx_dim_size,
                                           int64_t idx_inner) {
    return outer <= 65535 && idx_dim_size <= 65535 &&
           idx_inner >= kContigThreads * kContigWork;
}

// Gather along the trailing dimension: every output element of a row picks one
// source slot, so the row rides on grid.y and the column index comes straight
// from the thread id. No per-element division, and each thread keeps
// kContigWork reads in flight.
template <typename T, typename IndexT>
__global__ void gather_tail_kernel(int64_t idx_dim_size, int64_t self_dim_size,
                                   int64_t self_inner, const T* __restrict__ s,
                                   const IndexT* __restrict__ ip, T* __restrict__ d) {
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    const int64_t out_base = row * idx_dim_size;
    const int64_t in_base = row * self_dim_size * self_inner;
    const int64_t t0 = static_cast<int64_t>(blockIdx.x) * kContigThreads + threadIdx.x;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    IndexT idxv[kContigWork];
#pragma unroll
    for (int k = 0; k < kContigWork; ++k) {
        const int64_t t = t0 + static_cast<int64_t>(k) * step;
        if (t < idx_dim_size) idxv[k] = ip[out_base + t];
    }
#pragma unroll
    for (int k = 0; k < kContigWork; ++k) {
        const int64_t t = t0 + static_cast<int64_t>(k) * step;
        if (t >= idx_dim_size) continue;
        int64_t idx = static_cast<int64_t>(idxv[k]);
        if (idx < 0) idx += self_dim_size;
        d[out_base + t] = s[in_base + idx * self_inner];
    }
}

template <typename IndexT, typename T>
void launch_gather_typed(bool contig, int64_t n, int64_t outer, int64_t idx_dim_size,
                         int64_t idx_inner, int64_t self_dim_size, int64_t self_inner,
                         const T* s, const IndexT* ip, T* d, cudaStream_t stream) {
    if (contig) {
        const dim3 grid(static_cast<unsigned>(
                            (idx_inner + kContigThreads * kContigWork - 1) /
                            (kContigThreads * kContigWork)),
                        static_cast<unsigned>(idx_dim_size),
                        static_cast<unsigned>(outer));
        gather_contig_kernel<T, IndexT><<<grid, kContigThreads, 0, stream>>>(
            idx_dim_size, idx_inner, self_dim_size, self_inner, s, ip, d);
        return;
    }
    if (idx_inner == 1 && idx_dim_size >= kContigThreads && outer <= 65535) {
        const dim3 grid(static_cast<unsigned>(
                            (idx_dim_size + kContigThreads * kContigWork - 1) /
                            (kContigThreads * kContigWork)),
                        static_cast<unsigned>(outer));
        gather_tail_kernel<T, IndexT><<<grid, kContigThreads, 0, stream>>>(
            idx_dim_size, self_dim_size, self_inner, s, ip, d);
        return;
    }
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    gather_kernel<T, IndexT><<<blocks, kThreads, 0, stream>>>(
        n, idx_dim_size, idx_inner, self_dim_size, self_inner, s, ip, d);
}

template <typename IndexT>
void launch_gather_kernel_for_index(
        int64_t n, int64_t outer, int64_t idx_dim_size, int64_t idx_inner,
        int64_t self_dim_size, int64_t self_inner, const Tensor& self,
        const Tensor& index, Tensor& result, cudaStream_t stream) {
    const bool contig = scatter_gather_contig_eligible(outer, idx_dim_size, idx_inner);
#define TP_GA_CASE(ctype, name) \
    case DType::name: \
        launch_gather_typed<IndexT, ctype>( \
            contig, n, outer, idx_dim_size, idx_inner, self_dim_size, self_inner, \
            static_cast<const ctype*>(self.data_ptr()), index.data_ptr<IndexT>(), \
            static_cast<ctype*>(result.data_ptr()), stream); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_GA_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_GA_CASE)
        TP_GA_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_GA_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_GA_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_GA_CASE(tensorplay::complex<BFloat16>, BComplex32)
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

// Contiguous counterpart of scatter_kernel: the grid supplies the outer and row
// coordinates, so the per-element coordinate never needs a division.  A block
// takes a contiguous strip of a row and each thread covers `work` columns of it
// at a stride of kContigThreads, which keeps the scattered writes of one strip
// together and amortises the block preamble; `work` is chosen at launch because
// it trades block count against per-thread run length.
constexpr int kScatterContigMaxWork = 32;

template <typename T, bool Add, typename IndexT>
__global__ void scatter_contig_kernel(int64_t idx_dim_size, int64_t idx_inner,
                                      int64_t self_dim_size, int64_t self_inner,
                                      int64_t work, T* d, const IndexT* ip,
                                      const T* vp) {
    const int64_t outer_off = blockIdx.z;
    const int64_t row_base = (outer_off * idx_dim_size + blockIdx.y) * idx_inner;
    int64_t t = static_cast<int64_t>(blockIdx.x) * kContigThreads * work + threadIdx.x;
#pragma unroll 8
    for (int64_t k = 0; k < work; ++k) {
        if (t < idx_inner) {
            const int64_t flat = row_base + t;
            int64_t idx = ip[flat];
            if (idx < 0) idx += self_dim_size;
            const int64_t dst = (outer_off * self_dim_size + idx) * self_inner + t;
            if constexpr (Add) {
                if constexpr (scatter_add_supported_v<T>) {
                    indexed_atomic_add(&d[dst], vp[flat]);
                }
            }
            else d[dst] = vp[flat];
        }
        t += kContigThreads;
    }
}

template <typename IndexT, typename T, bool Add>
void launch_scatter_typed(bool contig, int64_t total_idx, int64_t outer,
                          int64_t idx_dim_size, int64_t idx_inner,
                          int64_t self_dim_size, int64_t self_inner,
                          T* d, const IndexT* ip, const T* vp, cudaStream_t stream) {
    if (contig) {
        // Prefer one block per row for narrow rows, and cap the per-thread
        // strip so wide rows still spread over many blocks.
        int64_t work = (idx_inner + kContigThreads - 1) / kContigThreads;
        if (work < kContigWork) work = kContigWork;
        if (work > kScatterContigMaxWork) work = kScatterContigMaxWork;
        const dim3 grid(static_cast<unsigned>(
                            (idx_inner + kContigThreads * work - 1) /
                            (kContigThreads * work)),
                        static_cast<unsigned>(idx_dim_size),
                        static_cast<unsigned>(outer));
        scatter_contig_kernel<T, Add, IndexT><<<grid, kContigThreads, 0, stream>>>(
            idx_dim_size, idx_inner, self_dim_size, self_inner, work, d, ip, vp);
        return;
    }
    scatter_kernel<T, Add, IndexT><<<
        (total_idx + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        total_idx, idx_dim_size, idx_inner, self_dim_size, self_inner, d, ip, vp);
}

template <typename IndexT, bool Add>
void launch_scatter_for_index(
        int64_t total_idx, int64_t outer, int64_t idx_dim_size, int64_t idx_inner,
        int64_t self_dim_size, int64_t self_inner, Tensor& result,
        const Tensor& index, const Tensor& source, cudaStream_t stream) {
    const bool contig = scatter_gather_contig_eligible(outer, idx_dim_size, idx_inner);
#define TP_SC_CASE(ctype, name) \
    case DType::name: \
        launch_scatter_typed<IndexT, ctype, Add>( \
            contig, total_idx, outer, idx_dim_size, idx_inner, self_dim_size, \
            self_inner, static_cast<ctype*>(result.data_ptr()), \
            index.data_ptr<IndexT>(), static_cast<const ctype*>(source.data_ptr()), \
            stream); \
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
            total_idx, idx_outer, idx_dim_size, idx_inner, self_dim_size, inner,
            result, idx_c, src_b, stream);
    } else {
        launch_scatter_for_index<int64_t, Add>(
            total_idx, idx_outer, idx_dim_size, idx_inner, self_dim_size, inner,
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
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= idx_c.size(i);
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
            n, outer, idx_dim_size, idx_inner, self_dim_size, self_inner,
            self_c, idx_c, result, stream);
    } else {
        launch_gather_kernel_for_index<int64_t>(
            n, outer, idx_dim_size, idx_inner, self_dim_size, self_inner,
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
                total_idx, idx_outer, idx_dim_size, idx_inner, self_dim_size, inner,
                result, idx_c, src_b, stream);
        } else {
            launch_scatter_for_index<int64_t, true>(
                total_idx, idx_outer, idx_dim_size, idx_inner, self_dim_size, inner,
                result, idx_c, src_b, stream);
        }
    } else {
        if (idx_c.dtype() == DType::Int32) {
            launch_scatter_for_index<int32_t, false>(
                total_idx, idx_outer, idx_dim_size, idx_inner, self_dim_size, inner,
                result, idx_c, src_b, stream);
        } else {
            launch_scatter_for_index<int64_t, false>(
                total_idx, idx_outer, idx_dim_size, idx_inner, self_dim_size, inner,
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

Tensor& index_add_inplace_cuda(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    NoGradGuard __tp_nograd;
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64 || index.dtype() == DType::Int32)
        ? index.contiguous() : index.to(DType::Int64).contiguous();
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return self;
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
    if (total == 0) return self;
    if (!self.is_contiguous()) {
        // The kernel addresses the destination as a dense buffer, so a
        // strided destination keeps the out-of-place path.
        self.copy_(index_add_cuda(self, dim, index, source));
        return self;
    }
    auto stream = getCurrentCUDAStream().stream();
    if (idx.dtype() == DType::Int32) {
        launch_index_add_for_index<int32_t>(
            total, inner, row, n_idx, self, idx, source_c, stream);
    } else {
        launch_index_add_for_index<int64_t>(
            total, inner, row, n_idx, self, idx, source_c, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return self;
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
    m.impl("index_add_", index_add_inplace_cuda);
}

}
}
