#include "QuantTensorOperators.h"
// High-throughput indexing kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "GradMode.h"

#include <cuda_runtime.h>
#include "GPUPrimitives.cuh"
#include "Complex.h"
#include "CUDALoops.cuh"

#include <vector>
#include <cstdint>
#include <limits>

// Index kernels validate their index values on the device itself: an
// out-of-range value faults the launch instead of reading or writing out of
// bounds.  The check is deliberately active in release builds, so invalid
// indexing input cannot corrupt memory silently.  Negative values wrap into
// range afterwards, matching advanced-indexing semantics.
#define TP_INDEX_RANGE_GUARD(iv, row)                 \
    if ((iv) < -(row) || (iv) >= (row)) { TP_DEVICE_TRAP(); } \
    if ((iv) < 0) { (iv) += (row); }
#include <string>

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

// Block size for the slice-structured kernels: just enough threads to cover
// one slice, with the larger blocks kept for wide slices.
inline int slice_block_threads(int64_t inner) {
    if (inner >= 1024) return 512;
    if (inner >= 256) return 256;
    return static_cast<int>((inner + 31) / 32 * 32);
}

// The slice kernels parallelize over (outer, index) pairs; below this many
// pairs the flat kernel spreads the work over more blocks.
inline bool use_slice_kernel(int64_t inner, int64_t n_idx, int64_t outer) {
    return inner >= 64 &&
           n_idx * std::min<int64_t>(outer, 65535) >= 64;
}

// The chunked slice kernels move each row in 16-byte units, which needs the
// row payload to divide evenly into chunks and both ends of every move to sit
// on a 16-byte boundary.
inline bool vec_slice_ok(const void* first, const void* second) {
    if (reinterpret_cast<uintptr_t>(first) % 16 != 0) return false;
    return second == nullptr ||
           reinterpret_cast<uintptr_t>(second) % 16 == 0;
}

// Block shape for the chunked slice kernels: `c_span` chunks of a row across
// the x dimension and `k_span` rows per block down the y dimension, tuned so
// that a block holds close to 128 threads for narrow rows while wide rows get
// the larger blocks.
inline void vec_slice_block(int64_t chunks, int& c_span, int& k_span) {
    if (chunks >= 512) {
        c_span = 512;
        k_span = 1;
    } else if (chunks >= 128) {
        c_span = static_cast<int>((chunks + 31) / 32 * 32);
        k_span = 1;
    } else {
        c_span = static_cast<int>(std::max<int64_t>(chunks, 1));
        k_span = static_cast<int>(
            std::min<int64_t>(std::max<int64_t>(1, 128 / c_span), 16));
    }
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

// One block column per index entry: the block walks the destination slices of
// that entry in order, so the coordinate split costs no division and the index
// value is loaded once per block.
template <typename T, typename IndexT>
__global__ void index_select_slice_kernel(int64_t n_idx, int64_t outer,
                                          int64_t inner, int64_t row,
                                          const T* s, const IndexT* ip, T* d) {
    const int64_t k_stride = static_cast<int64_t>(gridDim.x);
    const int64_t o_stride = static_cast<int64_t>(gridDim.y);
    for (int64_t o = blockIdx.y; o < outer; o += o_stride) {
        for (int64_t k = blockIdx.x; k < n_idx; k += k_stride) {
            int64_t source_index = ip[k];
            TP_INDEX_RANGE_GUARD(source_index, row);
            const T* source = s + (o * row + source_index) * inner;
            T* destination = d + (o * n_idx + k) * inner;
            for (int64_t c = threadIdx.x; c < inner; c += blockDim.x) {
                destination[c] = source[c];
            }
        }
    }
}

// Selecting along the trailing dimension: an index entry selects exactly one
// element of the output row, so the index vector lines up with the output and
// no coordinate split is needed.  Four 4-byte columns form a 16-byte cell: the
// cell's index quadruple and its destination both move as one vector, which
// keeps the instruction count per element low.
constexpr int kTailThreads = 128;

template <typename IndexT>
__global__ void index_select_tail4_kernel(int64_t cells, int64_t row,
                                          const uint32_t* __restrict__ s,
                                          const IndexT* __restrict__ ip,
                                          uint4* __restrict__ d) {
    const int64_t o = blockIdx.y;
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= cells) return;
    const int64_t base = c * 4;
    int64_t iv[4];
    if constexpr (sizeof(IndexT) == 4) {
        const int4 v = *reinterpret_cast<const int4*>(ip + base);
        iv[0] = v.x; iv[1] = v.y; iv[2] = v.z; iv[3] = v.w;
    } else {
        const longlong2 lo = *reinterpret_cast<const longlong2*>(ip + base);
        const longlong2 hi = *reinterpret_cast<const longlong2*>(ip + base + 2);
        iv[0] = lo.x; iv[1] = lo.y; iv[2] = hi.x; iv[3] = hi.y;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) { TP_INDEX_RANGE_GUARD(iv[k], row); }
    const uint32_t* src = s + o * row;
    uint4 cell;
    cell.x = src[iv[0]];
    cell.y = src[iv[1]];
    cell.z = src[iv[2]];
    cell.w = src[iv[3]];
    d[o * cells + c] = cell;
}

// Trailing-dimension select for every other element width: one output element
// per thread, still without any coordinate division.
template <typename T, typename IndexT>
__global__ void index_select_tail_scalar_kernel(int64_t n_idx, int64_t row,
                                                const T* __restrict__ s,
                                                const IndexT* __restrict__ ip,
                                                T* __restrict__ d) {
    const int64_t o = blockIdx.y;
    const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n_idx) return;
    int64_t iv = static_cast<int64_t>(ip[k]);
    TP_INDEX_RANGE_GUARD(iv, row);
    d[o * n_idx + k] = s[o * row + iv];
}

template <typename IndexT> void launch_index_select_tail(
        int64_t n_idx, int64_t row, int64_t outer, const Tensor& self,
        const Tensor& index, Tensor& result, cudaStream_t stream) {
    const int64_t isz = static_cast<int64_t>(self.itemsize());
    if (isz == 4 && n_idx % 4 == 0) {
        const int64_t cells = n_idx / 4;
        const unsigned blocks = static_cast<unsigned>(
            (cells + kTailThreads - 1) / kTailThreads);
        index_select_tail4_kernel<IndexT><<<
            dim3(blocks, static_cast<unsigned>(outer)), kTailThreads, 0, stream>>>(
            cells, row, static_cast<const uint32_t*>(self.data_ptr()),
            index.data_ptr<IndexT>(), static_cast<uint4*>(result.data_ptr()));
        return;
    }
#define TP_IST_CASE(ctype, name) \
    case DType::name: { \
        const unsigned blocks = static_cast<unsigned>( \
            (n_idx + kThreads - 1) / kThreads); \
        index_select_tail_scalar_kernel<ctype, IndexT><<< \
            dim3(blocks, static_cast<unsigned>(outer)), kThreads, 0, stream>>>( \
            n_idx, row, static_cast<const ctype*>(self.data_ptr()), \
            index.data_ptr<IndexT>(), static_cast<ctype*>(result.data_ptr())); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IST_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_IST_CASE)
        TP_IST_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IST_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IST_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IST_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "index_select: unsupported dtype");
    }
#undef TP_IST_CASE
}

template <typename IndexT>
void launch_index_select_for_index(
        int64_t total, int64_t n_idx, int64_t inner, int64_t row,
        int64_t outer, const Tensor& self, const Tensor& index, Tensor& result,
        int slice_threads, cudaStream_t stream) {
#define TP_IS_CASE(ctype, name) \
    case DType::name: { \
        if (use_slice_kernel(inner, n_idx, outer)) { \
            const unsigned index_blocks = static_cast<unsigned>( \
                std::min<int64_t>(n_idx, 2147483647)); \
            const unsigned outer_blocks = static_cast<unsigned>( \
                std::min<int64_t>(outer, 65535)); \
            index_select_slice_kernel<ctype, IndexT><<< \
                dim3(index_blocks, outer_blocks), slice_threads, 0, stream>>>( \
                n_idx, outer, inner, row, \
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

// Both walk (outer, index, inner) over the selected slices: every position
// before `dim` gets its own copy of the slices, so a flat position splits into
// the outer block, the index slot and the column within the slice.
template <typename T>
__global__ void index_copy_kernel(int64_t total, int64_t n_idx, int64_t inner, int64_t row,
                                  T* d, const int64_t* ip, const T* sp) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t block = n_idx * inner;
    for (; t < total; t += stride) {
        int64_t o = t / block, r = t % block;
        int64_t k = r / inner, c = r % inner;
        int64_t iv = ip[k];
        TP_INDEX_RANGE_GUARD(iv, row);
        d[(o * row + iv) * inner + c] = sp[t];
    }
}

// One block column per index entry, mirroring the select kernel: no per-element
// coordinate split, and the index value is used by a whole slice at once.
template <typename T>
__global__ void index_copy_slice_kernel(int64_t n_idx, int64_t outer,
                                        int64_t inner, int64_t row,
                                        T* d, const int64_t* ip, const T* sp) {
    const int64_t k_stride = static_cast<int64_t>(gridDim.x);
    const int64_t o_stride = static_cast<int64_t>(gridDim.y);
    for (int64_t o = blockIdx.y; o < outer; o += o_stride) {
        for (int64_t k = blockIdx.x; k < n_idx; k += k_stride) {
            int64_t iv = ip[k];
            TP_INDEX_RANGE_GUARD(iv, row);
            T* destination = d + (o * row + iv) * inner;
            const T* source = sp + (o * n_idx + k) * inner;
            for (int64_t c = threadIdx.x; c < inner; c += blockDim.x) {
                destination[c] = source[c];
            }
        }
    }
}

template <typename T>
__global__ void index_fill_kernel(int64_t total, int64_t n_idx, int64_t inner, int64_t row,
                                  T* d, const int64_t* ip, T v) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t block = n_idx * inner;
    for (; t < total; t += stride) {
        int64_t o = t / block, r = t % block;
        int64_t k = r / inner, c = r % inner;
        int64_t iv = ip[k];
        TP_INDEX_RANGE_GUARD(iv, row);
        d[(o * row + iv) * inner + c] = v;
    }
}

// One block column per index entry, mirroring the select kernel: no per-element
// coordinate split, and the index value is used by a whole slice at once.
template <typename T>
__global__ void index_fill_slice_kernel(int64_t n_idx, int64_t outer,
                                        int64_t inner, int64_t row,
                                        T* d, const int64_t* ip, T v) {
    const int64_t k_stride = static_cast<int64_t>(gridDim.x);
    const int64_t o_stride = static_cast<int64_t>(gridDim.y);
    for (int64_t o = blockIdx.y; o < outer; o += o_stride) {
        for (int64_t k = blockIdx.x; k < n_idx; k += k_stride) {
            int64_t iv = ip[k];
            TP_INDEX_RANGE_GUARD(iv, row);
            T* destination = d + (o * row + iv) * inner;
            for (int64_t c = threadIdx.x; c < inner; c += blockDim.x) {
                destination[c] = v;
            }
        }
    }
}

// Chunked variants of the slice kernels: a row moves in 16-byte units so each
// thread shifts one vector per step, and a block covers `k_span` consecutive
// entries with `c_span` chunks of each row across its two block dimensions.
// The payload size and pointer alignment that make the casts sound are checked
// before every launch.
//
// The select variant maps (index, chunk block, outer) to the three grid
// dimensions so every block of a chunk wave walks a different source row, and
// gives each block one warp: a block then moves a single 512-byte run of its
// source row, so all of its requests land in one contiguous segment.  The
// output is stored with a streaming hint because this kernel writes it and
// never reads it back.
constexpr int kCellThreads = 32;

template <typename IndexT>
__global__ void index_select_cell_kernel(int64_t cells, int64_t n_idx,
                                         int64_t row,
                                         const uint4* __restrict__ s,
                                         const IndexT* __restrict__ ip,
                                         uint4* __restrict__ d) {
    const int64_t o = blockIdx.z;
    const int64_t k = blockIdx.x;
    int64_t iv = static_cast<int64_t>(ip[k]);
    TP_INDEX_RANGE_GUARD(iv, row);
    const uint4* source = s + (o * row + iv) * cells;
    uint4* destination = d + (o * n_idx + k) * cells;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.y;
    for (int64_t c = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
         c < cells; c += step) {
        __stcs(destination + c, __ldg(source + c));
    }
}

__global__ void index_copy_slice_vec_kernel(int64_t n_idx, int64_t outer,
                                            int64_t chunks, int64_t row,
                                            uint4* d, const int64_t* ip,
                                            const uint4* sp) {
    const int64_t c_span = blockDim.x;
    const int64_t k_span = blockDim.y;
    const int64_t k_stride = static_cast<int64_t>(gridDim.x) * k_span;
    const int64_t o_stride = static_cast<int64_t>(gridDim.y);
    for (int64_t o = blockIdx.y; o < outer; o += o_stride) {
        for (int64_t k = static_cast<int64_t>(blockIdx.x) * k_span + threadIdx.y;
             k < n_idx; k += k_stride) {
            int64_t destination_index = ip[k];
            TP_INDEX_RANGE_GUARD(destination_index, row);
            uint4* destination = d + (o * row + destination_index) * chunks;
            const uint4* source = sp + (o * n_idx + k) * chunks;
            for (int64_t c = threadIdx.x; c < chunks; c += c_span) {
                destination[c] = source[c];
            }
        }
    }
}

template <typename T>
__global__ void index_fill_slice_vec_kernel(int64_t n_idx, int64_t outer,
                                            int64_t chunks, int64_t row,
                                            uint4* d, const int64_t* ip, T v) {
    alignas(16) unsigned char bytes[16];
    const unsigned char* value_bytes =
        reinterpret_cast<const unsigned char*>(&v);
    for (int i = 0; i < 16; i += static_cast<int>(sizeof(T))) {
        for (int b = 0; b < static_cast<int>(sizeof(T)); ++b) {
            bytes[i + b] = value_bytes[b];
        }
    }
    const uint4 pattern = *reinterpret_cast<const uint4*>(bytes);
    const int64_t c_span = blockDim.x;
    const int64_t k_span = blockDim.y;
    const int64_t k_stride = static_cast<int64_t>(gridDim.x) * k_span;
    const int64_t o_stride = static_cast<int64_t>(gridDim.y);
    for (int64_t o = blockIdx.y; o < outer; o += o_stride) {
        for (int64_t k = static_cast<int64_t>(blockIdx.x) * k_span + threadIdx.y;
             k < n_idx; k += k_stride) {
            int64_t destination_index = ip[k];
            TP_INDEX_RANGE_GUARD(destination_index, row);
            uint4* destination = d + (o * row + destination_index) * chunks;
            for (int64_t c = threadIdx.x; c < chunks; c += c_span) {
                destination[c] = pattern;
            }
        }
    }
}

// Launch the chunked select kernel when the row payload divides into 16-byte
// chunks and both ends are 16-byte aligned; returns false so the caller can
// fall back to the scalar kernels.
template <typename IndexT>
bool launch_index_select_slice_vec(int64_t n_idx, int64_t inner, int64_t row,
                                   int64_t outer, const Tensor& self,
                                   const Tensor& index, Tensor& result,
                                   cudaStream_t stream) {
    if (n_idx > 2147483647 || outer > 65535) return false;
    const int64_t row_bytes =
        inner * static_cast<int64_t>(self.itemsize());
    if (row_bytes % 16 != 0) return false;
    if (!vec_slice_ok(self.data_ptr(), result.data_ptr())) return false;
    const int64_t cells = row_bytes / 16;
    if (cells == 0) return true;
    const unsigned chunk_blocks = static_cast<unsigned>(
        std::min<int64_t>((cells + kCellThreads - 1) / kCellThreads, 65535));
    index_select_cell_kernel<IndexT><<<
        dim3(static_cast<unsigned>(n_idx), chunk_blocks,
             static_cast<unsigned>(outer)),
        kCellThreads, 0, stream>>>(
        cells, n_idx, row,
        static_cast<const uint4*>(self.data_ptr()),
        index.data_ptr<IndexT>(), static_cast<uint4*>(result.data_ptr()));
    return true;
}

// Same for the copy kernel: both ends of every move must be 16-byte aligned.
bool launch_index_copy_slice_vec(int64_t n_idx, int64_t inner, int64_t row,
                                 int64_t outer, Tensor& destination,
                                 const Tensor& source, const Tensor& index,
                                 cudaStream_t stream) {
    const int64_t row_bytes =
        inner * static_cast<int64_t>(destination.itemsize());
    if (row_bytes % 16 != 0) return false;
    if (!vec_slice_ok(destination.data_ptr(), source.data_ptr())) return false;
    const int64_t chunks = row_bytes / 16;
    int c_span = 1, k_span = 1;
    vec_slice_block(chunks, c_span, k_span);
    const unsigned index_blocks = static_cast<unsigned>(
        std::min<int64_t>((n_idx + k_span - 1) / k_span, 2147483647));
    const unsigned outer_blocks = static_cast<unsigned>(
        std::min<int64_t>(outer, 65535));
    index_copy_slice_vec_kernel<<<dim3(index_blocks, outer_blocks),
                                  dim3(c_span, k_span), 0, stream>>>(
        n_idx, outer, chunks, row,
        static_cast<uint4*>(destination.data_ptr()),
        index.data_ptr<int64_t>(),
        static_cast<const uint4*>(source.data_ptr()));
    return true;
}

// Same for the fill kernel; the value repeats across the 16-byte pattern.
template <typename T>
bool launch_index_fill_slice_vec(int64_t n_idx, int64_t inner, int64_t row,
                                 int64_t outer, Tensor& destination,
                                 const Tensor& index, T v,
                                 cudaStream_t stream) {
    const int64_t row_bytes =
        inner * static_cast<int64_t>(destination.itemsize());
    if (row_bytes % 16 != 0) return false;
    if (!vec_slice_ok(destination.data_ptr(), nullptr)) return false;
    const int64_t chunks = row_bytes / 16;
    int c_span = 1, k_span = 1;
    vec_slice_block(chunks, c_span, k_span);
    const unsigned index_blocks = static_cast<unsigned>(
        std::min<int64_t>((n_idx + k_span - 1) / k_span, 2147483647));
    const unsigned outer_blocks = static_cast<unsigned>(
        std::min<int64_t>(outer, 65535));
    index_fill_slice_vec_kernel<T><<<dim3(index_blocks, outer_blocks),
                                     dim3(c_span, k_span), 0, stream>>>(
        n_idx, outer, chunks, row,
        static_cast<uint4*>(destination.data_ptr()),
        index.data_ptr<int64_t>(), v);
    return true;
}

#define TP_NZ_ITEMS(ctype) (sizeof(ctype) >= 8 ? 2 : 4)

// nonzero: count the matches in one pass, then a single pass that scans each
// block's tile and writes the matching positions.  Materializing a per-element
// prefix array instead would move three times the input through memory before
// the positions are known.
// One vector load per thread over ITEMS adjacent elements.
template <typename T, int N>
struct NonZeroVec {
    T val[N];
};

template <typename T>
struct NonZeroPredicate {
    __device__ __forceinline__ bool operator()(const T& value) const {
        return value != T(0);
    }
};

// Pass one counts the matches without writing a per-element flag: each block
// reduces its own slice and publishes one value.
template <typename T, int BLOCK, int ITEMS>
__global__ void nonzero_count_kernel(int64_t n, const T* x, int32_t* total,
                                     int32_t* per_block) {
    int mine = 0;
    const int64_t tile_span = static_cast<int64_t>(gridDim.x) * (BLOCK * ITEMS);
    for (int64_t tile = static_cast<int64_t>(blockIdx.x) * (BLOCK * ITEMS);
         tile < n; tile += tile_span) {
        const int64_t base = tile + threadIdx.x;
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int64_t at = base + static_cast<int64_t>(i) * BLOCK;
            if (at < n && NonZeroPredicate<T>()(x[at])) ++mine;
        }
    }
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mine += __shfl_down_sync(0xffffffffffffffffull, mine, off);
    }
    __shared__ int warp_totals[BLOCK / 32];
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_totals[warp] = mine;
    __syncthreads();
    if (threadIdx.x == 0) {
        int sum = 0;
#pragma unroll
        for (int w = 0; w < BLOCK / 32; ++w) sum += warp_totals[w];
        per_block[blockIdx.x] = static_cast<int32_t>(sum);
        if (sum != 0) atomicAdd(total, static_cast<int32_t>(sum));
    }
}

// First level of the offset scan: each block scans one contiguous chunk of
// the per-block counts, leaves the chunk-local prefix in place and publishes
// the chunk total.  The single-block second level below then only has to walk
// one entry per chunk instead of the whole array.
template <int BLOCK>
__global__ void nonzero_block_scan(int32_t blocks, int32_t* per_block,
                                   int32_t* segment_totals) {
    __shared__ int32_t warp_sums[BLOCK / 32];
    const int32_t at = static_cast<int32_t>(blockIdx.x) * BLOCK +
                       static_cast<int32_t>(threadIdx.x);
    const int32_t v = at < blocks ? per_block[at] : 0;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    int32_t inclusive = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int32_t other = __shfl_up_sync(0xffffffffffffffffull, inclusive, off);
        if (lane >= off) inclusive += other;
    }
    if (lane == 31) warp_sums[warp] = inclusive;
    __syncthreads();
    int32_t warp_prefix = 0;
#pragma unroll
    for (int w = 0; w < BLOCK / 32; ++w) {
        if (w == warp) break;
        warp_prefix += warp_sums[w];
    }
    if (at < blocks) per_block[at] = warp_prefix + inclusive - v;
    if (threadIdx.x == BLOCK - 1) {
        segment_totals[blockIdx.x] = warp_prefix + warp_sums[BLOCK / 32 - 1];
    }
}

// Turns the per-block counts into the base slot of each block, so the scan
// pass can write straight into the final positions without a second pass over
// the input.  Runs over the chunk totals produced by nonzero_block_scan and
// leaves their global bases in place.
template <int BLOCK>
__global__ void nonzero_block_offsets(int32_t blocks, int32_t* per_block) {
    __shared__ int32_t carry;
    __shared__ int32_t warp_sums[BLOCK / 32];
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int32_t base = 0; base < blocks; base += BLOCK) {
        const int32_t at = base + static_cast<int32_t>(threadIdx.x);
        const int32_t v = at < blocks ? per_block[at] : 0;
        int32_t incl = v;
        const int lane = threadIdx.x & 31;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int32_t other = __shfl_up_sync(0xffffffffffffffffull, incl, off);
            if (lane >= off) incl += other;
        }
        if (lane == 31) warp_sums[threadIdx.x >> 5] = incl;
        __syncthreads();
        int32_t prefix = 0;
#pragma unroll
        for (int w = 0; w < BLOCK / 32; ++w) {
            if (w == (int)(threadIdx.x >> 5)) break;
            prefix += warp_sums[w];
        }
        if (at < blocks) per_block[at] = carry + prefix + incl - v;
        __syncthreads();
        if (threadIdx.x == BLOCK - 1) carry += prefix + incl;
        __syncthreads();
    }
}

// BLOCK x ITEMS contiguous elements per block.  Each thread owns ITEMS
// *consecutive* elements and loads them as one vector: consecutive lanes then
// cover consecutive vectors, and — because a thread's items are adjacent — the
// slots the block scan hands out are already in ascending order, so the
// coordinate rows come out sorted without a second exchange pass.
template <typename T, int BLOCK, int ITEMS>
__global__ void nonzero_flag_kernel(int64_t n, const T* x, int64_t ndim,
                                    const int64_t* dim_sizes, int64_t* out,
                                    const int32_t* block_base,
                                    const int32_t* segment_base,
                                    int32_t seg_shift) {
    using Vec = NonZeroVec<T, ITEMS>;
    constexpr int kWarps = BLOCK / 32;
    __shared__ int warp_totals[kWarps];
    __shared__ int block_total;
    const int64_t tile_span = static_cast<int64_t>(gridDim.x) * (BLOCK * ITEMS);
    // The base of a block is its segment's global base plus the block's offset
    // inside the segment (the two levels of the offset scan).
    int64_t slot_base =
        static_cast<int64_t>(block_base[blockIdx.x]) +
        static_cast<int64_t>(segment_base[blockIdx.x >> seg_shift]);
    for (int64_t tile = static_cast<int64_t>(blockIdx.x) * (BLOCK * ITEMS);
         tile < n; tile += tile_span) {
    const int64_t at = tile + static_cast<int64_t>(threadIdx.x) * ITEMS;
    int64_t idx[ITEMS];
    bool nz[ITEMS];
    int mine = 0;
    if (at + ITEMS <= n) {
        const Vec loaded = *reinterpret_cast<const Vec*>(x + at);
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            idx[i] = at + i;
            nz[i] = NonZeroPredicate<T>()(loaded.val[i]);
            mine += nz[i] ? 1 : 0;
        }
    } else {
#pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const int64_t pos = at + i;
            idx[i] = pos;
            nz[i] = pos < n && NonZeroPredicate<T>()(x[pos]);
            mine += nz[i] ? 1 : 0;
        }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int inclusive = mine;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int other = __shfl_up_sync(0xffffffffffffffffull, inclusive, off);
        if (lane >= off) inclusive += other;
    }
    if (lane == 31) warp_totals[warp] = inclusive;
    __syncthreads();
    int warp_prefix = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        if (w == warp) break;
        warp_prefix += warp_totals[w];
    }
    if (threadIdx.x == 0) {
        block_total = warp_prefix + warp_totals[kWarps - 1];
    }
    __syncthreads();
    int64_t slot = slot_base + warp_prefix + inclusive - mine;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        if (nz[i]) {
            int64_t rem = idx[i];
            int64_t* row = out + slot * ndim;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                row[d] = rem % dim_sizes[d];
                rem /= dim_sizes[d];
            }
            ++slot;
        }
    }
    __syncthreads();
    slot_base += block_total;
    }
}

}

// ---------------------------------------------------------------------------
// index_select.
// ---------------------------------------------------------------------------

Tensor index_select_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
    if (isQuantizedType(self.dtype())) { quantized::q_scale(self); return quantized::with_quantizer(index_select_cuda(quantized::storage_view(self), dim, index), self); }

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
    const int slice_threads = slice_block_threads(inner);
    if (inner == 1 && outer <= 65535) {
        if (idx.dtype() == DType::Int32) {
            launch_index_select_tail<int32_t>(n_idx, row, outer, self_c, idx,
                                              result, stream);
        } else {
            launch_index_select_tail<int64_t>(n_idx, row, outer, self_c, idx,
                                              result, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (use_slice_kernel(inner, n_idx, outer)) {
        const bool chunked = idx.dtype() == DType::Int32
            ? launch_index_select_slice_vec<int32_t>(
                  n_idx, inner, row, outer, self_c, idx, result, stream)
            : launch_index_select_slice_vec<int64_t>(
                  n_idx, inner, row, outer, self_c, idx, result, stream);
        if (chunked) {
            CUDA_CHECK(cudaGetLastError());
            return result;
        }
    }
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
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    int64_t total = outer * n_idx * inner;
    if (total == 0) return result;
    Tensor source_c = source.contiguous();
    auto stream = getCurrentCUDAStream().stream();
    const int slice_threads = slice_block_threads(inner);
    if (use_slice_kernel(inner, n_idx, outer) &&
        launch_index_copy_slice_vec(n_idx, inner, row, outer, result,
                                    source_c, idx, stream)) {
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
#define TP_IC_CASE(ctype, name) \
    case DType::name: { \
        if (use_slice_kernel(inner, n_idx, outer)) { \
            const unsigned index_blocks = static_cast<unsigned>( \
                std::min<int64_t>(n_idx, 2147483647)); \
            const unsigned outer_blocks = static_cast<unsigned>( \
                std::min<int64_t>(outer, 65535)); \
            index_copy_slice_kernel<ctype><<< \
                dim3(index_blocks, outer_blocks), slice_threads, 0, stream>>>( \
                n_idx, outer, inner, row, static_cast<ctype*>(result.data_ptr()), \
                idx.data_ptr<int64_t>(), static_cast<const ctype*>(source_c.data_ptr())); \
        } else { \
            index_copy_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                total, n_idx, inner, row, static_cast<ctype*>(result.data_ptr()), \
                idx.data_ptr<int64_t>(), static_cast<const ctype*>(source_c.data_ptr())); \
        } \
        break; \
    }
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

Tensor& index_copy_inplace_cuda(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    NoGradGuard __tp_nograd;
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return self;
    int64_t row = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    int64_t total = outer * n_idx * inner;
    if (total == 0) return self;
    if (!self.is_contiguous()) {
        // The kernel addresses the destination as a dense buffer, so a
        // strided destination keeps the out-of-place path.
        self.copy_(index_copy_cuda(self, dim, index, source));
        return self;
    }
    Tensor source_c = source.contiguous();
    auto stream = getCurrentCUDAStream().stream();
    const int slice_threads = slice_block_threads(inner);
    if (use_slice_kernel(inner, n_idx, outer) &&
        launch_index_copy_slice_vec(n_idx, inner, row, outer, self,
                                    source_c, idx, stream)) {
        CUDA_CHECK(cudaGetLastError());
        return self;
    }
#define TP_IC_IP_CASE(ctype, name) \
    case DType::name: { \
        if (use_slice_kernel(inner, n_idx, outer)) { \
            const unsigned index_blocks = static_cast<unsigned>( \
                std::min<int64_t>(n_idx, 2147483647)); \
            const unsigned outer_blocks = static_cast<unsigned>( \
                std::min<int64_t>(outer, 65535)); \
            index_copy_slice_kernel<ctype><<< \
                dim3(index_blocks, outer_blocks), slice_threads, 0, stream>>>( \
                n_idx, outer, inner, row, static_cast<ctype*>(self.data_ptr()), \
                idx.data_ptr<int64_t>(), static_cast<const ctype*>(source_c.data_ptr())); \
        } else { \
            index_copy_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                total, n_idx, inner, row, static_cast<ctype*>(self.data_ptr()), \
                idx.data_ptr<int64_t>(), static_cast<const ctype*>(source_c.data_ptr())); \
        } \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IC_IP_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_IC_IP_CASE)
        TP_IC_IP_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IC_IP_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IC_IP_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IC_IP_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "index_copy: unsupported dtype");
    }
#undef TP_IC_IP_CASE
    CUDA_CHECK(cudaGetLastError());
    return self;
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
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    int64_t total = outer * n_idx * inner;
    if (total == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
    const int slice_threads = slice_block_threads(inner);
#define TP_IF_CASE(ctype, name) \
    case DType::name: { \
        ctype v = value.to<ctype>(); \
        if (use_slice_kernel(inner, n_idx, outer)) { \
            const bool chunked = launch_index_fill_slice_vec<ctype>( \
                n_idx, inner, row, outer, result, idx, v, stream); \
            if (!chunked) { \
                const unsigned index_blocks = static_cast<unsigned>( \
                    std::min<int64_t>(n_idx, 2147483647)); \
                const unsigned outer_blocks = static_cast<unsigned>( \
                    std::min<int64_t>(outer, 65535)); \
                index_fill_slice_kernel<ctype><<< \
                    dim3(index_blocks, outer_blocks), slice_threads, 0, stream>>>( \
                    n_idx, outer, inner, row, static_cast<ctype*>(result.data_ptr()), \
                    idx.data_ptr<int64_t>(), v); \
            } \
        } else { \
            index_fill_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                total, n_idx, inner, row, static_cast<ctype*>(result.data_ptr()), \
                idx.data_ptr<int64_t>(), v); \
        } \
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
    NoGradGuard __tp_nograd;
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return self;
    int64_t row = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    int64_t total = outer * n_idx * inner;
    if (total == 0) return self;
    if (!self.is_contiguous()) {
        // The kernel addresses the destination as a dense buffer, so a
        // strided destination keeps the out-of-place path.
        self.copy_(index_fill_scalar_cuda(self, dim, index, value));
        return self;
    }
    auto stream = getCurrentCUDAStream().stream();
    const int slice_threads = slice_block_threads(inner);
#define TP_IF_IP_CASE(ctype, name) \
    case DType::name: { \
        ctype v = value.to<ctype>(); \
        if (use_slice_kernel(inner, n_idx, outer)) { \
            const bool chunked = launch_index_fill_slice_vec<ctype>( \
                n_idx, inner, row, outer, self, idx, v, stream); \
            if (!chunked) { \
                const unsigned index_blocks = static_cast<unsigned>( \
                    std::min<int64_t>(n_idx, 2147483647)); \
                const unsigned outer_blocks = static_cast<unsigned>( \
                    std::min<int64_t>(outer, 65535)); \
                index_fill_slice_kernel<ctype><<< \
                    dim3(index_blocks, outer_blocks), slice_threads, 0, stream>>>( \
                    n_idx, outer, inner, row, static_cast<ctype*>(self.data_ptr()), \
                    idx.data_ptr<int64_t>(), v); \
            } \
        } else { \
            index_fill_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                total, n_idx, inner, row, static_cast<ctype*>(self.data_ptr()), \
                idx.data_ptr<int64_t>(), v); \
        } \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IF_IP_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_IF_IP_CASE)
        TP_IF_IP_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IF_IP_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IF_IP_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IF_IP_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "index_fill: unsupported dtype");
    }
#undef TP_IF_IP_CASE
    CUDA_CHECK(cudaGetLastError());
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
// nonzero (device count pass followed by an ordered coordinate pass).
// ---------------------------------------------------------------------------

Tensor nonzero_cuda(const Tensor& self) {
    Tensor self_c = self.contiguous();
    const int64_t nd = self.dim();
    const int64_t n = self_c.numel();
    // Empty input: no matches. Launching with a 0-block grid is a CUDA error,
    if (n == 0) {
        return Tensor::zeros({0, nd}, DType::Int64, self.device());
    }
    const auto stream = getCurrentCUDAStream().stream();

    // Pass one counts the matches straight off the input: no per-element flag
    // array is written, so the input is read once and nothing else.
    Tensor count_d = Tensor::zeros({1}, DType::Int32, self.device());
    constexpr int kCountBlock = 256;
    // One tile per block: the output stays in ascending order only when a
    // block's elements are adjacent.  The 1-D grid reaches 2^31-1 blocks, so
    // the launch limit never forces a block to straddle tiles.
    constexpr int64_t kMaxBlocks = 2147483647;
    // Chunk size of the two-level offset scan and its log2, used to index the
    // segment bases from a flat block index.
    constexpr int kScanChunk = 256;
    constexpr int kScanShift = 8;
    static_assert((1 << kScanShift) == kScanChunk, "scan chunk must be a power of two");
    Tensor block_counts;
    Tensor segment_bases;
#define TP_NZC_COUNT(ctype, name)                                                 \
    case DType::name:                                                            \
        {                                                                    \
        constexpr int kItems = TP_NZ_ITEMS(ctype);                            \
        const int64_t tiles =                                                \
            (n + kCountBlock * kItems - 1) / (kCountBlock * kItems);           \
        const unsigned blocks = static_cast<unsigned>(                        \
            std::min<int64_t>(tiles, kMaxBlocks));                            \
        block_counts = Tensor::zeros({blocks}, DType::Int32, self.device());     \
        nonzero_count_kernel<ctype, kCountBlock, kItems>                      \
        <<<blocks, kCountBlock, 0, stream>>>(                                  \
            n, self_c.data_ptr<ctype>(), count_d.data_ptr<int32_t>(),           \
            block_counts.data_ptr<int32_t>());                                   \
        const int64_t segments =                                                \
            (static_cast<int64_t>(blocks) + kScanChunk - 1) / kScanChunk;       \
        segment_bases = Tensor::empty({segments}, DType::Int32, self.device()); \
        nonzero_block_scan<kScanChunk>                                          \
        <<<static_cast<unsigned>(segments), kScanChunk, 0, stream>>>(           \
            static_cast<int32_t>(blocks), block_counts.data_ptr<int32_t>(),     \
            segment_bases.data_ptr<int32_t>());                                 \
        nonzero_block_offsets<kScanChunk><<<1, kScanChunk, 0, stream>>>(        \
            static_cast<int32_t>(segments), segment_bases.data_ptr<int32_t>()); \
        break;                                                                   \
    }
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NZC_COUNT)
        TENSORPLAY_FORALL_FP8_TYPES(TP_NZC_COUNT)
            TP_NZC_COUNT(tensorplay::complex<Half>, ComplexHalf)
            TP_NZC_COUNT(tensorplay::complex<float>, ComplexFloat)
            TP_NZC_COUNT(tensorplay::complex<double>, ComplexDouble)
            TP_NZC_COUNT(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "nonzero: unsupported dtype");
    }
#undef TP_NZC_COUNT
    CUDA_CHECK(cudaGetLastError());
    int32_t count_host = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count_host, count_d.data_ptr<int32_t>(),
                               sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Tensor result = Tensor::empty({count_host, nd}, DType::Int64, self.device());
    if (count_host == 0) return result;

    // Pass two scans each block's tile and expands every match straight into
    // per-axis coordinates: a match's slot is its global ordinal, so the rows
    // are written in place with no intermediate position list.
    std::vector<int64_t> h_sizes(static_cast<std::vector<int64_t>>(self_c.shape()));
    Tensor sizes_d = Tensor::empty({nd}, DType::Int64, self.device());
    CUDA_CHECK(cudaMemcpyAsync(sizes_d.data_ptr<int64_t>(), h_sizes.data(),
                               nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    constexpr int kFlagBlock = 256;
#define TP_NZC_FLAG(ctype, name)                                                  \
    case DType::name:                                                            \
        {                                                                    \
        constexpr int kItems = TP_NZ_ITEMS(ctype);                            \
        const int64_t tiles =                                                \
            (n + kFlagBlock * kItems - 1) / (kFlagBlock * kItems);             \
        const unsigned blocks = static_cast<unsigned>(                        \
            std::min<int64_t>(tiles, kMaxBlocks));                            \
        nonzero_flag_kernel<ctype, kFlagBlock, kItems>                        \
        <<<blocks, kFlagBlock, 0, stream>>>(                                  \
            n, self_c.data_ptr<ctype>(), nd, sizes_d.data_ptr<int64_t>(),      \
            result.data_ptr<int64_t>(), block_counts.data_ptr<int32_t>(),     \
            segment_bases.data_ptr<int32_t>(), kScanShift);                    \
        break;                                                             \
    }
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NZC_FLAG)
        TENSORPLAY_FORALL_FP8_TYPES(TP_NZC_FLAG)
            TP_NZC_FLAG(tensorplay::complex<Half>, ComplexHalf)
            TP_NZC_FLAG(tensorplay::complex<float>, ComplexFloat)
            TP_NZC_FLAG(tensorplay::complex<double>, ComplexDouble)
            TP_NZC_FLAG(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "nonzero: unsupported dtype");
    }
#undef TP_NZC_FLAG
    CUDA_CHECK(cudaGetLastError());
    return result;
}


// take (reshape -> index_select -> reshape).
// ---------------------------------------------------------------------------

Tensor take_cuda(const Tensor& self, const Tensor& index) {
    Tensor flat = self.reshape({self.numel()});
    return index_select_cuda(flat, 0, index.reshape({index.numel()}))
        .reshape(static_cast<std::vector<int64_t>>(index.shape()));
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

TENSORPLAY_LIBRARY_IMPL(CUDA, IndexingKernels) {
    m.impl("index_select", index_select_cuda);
    m.impl("index_copy", index_copy_cuda);
    m.impl("index_copy_", index_copy_inplace_cuda);
    m.impl("index_fill.Tensor", index_fill_tensor_cuda);
    m.impl("index_fill.Scalar", index_fill_scalar_cuda);
    m.impl("index_fill_.Tensor", index_fill_tensor__cuda);
    m.impl("index_fill_.Scalar", index_fill_scalar__cuda);
    m.impl("nonzero", nonzero_cuda);
    m.impl("take", take_cuda);
}

} // namespace cuda

} // namespace tensorplay
