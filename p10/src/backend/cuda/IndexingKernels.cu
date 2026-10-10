// High-throughput indexing kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "CUDARuntime.h"

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
    if ((iv) < -(row) || (iv) >= (row)) { __trap(); } \
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

// Turns the per-block counts into the base slot of each block, so the scan
// pass can write straight into the final positions without a second pass over
// the input.
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
// positions come out sorted without a second exchange pass.
template <typename T, int BLOCK, int ITEMS>
__global__ void nonzero_flag_kernel(int64_t n, const T* x, int64_t* flat,
                                    const int32_t* block_base) {
    using Vec = NonZeroVec<T, ITEMS>;
    constexpr int kWarps = BLOCK / 32;
    __shared__ int warp_totals[kWarps];
    __shared__ int block_total;
    const int64_t tile_span = static_cast<int64_t>(gridDim.x) * (BLOCK * ITEMS);
    int64_t slot_base = block_base[blockIdx.x];
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
    int slot = slot_base + warp_prefix + inclusive - mine;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        if (nz[i]) flat[slot++] = idx[i];
    }
    __syncthreads();
    slot_base += block_total;
    }
}

__global__ void nonzero_write_indices(int64_t count, int64_t ndim,
                                      const int64_t* sizes, const int64_t* flat,
                                      int64_t* out) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t m = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         m < count; m += stride) {
        int64_t rem = flat[m];
        int64_t* row = out + m * ndim;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            const int64_t size = sizes[d];
            row[d] = rem % size;
            rem /= size;
        }
    }
}
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
#define TP_IC_CASE(ctype, name) \
    case DType::name: \
        index_copy_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total, n_idx, inner, row, static_cast<ctype*>(result.data_ptr()), \
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
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    int64_t total = outer * n_idx * inner;
    if (total == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
#define TP_IF_CASE(ctype, name) \
    case DType::name: { \
        ctype v = value.to<ctype>(); \
        index_fill_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total, n_idx, inner, row, static_cast<ctype*>(result.data_ptr()), \
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
    Tensor block_counts;
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
        nonzero_block_offsets<256><<<1, 256, 0, stream>>>(                      \
            static_cast<int32_t>(blocks), block_counts.data_ptr<int32_t>());    \
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
    Tensor result = Tensor::zeros({count_host, nd}, DType::Int64, self.device());
    if (count_host == 0) return result;

    // Pass two scans each block's tile and writes the matching positions.
    Tensor flat = Tensor::empty({count_host}, DType::Int64, self.device());
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
            n, self_c.data_ptr<ctype>(), flat.data_ptr<int64_t>(),                 \
            block_counts.data_ptr<int32_t>());                               \
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

    // Pass three expands the positions into per-axis coordinates.
    std::vector<int64_t> h_sizes(static_cast<std::vector<int64_t>>(self_c.shape()));
    Tensor sizes_d = Tensor::empty({nd}, DType::Int64, self.device());
    CUDA_CHECK(cudaMemcpyAsync(sizes_d.data_ptr<int64_t>(), h_sizes.data(),
                               nd * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    const int64_t expand_blocks = std::min<int64_t>(
        (count_host + kThreads - 1) / kThreads, 65535);
    nonzero_write_indices<<<static_cast<unsigned>(expand_blocks), kThreads, 0, stream>>>(
        static_cast<int64_t>(count_host), nd, sizes_d.data_ptr<int64_t>(),
        flat.data_ptr<int64_t>(), result.data_ptr<int64_t>());
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
    m.impl("index_fill.Tensor", index_fill_tensor_cuda);
    m.impl("index_fill.Scalar", index_fill_scalar_cuda);
    m.impl("index_fill_.Tensor", index_fill_tensor__cuda);
    m.impl("index_fill_.Scalar", index_fill_scalar__cuda);
    m.impl("nonzero", nonzero_cuda);
    m.impl("take", take_cuda);
}

} // namespace cuda

} // namespace tensorplay
