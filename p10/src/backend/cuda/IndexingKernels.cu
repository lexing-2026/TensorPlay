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
