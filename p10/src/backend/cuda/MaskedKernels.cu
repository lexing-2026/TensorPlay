#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "CUDARuntime.h"
#include "GPUPrimitives.cuh"
#include "CUDALoops.cuh"
#include "OutWrite.h"

#include <cuda_runtime.h>

#include <cassert>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kThreads = 256;

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

inline std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                             const std::vector<int64_t>& b) {
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

// One block per (batch, row) and one thread per column: the mask only depends
// on the column, so the row can stay block-wide and the whole grid runs at
// element granularity.  A thread per row instead would leave the machine idle
// on all but as many blocks as there are rows, and every column a serial
// per-thread loop.
template <typename T, bool Lower>
__global__ void triangular_mask_kernel(int64_t batch_rows, int64_t rows, int64_t cols,
                                       const T* in, T* out, int64_t diagonal) {
    const int64_t br = static_cast<int64_t>(blockIdx.x);
    const int64_t bi = br / rows;
    const int64_t row = br - bi * rows;
    const int64_t base = bi * rows * cols + row * cols;
    const int64_t limit = row + diagonal;
    const int64_t step = static_cast<int64_t>(gridDim.y) * blockDim.x;
    for (int64_t c = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
         c < cols; c += step) {
        // A masked-out element is a pure store: branching keeps the load out
        // of the fully masked half of the matrix, which is half the traffic.
        if (Lower ? (c > limit) : (c < limit)) {
            out[base + c] = static_cast<T>(0);
        } else {
            out[base + c] = in[base + c];
        }
    }
}

// Short rows: one thread per row walking its few columns.  Handing a whole row
// to a block would leave most of its threads idle when the row is narrower than
// the block, and the parallelism is already there in the row count.
template <typename T, bool Lower>
__global__ void triangular_mask_rows_kernel(int64_t batch_rows, int64_t rows,
                                            int64_t cols, const T* in, T* out,
                                            int64_t diagonal) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < batch_rows; t += stride) {
        const int64_t bi = t / rows;
        const int64_t row = t - bi * rows;
        const int64_t base = bi * rows * cols + row * cols;
        const int64_t limit = row + diagonal;
        for (int64_t c = 0; c < cols; ++c) {
            if (Lower ? (c > limit) : (c < limit)) {
                out[base + c] = static_cast<T>(0);
            } else {
                out[base + c] = in[base + c];
            }
        }
    }
}

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
        (void)selected;
    }
}

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
    // A row narrower than a wave is cheaper with one thread per row; wider rows
    // hand the columns to the threads of a block so every element gets its own
    // lane.
    // A row narrower than a wave cannot keep a block busy anyway; past that
    // the block-per-row shape pays off and stays the better mapping.
    constexpr int64_t kRowPerThreadMax = 32;
    if (cols <= kRowPerThreadMax) {
        const dim3 grid(static_cast<unsigned>((work + kThreads - 1) / kThreads));
#define TP_TRI_ROW_CASE(ctype, name) \
        case DType::name: \
            triangular_mask_rows_kernel<ctype, Lower><<<grid, kThreads, 0, stream>>>( \
                work, rows, cols, self_c.data_ptr<ctype>(), result.data_ptr<ctype>(), \
                diagonal); \
            break;
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_TRI_ROW_CASE)
            default: TP_THROW(TypeError, "tril/triu: unsupported dtype");
        }
#undef TP_TRI_ROW_CASE
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    // Rows across grid.x, column tiles across grid.y (the y extent is the
    // smaller hardware limit, and a wide row simply loops).
    const unsigned column_tiles = static_cast<unsigned>(std::min<int64_t>(
        (cols + kThreads - 1) / kThreads, 65535));
    const dim3 grid(static_cast<unsigned>(work), column_tiles);
    const dim3 block(kThreads);
#define TP_TRI_CASE(ctype, name) \
    case DType::name: \
        triangular_mask_kernel<ctype, Lower><<<grid, block, 0, stream>>>( \
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

}

Tensor masked_fill_cuda(const Tensor& self, const Tensor& mask, const Scalar& value) {
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

Tensor tril_cuda(const Tensor& self, int64_t diagonal) {
    return triangular_mask_entry<true>(self, diagonal);
}

Tensor triu_cuda(const Tensor& self, int64_t diagonal) {
    return triangular_mask_entry<false>(self, diagonal);
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

TENSORPLAY_LIBRARY_IMPL(CUDA, MaskedKernels) {
    m.impl("masked_fill", masked_fill_cuda);
    m.impl("masked_fill_", masked_fill__cuda);
    m.impl("masked_fill.Tensor", masked_fill_tensor_cuda);
    m.impl("masked_fill_.Tensor", masked_fill_tensor__cuda);
    m.impl("tril", tril_cuda);
    m.impl("triu", triu_cuda);
    m.impl("masked_scatter", masked_scatter_cuda);
    m.impl("tril.out", interop_tril_out_cuda);
    m.impl("triu.out", interop_triu_out_cuda);
    m.impl("masked_fill_.Scalar", interop_masked_fill__Scalar_cuda);
}

}
}
