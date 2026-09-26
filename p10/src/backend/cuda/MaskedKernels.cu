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

// Shape and strides of one operand, carried in the parameter block: the kernel
// walks a flat index and needs both to turn it into an offset.
struct TriTensorInfo {
    static constexpr int kMaxDims = 12;
    int ndim = 0;
    int64_t sizes[kMaxDims]{};
    int64_t strides[kMaxDims]{};
};

TriTensorInfo make_tri_info(const Tensor& tensor) {
    TriTensorInfo info;
    info.ndim = static_cast<int>(tensor.dim());
    if (info.ndim > TriTensorInfo::kMaxDims) {
        TP_THROW(RuntimeError, "tril/triu: tensor rank exceeds ",
                 TriTensorInfo::kMaxDims, " dimensions on CUDA");
    }
    for (int d = 0; d < info.ndim; ++d) {
        info.sizes[d] = tensor.size(d);
        info.strides[d] = tensor.stride(d);
    }
    return info;
}

constexpr int kTriBlockSize = 128;

// The grid walks the tensor as one flat run.  A thread takes a group of
// consecutive elements along the last axis, so one round of index arithmetic
// covers the whole group, and a group that falls entirely on one side of the
// diagonal is filled without ever touching the input.
template <typename T, typename IndexT, bool Upper, int ElementsPerThread>
__global__ void __launch_bounds__(kTriBlockSize) triangular_mask_kernel(
    TriTensorInfo result_info, const TriTensorInfo self_info, const T* self_data,
    T* result_data, const int64_t k, const int64_t N_padded,
    const IndexT last_dim_padded) {
    const int64_t dims = self_info.ndim;
    int64_t linear_idx =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
        ElementsPerThread;
    if (linear_idx >= N_padded) return;

    // Column and row of the group's first element.
    IndexT col = static_cast<IndexT>(linear_idx % last_dim_padded);
    linear_idx /= last_dim_padded;
    IndexT row = static_cast<IndexT>(linear_idx % self_info.sizes[dims - 2]);

    int64_t self_offset = 0, result_offset = 0;
    self_offset += self_info.strides[dims - 1] * col;
    result_offset += result_info.strides[dims - 1] * col;
    linear_idx /= self_info.sizes[dims - 2];
    self_offset += self_info.strides[dims - 2] * row;
    result_offset += result_info.strides[dims - 2] * row;

    IndexT running_index;
#pragma unroll
    for (IndexT i = dims - 3; i >= 0; --i) {
        running_index = static_cast<IndexT>(linear_idx % self_info.sizes[i]);
        linear_idx /= self_info.sizes[i];
        self_offset += running_index * self_info.strides[i];
        result_offset += running_index * result_info.strides[i];
    }

    const int64_t last = self_info.sizes[dims - 1];
    T frag[ElementsPerThread] = {};
    // True when at least one element of the group survives the mask, which is
    // when the input has to be read at all.
    const bool has_mask =
        (Upper && col + ElementsPerThread - row >= k) ||
        (!Upper && col - row <= k);
    if (has_mask) {
#pragma unroll
        for (int i = 0; i < ElementsPerThread && col + i < last; ++i) {
            frag[i] = self_data[self_offset + i * self_info.strides[dims - 1]];
        }
#pragma unroll
        for (int i = 0; i < ElementsPerThread; ++i) {
            const bool mask =
                Upper ? (col + i - row >= k) : (col + i - row <= k);
            frag[i] = mask ? frag[i] : static_cast<T>(0);
        }
    }
#pragma unroll
    for (int i = 0; i < ElementsPerThread && col + i < last; ++i) {
        result_data[result_offset + i * result_info.strides[dims - 1]] = frag[i];
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
    const int64_t ndim = self.dim();
    if (ndim < 2) {
        TP_THROW(RuntimeError, "tril/triu requires tensor with at least 2 dimensions");
    }
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                  self.dtype(), self.device());
    const int64_t numel = self.numel();
    if (numel == 0) return result;
    const TriTensorInfo self_info = make_tri_info(self);
    const TriTensorInfo result_info = make_tri_info(result);
    // A group of consecutive elements can reach past the end of a row, so the
    // row length is rounded up to a whole number of groups and the grid covers
    // that padded extent; the tail beyond the row is never written.
    constexpr int64_t kGroupBytes = 8;
    auto stream = getCurrentCUDAStream().stream();
    const dim3 block(kTriBlockSize);
#define TP_TRI_CASE(ctype, name)                                                  \
    case DType::name: {                                                            \
        constexpr int kEpt =                                                       \
            sizeof(ctype) < kGroupBytes ? kGroupBytes / sizeof(ctype) : 1;         \
        const int64_t last_dim_padded =                                             \
            ((self.size(ndim - 1) + kEpt - 1) / kEpt) * kEpt;                     \
        const int64_t n_padded = (numel / self.size(ndim - 1)) * last_dim_padded;   \
        const dim3 grid(static_cast<unsigned>(                                      \
            (n_padded / kEpt + block.x - 1) / block.x));                           \
        if (numel <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {   \
            triangular_mask_kernel<ctype, int32_t, !Lower, kEpt>                    \
                <<<grid, block, 0, stream>>>(result_info, self_info,                 \
                                              self.data_ptr<ctype>(),               \
                                              result.data_ptr<ctype>(), diagonal,   \
                                              n_padded,                             \
                                              static_cast<int32_t>(last_dim_padded));\
        } else {                                                                    \
            triangular_mask_kernel<ctype, int64_t, !Lower, kEpt>                    \
                <<<grid, block, 0, stream>>>(result_info, self_info,                 \
                                              self.data_ptr<ctype>(),               \
                                              result.data_ptr<ctype>(), diagonal,   \
                                              n_padded, last_dim_padded);            \
        }                                                                           \
        break;                                                                      \
    }
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
