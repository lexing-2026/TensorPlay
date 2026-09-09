// scatter_reduce / index_reduce forward and backward for the CUDA backend.
// Indexed slices are reset to the operation identity before accumulation
// when include_self=false; untouched slices retain their original values.
// Mean divides by full-rank counts, replacing zero counts with one before
// division.  Split from IndexingKernels.cu so reduction-semantic edits do
// not recompile the gather / sort / unique families.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "Context.h"
#include "Utils.h"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition)                                                 \
  do {                                                                        \
    cudaError_t error = condition;                                            \
    if (error != cudaSuccess) {                                               \
      TP_THROW(RuntimeError,                                                  \
               std::string("CUDA Error: ") + cudaGetErrorString(error));      \
    }                                                                         \
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

// ---------------------------------------------------------------------------
// Reduction functors and indexed reduction helpers. With include_self=False,
// indexed slices are reset to the operation identity before accumulation;
// untouched slices retain their original values. Mean divides by full-rank
// counts, replacing zero counts with one before division.
// scatter_reduce_backward / index_reduce_backward.
// ---------------------------------------------------------------------------



enum class SrReduceCuda { Sum, Prod, Mean, AMin, AMax };

SrReduceCuda parse_sr_reduce_cuda(const std::string& r) {
    if (r == "sum") return SrReduceCuda::Sum;
    if (r == "prod") return SrReduceCuda::Prod;
    if (r == "mean") return SrReduceCuda::Mean;
    if (r == "amin") return SrReduceCuda::AMin;
    if (r == "amax") return SrReduceCuda::AMax;
    TP_THROW(ValueError,
             "reduce argument must be one of 'sum', 'prod', 'mean', 'amin', "
             "'amax' but got: " + r);
}

template <typename T>
inline T sr_identity_cuda(SrReduceCuda op) {
    switch (op) {
        case SrReduceCuda::Sum:
        case SrReduceCuda::Mean: return static_cast<T>(0);
        case SrReduceCuda::Prod: return static_cast<T>(1);
        case SrReduceCuda::AMin:
            if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
                return static_cast<T>(std::numeric_limits<float>::infinity());
            } else {
                return std::numeric_limits<T>::has_infinity
                           ? std::numeric_limits<T>::infinity()
                           : std::numeric_limits<T>::max();
            }
        case SrReduceCuda::AMax:
            if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
                return static_cast<T>(-std::numeric_limits<float>::infinity());
            } else {
                return std::numeric_limits<T>::has_infinity
                           ? -std::numeric_limits<T>::infinity()
                           : std::numeric_limits<T>::lowest();
            }
    }
    return static_cast<T>(0);  // unreachable
}

template <typename T>
__device__ __forceinline__ void sr_atomic_reduce(T* addr, T value,
                                                  SrReduceCuda op) {
    switch (op) {
        case SrReduceCuda::Sum:
        case SrReduceCuda::Mean:
            gpuAtomicAdd(addr, value);
            break;
        case SrReduceCuda::Prod:
            gpuAtomicMul(addr, value);
            break;
        case SrReduceCuda::AMin:
            gpuAtomicMin(addr, value);
            break;
        case SrReduceCuda::AMax:
            gpuAtomicMax(addr, value);
            break;
    }
}

template <typename T>
__device__ __forceinline__ T sr_mean_divide_value(T value, int64_t count) {
    if (count == 0 || std::is_same_v<T, bool>) return value;
    if constexpr (std::is_integral_v<T>) {
        const T divisor = static_cast<T>(count);
        T quotient = static_cast<T>(value / divisor);
        if constexpr (std::is_signed_v<T>) {
            const T remainder = static_cast<T>(value - quotient * divisor);
            if (remainder != static_cast<T>(0) && remainder < static_cast<T>(0)) {
                quotient = static_cast<T>(quotient - static_cast<T>(1));
            }
        }
        return quotient;
    } else if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        return static_cast<T>(static_cast<float>(value) /
                              static_cast<float>(count));
    } else {
        return static_cast<T>(value / static_cast<T>(count));
    }
}

template <typename T>
__global__ void sr_mean_divide_kernel(int64_t n, T* data,
                                      const int64_t* counts) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        data[i] = sr_mean_divide_value(data[i], counts[i]);
    }
}

template <bool AllowNegative>
__global__ void sr_validate_indices_kernel(int64_t n, const int64_t* indices,
                                            int64_t size, int32_t* invalid,
                                            int64_t* bad_value) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const int64_t value = indices[i];
        const bool out = AllowNegative
            ? (value < -size || value >= size)
            : (value < 0 || value >= size);
        if (out && atomicCAS(invalid, 0, 1) == 0) {
            *bad_value = value;
        }
    }
}

template <bool AllowNegative>
std::optional<int64_t> sr_validate_indices_cuda(const Tensor& indices,
                                                int64_t size) {
    const int64_t n = indices.numel();
    if (n == 0) return std::nullopt;
    Tensor invalid = Tensor::zeros({1}, DType::Int32, indices.device());
    Tensor bad_value = Tensor::zeros({1}, DType::Int64, indices.device());
    auto stream = getCurrentCUDAStream().stream();
    sr_validate_indices_kernel<AllowNegative>
        <<<static_cast<uint32_t>((n + kThreads - 1) / kThreads), kThreads,
           0, stream>>>(n, indices.data_ptr<int64_t>(), size,
                        invalid.data_ptr<int32_t>(),
                        bad_value.data_ptr<int64_t>());
    CUDA_CHECK(cudaGetLastError());
    int32_t invalid_host = 0;
    CUDA_CHECK(cudaMemcpy(&invalid_host, invalid.data_ptr<int32_t>(),
                          sizeof(invalid_host), cudaMemcpyDeviceToHost));
    if (invalid_host == 0) return std::nullopt;
    int64_t bad_host = 0;
    CUDA_CHECK(cudaMemcpy(&bad_host, bad_value.data_ptr<int64_t>(),
                          sizeof(bad_host), cudaMemcpyDeviceToHost));
    return bad_host;
}

template <typename T>
__global__ void sr_exclude_init_kernel(int64_t total_idx, int64_t idx_dim_size,
                                       int64_t idx_inner, int64_t self_dim_size,
                                       int64_t self_inner, T* d,
                                       const int64_t* ip, T init_v) {
    // One thread per index element: reset that destination slot to the op
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= total_idx) return;
    int64_t rem = flat;
    int64_t oo = rem / (idx_dim_size * idx_inner);
    rem -= oo * idx_dim_size * idx_inner;
    int64_t j = rem % idx_inner;
    int64_t idx = ip[flat];
    if (idx < 0) idx += self_dim_size;
    d[(oo * self_dim_size + idx) * self_inner + j] = init_v;
}

template <typename T>
__global__ void sr_accum_kernel(int64_t total_idx, int64_t idx_dim_size,
                                int64_t idx_inner, int64_t self_dim_size,
                                int64_t self_inner, T* d, const int64_t* ip,
                                const T* vp, int64_t* cp, int op_int) {
    // One thread per index element: one destination element per thread, so
    // collisions go through the atomics exactly like gpu_scatter_reduce.
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= total_idx) return;
    const SrReduceCuda op = static_cast<SrReduceCuda>(op_int);
    int64_t rem = flat;
    int64_t oo = rem / (idx_dim_size * idx_inner);
    rem -= oo * idx_dim_size * idx_inner;
    int64_t j = rem % idx_inner;
    int64_t idx = ip[flat];
    if (idx < 0) idx += self_dim_size;
    const int64_t dst = (oo * self_dim_size + idx) * self_inner + j;
    const T v = vp[flat];
    sr_atomic_reduce(&d[dst], v, op);
    if (op == SrReduceCuda::Mean) {
        gpuAtomicAdd(&cp[dst], static_cast<int64_t>(1));
    }
}

template <typename T>
__global__ void sr_exclude_init_rows_kernel(int64_t total, int64_t K,
                                            int64_t self_dim_size,
                                            int64_t self_inner, T* d,
                                            const int64_t* ip, T init_v) {
    // index_reduce: the index is a vector, so each destination is a whole
    int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; w < total; w += stride) {
        int64_t oo = w / (K * self_inner);
        int64_t rem = w - oo * K * self_inner;
        int64_t j = rem / self_inner;
        int64_t t = rem - j * self_inner;
        d[(oo * self_dim_size + ip[j]) * self_inner + t] = init_v;
    }
}

template <typename T>
__global__ void sr_accum_rows_kernel(int64_t total, int64_t K,
                                     int64_t self_dim_size,
                                     int64_t self_inner, T* d,
                                     const int64_t* ip, const T* vp,
                                     int64_t* cp, int op_int) {
    // One thread per source element; collisions go through atomics exactly
    int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const SrReduceCuda op = static_cast<SrReduceCuda>(op_int);
    for (; w < total; w += stride) {
        int64_t oo = w / (K * self_inner);
        int64_t rem = w - oo * K * self_inner;
        int64_t j = rem / self_inner;
        int64_t t = rem - j * self_inner;
        const int64_t dst = (oo * self_dim_size + ip[j]) * self_inner + t;
        const T v = vp[w];
        sr_atomic_reduce(&d[dst], v, op);
        if (op == SrReduceCuda::Mean) {
            gpuAtomicAdd(&cp[dst], static_cast<int64_t>(1));
        }
    }
}

inline Tensor sr_where_cuda(const Tensor& cond, const Tensor& a,
                            const Tensor& b) {
    return Tensor::where(cond, a, b);
}

} // anonymous namespace

Tensor sr_forward_cuda_impl(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& src_in,
                            const std::string& reduce, bool include_self);

Tensor sr_result_for_backward_cuda(const Tensor& self, int64_t dim,
                                   const Tensor& index, const Tensor& src,
                                   const std::string& reduce,
                                   bool include_self) {
    return sr_forward_cuda_impl(self, dim, index, src, reduce, include_self);
}

Tensor sr_forward_cuda_impl(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& src_in,
                            const std::string& reduce, bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError,
                 "index must have the same number of dimensions as self");
    }
    if (index.numel() != 0 && index.dtype() != DType::Int32 &&
        index.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "scatter_reduce(): Expected dtype int32/int64 for index");
    }
    if (src_in.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "scatter_reduce(): Expected self.dtype to be equal to src.dtype");
    }
    if (index.device() != self.device() || src_in.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "scatter_reduce: self, index, and src must be on the same device");
    }
    if (index.numel() == 0) return ::tensorplay::detail::contiguous_clone(self);
    Tensor idx_c = (index.dtype() == DType::Int64)
                       ? index.contiguous()
                       : index.to(DType::Int64).contiguous();
    std::vector<int64_t> idx_shape(
        static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    if (src_in.dim() == 0) {
        if (nd != 1 || idx_c.size(0) != 1) {
            TP_THROW(RuntimeError,
                     "src/source shape must match the index shape");
        }
        src_b = src_in.expand(idx_shape).contiguous();
    } else {
        if (src_in.dim() != nd) {
            TP_THROW(IndexError,
                     "src/source must have the same number of dimensions as index");
        }
        for (int64_t i = 0; i < nd; ++i) {
            if (i != dim && idx_c.size(i) > self.size(i)) {
                TP_THROW(RuntimeError,
                         "index shape must not exceed self shape outside the reduced dimension");
            }
            if (idx_c.size(i) > src_in.size(i)) {
                TP_THROW(RuntimeError,
                         "index shape must not exceed source shape");
            }
        }
        Tensor src_view = src_in;
        for (int64_t i = 0; i < nd; ++i) {
            if (src_view.size(i) > idx_shape[static_cast<size_t>(i)]) {
                src_view = src_view.narrow(
                    i, 0, idx_shape[static_cast<size_t>(i)]);
            }
        }
        src_b = src_view.contiguous();
    }

    const int64_t idx_inner = [&] {
        int64_t v = 1;
        for (int64_t i = dim + 1; i < nd; ++i) v *= idx_c.size(i);
        return v;
    }();
    const int64_t self_inner = [&] {
        int64_t v = 1;
        for (int64_t i = dim + 1; i < nd; ++i) v *= self.size(i);
        return v;
    }();
    const int64_t idx_dim_size = idx_c.size(dim);
    const int64_t total_idx = idx_c.numel();
    const int64_t self_dim_size = self.size(dim);

    if (auto bad_index = sr_validate_indices_cuda<false>(idx_c, self_dim_size)) {
        TP_THROW(IndexError, "index ", *bad_index,
                 " is out of bounds for dimension ", dim,
                 " with size ", self_dim_size);
    }

    Tensor result = ::tensorplay::detail::contiguous_clone(self);
    Tensor count;
    int64_t* cp = nullptr;
    if (op == SrReduceCuda::Mean) {
        count = Tensor::full(static_cast<std::vector<int64_t>>(self.shape()),
                             include_self ? 1 : 0, DType::Int64,
                             self.device());
        cp = count.data_ptr<int64_t>();
    }

    auto stream = getCurrentCUDAStream().stream();
    const int64_t blocks =
        total_idx > 0 ? (total_idx + kThreads - 1) / kThreads : 1;
    const int64_t result_numel = result.numel();
    const int64_t result_blocks =
        result_numel > 0 ? (result_numel + kThreads - 1) / kThreads : 1;

#define TP_SR_CUDA_CASE(ctype, name)                                            \
    case DType::name: {                                                         \
        ctype* dp = result.data_ptr<ctype>();                                   \
        const int64_t* ip = idx_c.data_ptr<int64_t>();                          \
        const ctype* vp = src_b.data_ptr<ctype>();                              \
        if (!include_self && total_idx > 0 && result.numel() > 0) {              \
            sr_exclude_init_kernel<ctype>                                       \
                <<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(        \
                    total_idx, idx_dim_size, idx_inner, self_dim_size,           \
                    self_inner, dp, ip, sr_identity_cuda<ctype>(op));            \
        }                                                                       \
        if (total_idx > 0) {                                                     \
            sr_accum_kernel<ctype>                                              \
                <<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(        \
                    total_idx, idx_dim_size, idx_inner, self_dim_size,           \
                    self_inner, dp, ip, vp, cp, static_cast<int>(op));            \
        }                                                                       \
        if (op == SrReduceCuda::Mean && result_numel > 0) {                       \
            sr_mean_divide_kernel<ctype>                                         \
                <<<static_cast<uint32_t>(result_blocks), kThreads, 0, stream>>>(  \
                    result_numel, dp, cp);                                       \
        }                                                                       \
        break;                                                                   \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SR_CUDA_CASE)
        default:
            TP_THROW(TypeError, "scatter_reduce: unsupported dtype");
    }
#undef TP_SR_CUDA_CASE
    CUDA_CHECK(cudaGetLastError());

    return result;
}

Tensor scatter_reduce_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                           const Tensor& src, const std::string& reduce,
                           bool include_self) {
    globalContext().alertNotDeterministic("scatter_reduce_cuda");
    return sr_forward_cuda_impl(self, dim, index, src, reduce, include_self);
}

// scatter_reduce this variant takes a 1-D index, a source of self's rank
// (equal sizes except dim == index.numel()), and rejects 'sum'.
Tensor ir_forward_cuda_impl(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& source_in,
                            const std::string& reduce, bool include_self);

Tensor index_reduce_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                         const Tensor& source, const std::string& reduce,
                         bool include_self) {
    globalContext().alertNotDeterministic("index_reduce_cuda");
    return ir_forward_cuda_impl(self, dim, index, source, reduce,
                                include_self);
}

// scatter_reduce this variant takes a 1-D index, a source of self's rank
// (equal sizes except dim == index.numel()), and rejects 'sum'.
Tensor ir_forward_cuda_impl(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& source_in,
                            const std::string& reduce, bool include_self) {
    if (reduce != "prod" && reduce != "mean" && reduce != "amax" &&
        reduce != "amin") {
        TP_THROW(ValueError,
                 "index_reduce(): Expected reduce to be one of prod, mean, "
                 "amax or amin but got ",
                 reduce);
    }
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (nd == 0) {
        TP_THROW(RuntimeError,
                 "index_reduce(): dimension not supported for scalar tensors");
    }
    if (source_in.dim() != nd) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector");
    }
    if (index.dim() > 1) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector, but got dim: ",
                 index.dim());
    }
    if (index.dtype() != DType::Int32 && index.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "index_reduce(): Expected dtype int32/int64 for index");
    }
    if (source_in.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "index_reduce(): Expected self.dtype to be equal to source.dtype");
    }
    if (index.device() != self.device() || source_in.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "index_reduce: self, index, and source must be on the same device");
    }
    for (int64_t i = 0; i < nd; ++i) {
        if (i == dim) continue;
        if (source_in.size(i) != self.size(i)) {
            TP_THROW(IndexError,
                     "index_reduce(): Expected source and self to have the "
                     "same size at dimension ", i);
        }
    }
    Tensor idx_c = (index.dtype() == DType::Int64)
                       ? index.contiguous()
                       : index.to(DType::Int64).contiguous();
    Tensor src_c = (source_in.dtype() == self.dtype())
                       ? source_in.contiguous()
                       : source_in.to(self.dtype()).contiguous();
    const int64_t K = idx_c.numel();
    if (src_c.size(dim) != K) {
        TP_THROW(IndexError,
                 "index_reduce(): Number of indices (", K,
                 ") should be equal to source.size(dim): (", src_c.size(dim),
                 "),");
    }
    const int64_t self_dim_size = self.size(dim);
    if (auto bad_index = sr_validate_indices_cuda<false>(idx_c, self_dim_size)) {
        TP_THROW(IndexError, "index ", *bad_index,
                 " is out of bounds for dimension ", dim,
                 " with size ", self_dim_size);
    }
    const int64_t* ip = idx_c.data_ptr<int64_t>();
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= self.size(i);
    int64_t self_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) self_inner *= self.size(i);

    Tensor result = ::tensorplay::detail::contiguous_clone(self);
    Tensor count;
    int64_t* cp = nullptr;
    if (reduce == "mean") {
        count = Tensor::full(static_cast<std::vector<int64_t>>(self.shape()),
                             include_self ? 1 : 0, DType::Int64,
                             self.device());
        cp = count.data_ptr<int64_t>();
    }

    auto stream = getCurrentCUDAStream().stream();
    const int64_t total = outer * K * self_inner;
    const int64_t blocks =
        total > 0 ? (total + kThreads - 1) / kThreads : 1;
    const int64_t result_numel = result.numel();
    const int64_t result_blocks =
        result_numel > 0 ? (result_numel + kThreads - 1) / kThreads : 1;

#define TP_IR_CUDA_CASE(ctype, name)                                           \
    case DType::name: {                                                         \
        ctype* dp = result.data_ptr<ctype>();                                   \
        const ctype* sp = src_c.data_ptr<ctype>();                              \
        const ctype init_v = sr_identity_cuda<ctype>(op);                        \
        if (!include_self && K > 0 && total > 0) {                               \
            sr_exclude_init_rows_kernel<ctype>                                  \
                <<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(        \
                    total, K, self_dim_size, self_inner, dp, ip, init_v);         \
        }                                                                       \
        if (total > 0) {                                                         \
            sr_accum_rows_kernel<ctype>                                         \
                <<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(        \
                    total, K, self_dim_size, self_inner, dp, ip, sp, cp,          \
                    static_cast<int>(op));                                        \
        }                                                                       \
        if (reduce == "mean" && result_numel > 0) {                               \
            sr_mean_divide_kernel<ctype>                                         \
                <<<static_cast<uint32_t>(result_blocks), kThreads, 0, stream>>>(  \
                    result_numel, dp, cp);                                       \
        }                                                                       \
        break;                                                                   \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IR_CUDA_CASE)
        default:
            TP_THROW(TypeError, "index_reduce: unsupported dtype");
    }
#undef TP_IR_CUDA_CASE
    CUDA_CHECK(cudaGetLastError());

    return result;
}

Tensor scatter_reduce_backward_self_cuda(const Tensor& grad,
                                         const Tensor& self, int64_t dim,
                                         const Tensor& index,
                                         const Tensor& src,
                                         const std::string& reduce,
                                         bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    if (op == SrReduceCuda::Sum) {
        if (!include_self) return grad.scatter(dim, index, Scalar(0));
        return grad;
    }
    if (op == SrReduceCuda::Mean) {
        Tensor N = include_self
                       ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                       : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = N.scatter_add(
            dim, index, Tensor::ones_like(src, src.dtype(), src.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        Tensor gself = grad.div(N);
        if (!include_self) gself = gself.scatter(dim, index, 0.0);
        return gself;
    }
    if (op == SrReduceCuda::AMin || op == SrReduceCuda::AMax) {
        Tensor result = sr_result_for_backward_cuda(self, dim, index, src,
                                                    reduce, include_self);
        Tensor value = result.gather(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor src_is_result = src.eq(value).to(self.dtype());
        Tensor n_dist = self_is_result.scatter_add(dim, index, src_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = self_is_result.mul(distributed);
        if (!include_self) out = out.scatter(dim, index, Scalar(0.0));
        return out;
    }
    // prod
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_result = sr_result_for_backward_cuda(masked_self, dim, index,
                                                       src, reduce,
                                                       include_self);
    Tensor gself = grad.mul(masked_result).div(masked_self);
    if (!include_self) gself = gself.scatter(dim, index, 0.0);
    return gself;
}

Tensor scatter_reduce_backward_src_cuda(const Tensor& grad,
                                        const Tensor& self, int64_t dim,
                                        const Tensor& index,
                                        const Tensor& src,
                                        const std::string& reduce,
                                        bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    if (op == SrReduceCuda::Sum) return grad.gather(dim, index);
    if (op == SrReduceCuda::Mean) {
        Tensor N = include_self
                       ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                       : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = N.scatter_add(
            dim, index, Tensor::ones_like(src, src.dtype(), src.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        return grad.gather(dim, index).div(N.gather(dim, index));
    }
    if (op == SrReduceCuda::AMin || op == SrReduceCuda::AMax) {
        Tensor result = sr_result_for_backward_cuda(self, dim, index, src,
                                                    reduce, include_self);
        Tensor value = result.gather(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor src_is_result = src.eq(value).to(self.dtype());
        Tensor n_dist = self_is_result.scatter_add(dim, index, src_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = src_is_result.mul(distributed.gather(dim, index));
        // The source gradient is defined for every selected source entry.
        return out;
    }
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_self_result = sr_result_for_backward_cuda(
        masked_self, dim, index, src, reduce, include_self);
    Tensor src_zero = src.eq(0);
    Tensor num_zeros = Tensor::zeros_like(self, self.dtype(), self.device())
                           .scatter_add(dim, index, src_zero.to(self.dtype()))
                           .gather(dim, index);
    Tensor single_zero = src_zero.bitwise_and(num_zeros.eq(1));
    Tensor masked_src = src.masked_fill(single_zero, 1.0);
    Tensor masked_src_result = sr_result_for_backward_cuda(
        self, dim, index, masked_src, reduce, include_self);
    Tensor result = sr_result_for_backward_cuda(self, dim, index, src, reduce,
                                                include_self);
    Tensor gsrc = sr_where_cuda(
        single_zero,
        grad.mul(masked_src_result).gather(dim, index),
        grad.mul(result).gather(dim, index).div(src.masked_fill(src_zero, 1.0)));
    return gsrc;
}

Tensor index_reduce_backward_self_cuda(const Tensor& grad,
                                       const Tensor& self, int64_t dim,
                                       const Tensor& index,
                                       const Tensor& source,
                                       const std::string& reduce,
                                       bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    // Exclude the original values from the gradient when requested.
    if (op == SrReduceCuda::Sum) {
        if (!include_self) return grad.index_fill(dim, index, Scalar(0));
        return grad;
    }
    if (op == SrReduceCuda::Mean) {
        Tensor N = include_self
                       ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                       : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = Tensor::index_add(N, dim, index,
                              Tensor::ones_like(source, source.dtype(),
                                                source.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        Tensor gself = grad.div(N);
        if (!include_self) gself = gself.index_fill(dim, index, 0.0);
        return gself;
    }
    Tensor result = index_reduce_cuda(self, dim, index, source, reduce,
                                      include_self);
    if (op == SrReduceCuda::AMin || op == SrReduceCuda::AMax) {
        Tensor value = result.index_select(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor source_is_result = source.eq(value).to(self.dtype());
        Tensor n_dist = Tensor::index_add(self_is_result, dim, index,
                                          source_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = self_is_result.mul(distributed);
        if (!include_self) out = out.index_fill(dim, index, Scalar(0.0));
        return out;
    }
    // prod
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_result = index_reduce_cuda(masked_self, dim, index, source,
                                             reduce, include_self);
    Tensor gself = grad.mul(masked_result).div(masked_self);
    if (!include_self) gself = gself.index_fill(dim, index, 0.0);
    return gself;
}

Tensor index_reduce_backward_src_cuda(const Tensor& grad,
                                      const Tensor& self, int64_t dim,
                                      const Tensor& index,
                                      const Tensor& source,
                                      const std::string& reduce,
                                      bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    if (op == SrReduceCuda::Sum) return grad.index_select(dim, index);
    if (op == SrReduceCuda::Mean) {
        Tensor N = include_self
                       ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                       : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = Tensor::index_add(N, dim, index,
                              Tensor::ones_like(source, source.dtype(),
                                                source.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        return grad.index_select(dim, index).div(N.index_select(dim, index));
    }
    Tensor result = index_reduce_cuda(self, dim, index, source, reduce,
                                      include_self);
    if (op == SrReduceCuda::AMin || op == SrReduceCuda::AMax) {
        Tensor value = result.index_select(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor source_is_result = source.eq(value).to(self.dtype());
        Tensor n_dist = Tensor::index_add(self_is_result, dim, index,
                                          source_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = source_is_result.mul(distributed.index_select(dim, index));
        return out;
    }
    // prod
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_self_result = index_reduce_cuda(masked_self, dim, index,
                                                  source, reduce,
                                                  include_self);
    Tensor src_zero = source.eq(0);
    Tensor num_zeros =
        Tensor::zeros_like(self, self.dtype(), self.device())
            .index_add(dim, index, src_zero.to(self.dtype()))
            .index_select(dim, index);
    Tensor single_zero = src_zero.bitwise_and(num_zeros.eq(1));
    Tensor masked_source = source.masked_fill(single_zero, 1.0);
    Tensor masked_result = index_reduce_cuda(self, dim, index, masked_source,
                                             reduce, include_self);
    Tensor gsrc = sr_where_cuda(
        single_zero,
        grad.mul(masked_result).index_select(dim, index),
        grad.mul(result).index_select(dim, index).div(
            source.masked_fill(src_zero, 1.0)));
    return gsrc;
}

Tensor& interop_index_reduce_out_cuda(const Tensor& self, int64_t dim, const Tensor& index,
              const Tensor& source, std::string reduce, bool include_self,
              Tensor& out) {
        out = index_reduce_cuda(self, dim, index, source, reduce, include_self);
        return out;

}

} // namespace


} // namespace

namespace tensorplay {
namespace cuda {

TENSORPLAY_LIBRARY_IMPL(CUDA, ScatterReduceKernels) {
    m.impl("scatter_reduce", scatter_reduce_cuda);
    m.impl("index_reduce", index_reduce_cuda);
    m.impl("_scatter_reduce_backward_self", scatter_reduce_backward_self_cuda);
    m.impl("_scatter_reduce_backward_src", scatter_reduce_backward_src_cuda);
    m.impl("_index_reduce_backward_self", index_reduce_backward_self_cuda);
    m.impl("_index_reduce_backward_src", index_reduce_backward_src_cuda);
    m.impl("index_reduce.out", interop_index_reduce_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
