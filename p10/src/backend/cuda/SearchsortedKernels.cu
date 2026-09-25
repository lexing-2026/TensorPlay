#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "Bucketization.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
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

template <typename T>
__global__ void searchsorted_kernel(int64_t n, int64_t seq_len, bool right,
                                    int64_t idim_in, bool is_1d_boundaries,
                                    const T* sp, const T* vp, int64_t* rp) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const T v = vp[i];
        int64_t lo = is_1d_boundaries ? 0 : i / idim_in * seq_len;
        int64_t hi = lo + seq_len;
        const int64_t base = lo;
        while (lo < hi) {
            const int64_t mid = lo + ((hi - lo) >> 1);
            const bool go_right = right ? !(sp[mid] > v) : !(sp[mid] >= v);
            if (go_right) lo = mid + 1; else hi = mid;
        }
        rp[i] = lo - base;
    }
}

template <typename T>
__global__ void sorter_gather_kernel(int64_t n, const T* src,
                                     const int64_t* sorter, T* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = src[sorter[i]];
}

Tensor searchsorted_apply_sorter_cuda(const Tensor& boundaries,
                                      const Tensor& sorter) {
    Tensor sorted = Tensor::empty(
        static_cast<std::vector<int64_t>>(boundaries.shape()),
        boundaries.dtype(), boundaries.device());
    Tensor seq_c = boundaries.contiguous();
    Tensor sorter_c = sorter.contiguous();
    const int64_t n = seq_c.numel();
    if (n == 0) return sorted;
    auto stream = getCurrentCUDAStream().stream();
#define TP_SS_SORTER_CASE(ctype, name) \
    case DType::name: \
        sorter_gather_kernel<ctype><<<(n + kThreads - 1) / kThreads, \
                                      kThreads, 0, stream>>>( \
            n, seq_c.data_ptr<ctype>(), sorter_c.data_ptr<int64_t>(), \
            sorted.data_ptr<ctype>()); \
        break;
    switch (seq_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SS_SORTER_CASE)
        default: TP_THROW(TypeError,
                          "searchsorted(): unsupported boundaries dtype ",
                          toString(seq_c.dtype()));
    }
#undef TP_SS_SORTER_CASE
    CUDA_CHECK(cudaGetLastError());
    return sorted;
}

Tensor searchsorted_impl_cuda(const Tensor& seq_f, const Tensor& vals_f,
                              bool out_int32, bool right) {
    Tensor seq = seq_f.contiguous();
    Tensor vals = vals_f.contiguous();
    int64_t seq_len = seq.size(-1);
    const bool is_1d_boundaries = seq.dim() == 1;
    const int64_t idim_in =
        (vals.dim() == 0 && vals.numel() == 1) ? 1 : vals.size(-1);
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(vals.shape()),
                                  out_int32 ? DType::Int32 : DType::Int64,
                                  vals.device());
    int64_t n = vals.numel();
    if (n == 0) return result;
    auto stream = getCurrentCUDAStream().stream();

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        if (out_int32) {
            Tensor tmp = Tensor::empty(
                static_cast<std::vector<int64_t>>(vals.shape()),
                DType::Int64, vals.device());
            searchsorted_kernel<T><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, seq_len, right, idim_in, is_1d_boundaries,
                seq.data_ptr<T>(), vals.data_ptr<T>(), tmp.data_ptr<int64_t>());
            CUDA_CHECK(cudaGetLastError());
            return tmp.to(DType::Int32);
        }
        searchsorted_kernel<T><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, seq_len, right, idim_in, is_1d_boundaries,
            seq.data_ptr<T>(), vals.data_ptr<T>(), result.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());
        return result;
    };

#define TP_SS_CASE(ctype, name) \
    case DType::name: return run(ctype{});
    switch (vals.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SS_CASE)
        default:
            TP_THROW(TypeError, "searchsorted: unsupported dtype ",
                    toString(vals.dtype()));
    }
#undef TP_SS_CASE
    return result;
}

}

Tensor& searchsorted_out_cuda_impl(const Tensor& sorted_sequence,
                                   const Tensor& self, bool out_int32,
                                   bool right,
                                   const std::optional<std::string>& side_opt,
                                   const Tensor& sorter_opt, Tensor& result) {
    bucketization::pre_check(sorted_sequence, self, result, out_int32, right,
                             side_opt, sorter_opt);
    result.resize_(static_cast<std::vector<int64_t>>(self.shape()));
    const bool is_right = side_opt.has_value() ? *side_opt == "right" : right;
    if (self.numel() == 0) return result;

    Tensor seq = sorted_sequence;
    Tensor sorter = sorter_opt;
    if (sorter.defined()) {
        seq = searchsorted_apply_sorter_cuda(seq, sorter);
    }

    Tensor vals = self;
    Tensor trimmed_input, trimmed_boundaries;
    bucketization::maybe_trim_input_tensors(trimmed_input, trimmed_boundaries,
                                            vals, seq);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : vals;
    const Tensor& final_boundaries =
        trimmed_boundaries.defined() ? trimmed_boundaries : seq;
    Tensor computed = searchsorted_impl_cuda(final_boundaries, final_input,
                                             out_int32, is_right);
    if (&result != &computed) {
        result.copy_(computed);
    }
    return result;
}

Tensor& searchsorted_out_cuda(const Tensor& sorted_sequence, const Tensor& self,
                              bool out_int32, bool right,
                              const std::optional<std::string>& side_opt,
                              const std::optional<Tensor>& sorter_opt,
                              Tensor& result) {
    return searchsorted_out_cuda_impl(
        sorted_sequence, self, out_int32, right, side_opt,
        sorter_opt.value_or(Tensor()), result);
}

Tensor searchsorted_cuda(const Tensor& sorted_sequence, const Tensor& self,
                         bool out_int32, bool right,
                         const std::optional<std::string>& side_opt,
                         const std::optional<Tensor>& sorter_opt) {
    Tensor result = Tensor::empty(
        {}, out_int32 ? DType::Int32 : DType::Int64, self.device());
    searchsorted_out_cuda_impl(sorted_sequence, self, out_int32, right,
                               side_opt, sorter_opt.value_or(Tensor()),
                               result);
    return result;
}

Tensor& searchsorted_scalar_out_cuda(const Tensor& sorted_sequence,
                                     const Scalar& self, bool out_int32,
                                     bool right,
                                     const std::optional<std::string>& side_opt,
                                     const std::optional<Tensor>& sorter_opt,
                                     Tensor& result) {
    Tensor scalar_tensor =
        bucketization::scalar_tensor(self, sorted_sequence.device());
    return searchsorted_out_cuda_impl(
        sorted_sequence, scalar_tensor, out_int32, right, side_opt,
        sorter_opt.value_or(Tensor()), result);
}

Tensor searchsorted_scalar_cuda(const Tensor& sorted_sequence, const Scalar& self,
                                bool out_int32, bool right,
                                const std::optional<std::string>& side_opt,
                                const std::optional<Tensor>& sorter_opt) {
    Tensor result = Tensor::empty(
        {}, out_int32 ? DType::Int32 : DType::Int64, sorted_sequence.device());
    searchsorted_scalar_out_cuda(sorted_sequence, self, out_int32, right,
                                 side_opt, sorter_opt.value_or(Tensor()),
                                 result);
    return result;
}

Tensor& bucketize_out_cuda(const Tensor& self, const Tensor& boundaries,
                           bool out_int32, bool right, Tensor& result) {
    TP_CHECK(boundaries.dim() == 1,
             "bucketize(): boundaries tensor must be 1 dimension, but got dim(",
             boundaries.dim(), ")");
    return searchsorted_out_cuda_impl(boundaries, self, out_int32, right,
                                      std::nullopt, Tensor(), result);
}

Tensor bucketize_cuda(const Tensor& self, const Tensor& boundaries,
                      bool out_int32, bool right) {
    Tensor result = Tensor::empty(
        {}, out_int32 ? DType::Int32 : DType::Int64, self.device());
    bucketize_out_cuda(self, boundaries, out_int32, right, result);
    return result;
}

Tensor& bucketize_scalar_out_cuda(const Scalar& self, const Tensor& boundaries,
                                  bool out_int32, bool right, Tensor& result) {
    Tensor scalar_tensor =
        bucketization::scalar_tensor(self, boundaries.device());
    return bucketize_out_cuda(scalar_tensor, boundaries, out_int32, right,
                              result);
}

Tensor bucketize_scalar_cuda(const Scalar& self, const Tensor& boundaries,
                             bool out_int32, bool right) {
    Tensor result = Tensor::empty(
        {}, out_int32 ? DType::Int32 : DType::Int64, boundaries.device());
    bucketize_scalar_out_cuda(self, boundaries, out_int32, right, result);
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, SearchsortedKernels) {
    m.impl("searchsorted.Tensor", searchsorted_cuda);
    m.impl("searchsorted.Tensor_out", searchsorted_out_cuda);
    m.impl("searchsorted.Scalar", searchsorted_scalar_cuda);
    m.impl("searchsorted.Scalar_out", searchsorted_scalar_out_cuda);
    m.impl("bucketize.Tensor", bucketize_cuda);
    m.impl("bucketize.Tensor_out", bucketize_out_cuda);
    m.impl("bucketize.Scalar", bucketize_scalar_cuda);
    m.impl("bucketize.Scalar_out", bucketize_scalar_out_cuda);
}

}
}
