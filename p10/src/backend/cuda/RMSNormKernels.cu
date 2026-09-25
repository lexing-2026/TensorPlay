#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {

static void check_normalization_cuda_status(cudaError_t error,
                                           const char* operation) {
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string(operation) + ": " +
                                   cudaGetErrorString(error));
    }
}

static void check_normalization_cuda_launch(const char* operation) {
    check_normalization_cuda_status(cudaGetLastError(), operation);
}

namespace {

template <typename T, typename ACC>
__global__ void rms_norm_row_kernel(const T* __restrict__ x,
                                    const T* __restrict__ w,
                                    T* __restrict__ out,
                                    int64_t inner, double eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const T* xr = x + row * inner;
    T* orow = out + row * inner;

    ACC local = ACC(0);
    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const ACC v = static_cast<ACC>(xr[j]);
        local += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffffffffffffull, local, off);
    __shared__ ACC warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = local;
    __syncthreads();
    const int nwarps = (static_cast<int>(blockDim.x) + 31) >> 5;
    ACC total = ACC(0);
    if (wid == 0) {
        total = (lane < nwarps) ? warp_sums[lane] : ACC(0);
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_down_sync(0xffffffffffffffffull, total, off);
    }
    __shared__ ACC s_inv;
    if (threadIdx.x == 0)
        s_inv = static_cast<ACC>(rsqrt(static_cast<double>(total) / static_cast<double>(inner) + eps));
    __syncthreads();
    const ACC inv = s_inv;

    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        ACC v = static_cast<ACC>(xr[j]) * inv;
        if (w) v *= static_cast<ACC>(w[j]);
        orow[j] = static_cast<T>(v);
    }
}

}

Tensor rms_norm_cuda(const Tensor& input,
                     const std::vector<int64_t>& normalized_shape,
                     const std::optional<Tensor>& weight_opt,
                     std::optional<double> eps_opt) {
    const double eps = eps_opt.has_value() ? *eps_opt
        : (input.dtype() == DType::Float64 ? std::numeric_limits<double>::epsilon()
                                           : static_cast<double>(std::numeric_limits<float>::epsilon()));
    const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
    const int64_t input_ndim = input.dim();
    if (norm_ndim > input_ndim)
        TP_THROW(RuntimeError, "rms_norm: normalized_shape dim larger than input dim");
    int64_t N = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(input_ndim - norm_ndim + i) != normalized_shape[i])
            TP_THROW(RuntimeError, "rms_norm: Input shape mismatch with normalized_shape");
        N *= normalized_shape[i];
    }
    const int64_t M = input.numel() / (N == 0 ? 1 : N);
    const bool has_weight = weight_opt.has_value() && weight_opt->defined();

    Tensor in_contig = input.contiguous();
    Tensor weight = has_weight ? weight_opt->contiguous() : Tensor();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(in_contig.shape()),
                               in_contig.dtype(), in_contig.device());
    if (in_contig.numel() == 0 || M == 0 || N == 0) return out;

    const int threads = 256;
    const dim3 grid(static_cast<unsigned int>(M));
    const auto stream = getCurrentCUDAStream().stream();
    switch (in_contig.dtype()) {
#define RMS_FORWARD_CASE(ctype, name, acc_t)                                   \
        case DType::name:                                                      \
            rms_norm_row_kernel<ctype, acc_t><<<grid, threads, 0, stream>>>(    \
                in_contig.data_ptr<ctype>(),                                   \
                has_weight ? weight.data_ptr<ctype>() : nullptr,               \
                out.data_ptr<ctype>(), N, eps);                                \
            break;
        RMS_FORWARD_CASE(float, Float32, float)
        RMS_FORWARD_CASE(double, Float64, double)
        RMS_FORWARD_CASE(Half, Float16, float)
        RMS_FORWARD_CASE(BFloat16, BFloat16, float)
#undef RMS_FORWARD_CASE
        default:
            TP_THROW(NotImplementedError,
                     "rms_norm CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    check_normalization_cuda_launch("rms_norm_cuda");
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, RMSNormKernels) {
    m.impl("rms_norm", rms_norm_cuda);
}

}
}
