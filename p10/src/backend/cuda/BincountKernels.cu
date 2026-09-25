#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "GPUPrimitives.cuh"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
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

__device__ __forceinline__ void atomic_add_rel(int64_t* addr, int64_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(float* addr, float v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(double* addr, double v) {
    gpuAtomicAdd(addr, v);
}

template <typename T>
__global__ void minmax_reduce_kernel(int64_t n, const T* x, T* min_out,
                                     T* max_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    T local_min = std::numeric_limits<T>::max();
    T local_max = std::numeric_limits<T>::lowest();
    for (; i < n; i += stride) {
        local_min = ::min(local_min, x[i]);
        local_max = ::max(local_max, x[i]);
    }
    __shared__ T min_shm[kThreads];
    __shared__ T max_shm[kThreads];
    int tid = threadIdx.x;
    min_shm[tid] = local_min;
    max_shm[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            min_shm[tid] = ::min(min_shm[tid], min_shm[tid + s]);
            max_shm[tid] = ::max(max_shm[tid], max_shm[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        min_out[blockIdx.x] = min_shm[0];
        max_out[blockIdx.x] = max_shm[0];
    }
}

template <typename T>
__global__ void bincount_count_kernel(int64_t n, const T* x, int64_t* bins) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) atomic_add_rel(&bins[x[i]], static_cast<int64_t>(1));
}

template <typename W>
__global__ void bincount_weighted_kernel(int64_t n, const int64_t* x, const W* wp, W* bins) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) atomic_add_rel(&bins[x[i]], wp[i]);
}

}

Tensor bincount_cuda(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
    Tensor weights = weights_opt.value_or(Tensor());
    globalContext().alertNotDeterministic("bincount_cuda");
    if (minlength < 0) TP_THROW(RuntimeError, "minlength should be >= 0");
    if (isFloatingType(self.dtype())) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    Tensor inp = self.to(DType::Int64).contiguous();
    int64_t n = inp.numel();
    if (self.dim() == 1 && n == 0) {
        return Tensor::zeros({minlength}, DType::Int64, self.device());
    }
    if (self.dim() != 1) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    bool has_weights = weights.defined() && weights.numel() > 0;
    if (has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))) {
        TP_THROW(RuntimeError, "weights should be 1-d and have the same length as input");
    }
    Tensor bounds_d = Tensor::empty({2}, DType::Int64, self.device());
    auto stream = getCurrentCUDAStream().stream();
    minmax_reduce_kernel<int64_t><<<1, kThreads, 0, stream>>>(
        n, inp.data_ptr<int64_t>(), bounds_d.data_ptr<int64_t>(),
        bounds_d.data_ptr<int64_t>() + 1);
    CUDA_CHECK(cudaGetLastError());
    int64_t bounds[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(bounds, bounds_d.data_ptr<int64_t>(),
                          sizeof(bounds), cudaMemcpyDeviceToHost));
    const int64_t min_v = bounds[0];
    const int64_t max_v = bounds[1];
    if (min_v < 0) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    if (max_v >= std::numeric_limits<int64_t>::max()) {
        TP_THROW(RuntimeError, "maximum value of input overflowed");
    }
    int64_t nbins = std::max(max_v + 1, minlength);
    if (has_weights) {
        if (weights.dtype() == DType::Float32) {
            Tensor rf = Tensor::zeros({nbins}, DType::Float32, self.device());
            Tensor w = weights.contiguous();
            bincount_weighted_kernel<float><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, inp.data_ptr<int64_t>(), w.data_ptr<float>(), rf.data_ptr<float>());
            CUDA_CHECK(cudaGetLastError());
            return rf;
        }
        Tensor rf = Tensor::zeros({nbins}, DType::Float64, self.device());
        Tensor w = weights.to(DType::Float64).contiguous();
        bincount_weighted_kernel<double><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, inp.data_ptr<int64_t>(), w.data_ptr<double>(), rf.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
        return rf;
    }
    Tensor result = Tensor::zeros({nbins}, DType::Int64, self.device());
    bincount_count_kernel<int64_t><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        n, inp.data_ptr<int64_t>(), result.data_ptr<int64_t>());
    CUDA_CHECK(cudaGetLastError());
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, BincountKernels) {
    m.impl("bincount", bincount_cuda);
}

}
}
