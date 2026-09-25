#pragma once

#include <cuda_runtime.h>

namespace tensorplay {
namespace cuda {
namespace layer_norm {

constexpr int kLNThreads = 256;

__device__ inline float ln_rsqrt(float v) { return rsqrtf(v); }
__device__ inline double ln_rsqrt(double v) { return 1.0 / ::sqrt(v); }

template <typename ACC>
__device__ inline void ln_block_reduce2(ACC& v0, ACC& v1, ACC* smem0, ACC* smem1) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wid = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffffffffffffull, v0, offset);
        v1 += __shfl_down_sync(0xffffffffffffffffull, v1, offset);
    }
    if (lane == 0) { smem0[wid] = v0; smem1[wid] = v1; }
    __syncthreads();
    const int nw = static_cast<int>(blockDim.x >> 5);
    v0 = (lane < nw) ? smem0[lane] : ACC(0);
    v1 = (lane < nw) ? smem1[lane] : ACC(0);
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffffffffffffull, v0, offset);
        v1 += __shfl_down_sync(0xffffffffffffffffull, v1, offset);
    }
    v0 = __shfl_sync(0xffffffffffffffffull, v0, 0);
    v1 = __shfl_sync(0xffffffffffffffffull, v1, 0);
}

}
}
}
