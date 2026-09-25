#pragma once

#include "Tensor.h"
#include <cuda_runtime.h>

namespace tensorplay {
namespace cuda {
namespace {

template <typename T>
__device__ inline T warpReduceMax(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffffffffffffull, val, offset));
    return val;
}

template <typename T>
__device__ inline T warpReduceSum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffffffffffull, val, offset);
    return val;
}

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<tensorplay::Half>(tensorplay::Half v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<tensorplay::BFloat16>(tensorplay::BFloat16 v) {
    return static_cast<float>(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ tensorplay::Half from_float<tensorplay::Half>(float v) {
    return tensorplay::Half(v);
}

template <>
__device__ __forceinline__ tensorplay::BFloat16 from_float<tensorplay::BFloat16>(float v) {
    return tensorplay::BFloat16(v);
}

}
}
}
