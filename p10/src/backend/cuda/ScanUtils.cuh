#pragma once

#include <cuda_runtime.h>

namespace tensorplay {
namespace cuda {
namespace topk_detail {

template <typename T>
struct TopKAddOp {
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return lhs + rhs;
  }
};

template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void topk_inclusive_binary_prefix_scan(
    T* smem, bool in, T* out, BinaryFunction binop) {
  unsigned long long vote = __ballot_sync(0xffffffffffffffffull, in);
  // Lanes at or below the caller, derived arithmetically so the scan needs
  // no platform-specific lane-mask intrinsic or inline assembly.  Kernels
  // address warps with the 32-lane model on both GPU toolchains.
  const unsigned lane = threadIdx.x & 31u;
  const unsigned long long lane_mask_le =
      (lane == 31u) ? 0xffffffffull : ((1ull << (lane + 1)) - 1ull);
  T index = static_cast<T>(__popcll(lane_mask_le & vote));
  T carry = static_cast<T>(__popcll(vote));

  const int warp = threadIdx.x / 32;
  const int lane_id = static_cast<int>(threadIdx.x & 31u);
  if (lane_id == 0) {
    smem[warp] = carry;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    T current = 0;
    for (int i = 0; i < (blockDim.x + 31) / 32; ++i) {
      T value = smem[i];
      smem[i] = binop(value, current);
      current = binop(current, value);
    }
  }

  __syncthreads();

  if (warp >= 1) {
    index = binop(index, smem[warp - 1]);
  }

  *out = index;

  if (KillWARDependency) {
    __syncthreads();
  }
}

template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void topk_exclusive_binary_prefix_scan(
    T* smem, bool in, T* out, T* carry, BinaryFunction binop) {
  topk_inclusive_binary_prefix_scan<T, false, BinaryFunction>(
      smem, in, out, binop);
  *out -= static_cast<T>(in);
  *carry = smem[(blockDim.x + 31) / 32 - 1];
  if (KillWARDependency) {
    __syncthreads();
  }
}

}
}
}
