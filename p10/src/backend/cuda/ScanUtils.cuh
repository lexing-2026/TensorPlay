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

// Within-warp inclusive scan built on one ballot.  The lane id, the per-lane
// mask and the warp count all follow the device warp width: a hardware warp can
// be wider than 32 lanes, in which case two 32-thread groups share one ballot
// and a lane id truncated to 32 would have the upper group replay the lower
// group's prefix.
template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void topk_inclusive_binary_prefix_scan(
    T* smem, bool in, T* out, BinaryFunction binop) {
  const int warp_threads = warpSize;
  unsigned long long vote = __ballot_sync(~0ull, in);
  // Lanes at or below the caller, derived arithmetically so the scan needs
  // no platform-specific lane-mask intrinsic or inline assembly.
  const unsigned lane =
      static_cast<unsigned>(threadIdx.x) % static_cast<unsigned>(warp_threads);
  const unsigned long long lane_mask_le = ~0ull >> (64 - (lane + 1));
  T index = static_cast<T>(__popcll(lane_mask_le & vote));
  T carry = static_cast<T>(__popcll(vote));

  const int warp = threadIdx.x / warp_threads;
  if (lane == 0) {
    smem[warp] = carry;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    T current = 0;
    for (int i = 0; i < (blockDim.x + warp_threads - 1) / warp_threads; ++i) {
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
  *carry = smem[(blockDim.x + warpSize - 1) / warpSize - 1];
  if (KillWARDependency) {
    __syncthreads();
  }
}

}
}
}
