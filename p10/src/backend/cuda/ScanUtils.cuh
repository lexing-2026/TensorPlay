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
  unsigned vote = __ballot_sync(0xffffffffu, in);
  unsigned lane_mask_le;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(lane_mask_le));
  T index = static_cast<T>(__popc(lane_mask_le & vote));
  T carry = static_cast<T>(__popc(vote));

  const int warp = threadIdx.x / 32;
  int lane_id;
  asm("mov.s32 %0, %%laneid;" : "=r"(lane_id));
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
