// Arithmetic sequence fill for arange().  Each thread computes
// `start + tid * step` in double precision, then casts to the output type.

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void arange_fill(device T* out [[buffer(0)]],
                        constant double& start [[buffer(1)]],
                        constant double& step [[buffer(2)]],
                        constant uint& numel [[buffer(3)]],
                        uint tid [[thread_position_in_grid]]) {
  if (tid < numel) {
    out[tid] = static_cast<T>(start + static_cast<double>(tid) * step);
  }
}

#define REGISTER_ARANGE(DTYPE)                                            \
  template [[host_name("arange_" #DTYPE)]] kernel void                   \
  arange_fill<DTYPE>(device DTYPE * out [[buffer(0)]],                   \
                     constant double& start [[buffer(1)]],                \
                     constant double& step [[buffer(2)]],                \
                     constant uint& numel [[buffer(3)]],                \
                     uint tid [[thread_position_in_grid]])

REGISTER_ARANGE(float)
REGISTER_ARANGE(half)
REGISTER_ARANGE(bfloat)
REGISTER_ARANGE(long)
REGISTER_ARANGE(int)
REGISTER_ARANGE(short)
REGISTER_ARANGE(char)
REGISTER_ARANGE(uchar)