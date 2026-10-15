// Constant fill kernel for the MPS factory ops (full/zeros/ones/fill_).
// One thread per element writes the same value.

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void fill_constant(device T* out [[buffer(0)]],
                       constant T& value [[buffer(1)]],
                       constant uint& numel [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
  if (tid < numel) {
    out[tid] = value;
  }
}

#define REGISTER_FILL(DTYPE)                                              \
  template [[host_name("fill_" #DTYPE)]] kernel void                     \
  fill_constant<DTYPE>(device DTYPE * out [[buffer(0)]],                 \
                       constant DTYPE& value [[buffer(1)]],                \
                       constant uint& numel [[buffer(2)]],               \
                       uint tid [[thread_position_in_grid]]);

REGISTER_FILL(float)
REGISTER_FILL(half)
REGISTER_FILL(bfloat)
REGISTER_FILL(long)
REGISTER_FILL(int)
REGISTER_FILL(short)
REGISTER_FILL(char)
REGISTER_FILL(uchar)
REGISTER_FILL(bool)