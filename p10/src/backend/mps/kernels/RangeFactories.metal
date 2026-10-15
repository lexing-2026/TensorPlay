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
                     uint tid [[thread_position_in_grid]]);

REGISTER_ARANGE(float)
REGISTER_ARANGE(half)
REGISTER_ARANGE(bfloat)
REGISTER_ARANGE(long)
REGISTER_ARANGE(int)
REGISTER_ARANGE(short)
REGISTER_ARANGE(char)
REGISTER_ARANGE(uchar)

// linspace: start + i * step (step = (end-start)/(steps-1), precomputed).
template <typename T>
kernel void linspace_fill(device T* out [[buffer(0)]],
                          constant double& start [[buffer(1)]],
                          constant double& step [[buffer(2)]],
                          constant uint& numel [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
  if (tid < numel) {
    out[tid] = static_cast<T>(start + static_cast<double>(tid) * step);
  }
}

// logspace: base^(start + i * step).
template <typename T>
kernel void logspace_fill(device T* out [[buffer(0)]],
                          constant double& base [[buffer(1)]],
                          constant double& start [[buffer(2)]],
                          constant double& step [[buffer(3)]],
                          constant uint& numel [[buffer(4)]],
                          uint tid [[thread_position_in_grid]]) {
  if (tid < numel) {
    out[tid] = static_cast<T>(pow(base, start + static_cast<double>(tid) * step));
  }
}

#define REGISTER_LINSPACE(DTYPE)                                            \
  template [[host_name("linspace_" #DTYPE)]] kernel void                    \
  linspace_fill<DTYPE>(device DTYPE * out [[buffer(0)]],                    \
                       constant double& start [[buffer(1)]],               \
                       constant double& step [[buffer(2)]],                \
                       constant uint& numel [[buffer(3)]],                  \
                       uint tid [[thread_position_in_grid]]);

#define REGISTER_LOGSPACE(DTYPE)                                            \
  template [[host_name("logspace_" #DTYPE)]] kernel void                   \
  logspace_fill<DTYPE>(device DTYPE * out [[buffer(0)]],                   \
                       constant double& base [[buffer(1)]],                  \
                       constant double& start [[buffer(2)]],                \
                       constant double& step [[buffer(3)]],                \
                       constant uint& numel [[buffer(4)]],                \
                       uint tid [[thread_position_in_grid]]);

REGISTER_LINSPACE(float)
REGISTER_LINSPACE(half)
REGISTER_LINSPACE(bfloat)
REGISTER_LINSPACE(long)
REGISTER_LINSPACE(int)
REGISTER_LOGSPACE(float)
REGISTER_LOGSPACE(half)
REGISTER_LOGSPACE(bfloat)