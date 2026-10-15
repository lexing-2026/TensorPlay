// Eye and randint factory kernels.
// eye: one thread per element writes 1 on the diagonal and 0 elsewhere.
// randint: each element maps a hashed 64-bit draw into [low, low+range).

#include <metal_stdlib>
using namespace metal;

inline uint rng_hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

inline ulong rng_hash64(uint seed, uint idx) {
  const uint h0 = rng_hash(seed + 2 * idx * 0x9e3779b9U);
  const uint h1 = rng_hash(seed + (2 * idx + 1) * 0x9e3779b9U);
  return (static_cast<ulong>(h0) << 32) | static_cast<ulong>(h1);
}

template <typename T>
kernel void eye_fill(device T* out [[buffer(0)]],
                     constant uint& rows [[buffer(1)]],
                     constant uint& cols [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
  const uint i = tid / cols;
  const uint j = tid % cols;
  out[tid] = (i == j) ? T(1) : T(0);
}

template <typename T>
kernel void randint_fill(device T* out [[buffer(0)]],
                         constant long& low [[buffer(1)]],
                         constant ulong& range [[buffer(2)]],
                         constant uint& seed [[buffer(3)]],
                         constant uint& numel [[buffer(4)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid >= numel) {
    return;
  }
  const ulong draw = rng_hash64(seed, tid) % range;
  out[tid] = static_cast<T>(low + static_cast<long>(draw));
}

#define REGISTER_EYE(DTYPE)                                                \
  template [[host_name("eye_" #DTYPE)]] kernel void                        \
  eye_fill<DTYPE>(device DTYPE * out [[buffer(0)]],                        \
                  constant uint& rows [[buffer(1)]],                        \
                  constant uint& cols [[buffer(2)]],                       \
                  uint tid [[thread_position_in_grid]])

#define REGISTER_RANDINT(DTYPE)                                             \
  template [[host_name("randint_" #DTYPE)]] kernel void                     \
  randint_fill<DTYPE>(device DTYPE * out [[buffer(0)]],                     \
                      constant ulong& low [[buffer(1)]],                    \
                      constant ulong& range [[buffer(2)]],                 \
                      constant uint& seed [[buffer(3)]],                  \
                      constant uint& numel [[buffer(4)]],                \
                      uint tid [[thread_position_in_grid]])

REGISTER_EYE(float)
REGISTER_EYE(half)
REGISTER_EYE(bfloat)
REGISTER_EYE(long)
REGISTER_EYE(int)
REGISTER_EYE(short)
REGISTER_EYE(char)
REGISTER_EYE(uchar)
REGISTER_EYE(bool)
REGISTER_RANDINT(long)
REGISTER_RANDINT(int)
REGISTER_RANDINT(short)
REGISTER_RANDINT(char)
REGISTER_RANDINT(uchar)