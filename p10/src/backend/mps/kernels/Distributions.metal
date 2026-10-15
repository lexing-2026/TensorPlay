// GPU random number generation for the MPS factory kernels.  A 32-bit seed
// (drawn once on the host) is mixed per element with a hash so adjacent
// threads produce uncorrelated values.  `rand` maps the hash to [0,1);
// `randn` converts two uniforms with Box-Muller.

#include <metal_stdlib>
using namespace metal;

// 32-bit integer hash (a single-round xorshift-multiply construction).
inline uint rng_hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

inline float rng_uniform(uint seed, uint idx) {
  return static_cast<float>(rng_hash(seed + idx * 0x9e3779b9U)) *
      (1.0f / 4294967296.0f);
}

template <typename T>
kernel void rand_uniform(device T* out [[buffer(0)]],
                         constant uint& seed [[buffer(1)]],
                         constant uint& numel [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid >= numel) {
    return;
  }
  out[tid] = static_cast<T>(rng_uniform(seed, tid));
}

template <typename T>
kernel void rand_normal(device T* out [[buffer(0)]],
                        constant uint& seed [[buffer(1)]],
                        constant uint& numel [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
  if (tid >= numel) {
    return;
  }
  const float u1 = max(rng_uniform(seed, 2 * tid), 1e-9f);
  const float u2 = rng_uniform(seed, 2 * tid + 1);
  const float r = sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265358979f * u2);
  out[tid] = static_cast<T>(r);
}

#define REGISTER_RAND(DTYPE)                                              \
  template [[host_name("rand_uniform_" #DTYPE)]] kernel void             \
  rand_uniform<DTYPE>(device DTYPE * out [[buffer(0)]],                  \
                      constant uint& seed [[buffer(1)]],                  \
                      constant uint& numel [[buffer(2)]],               \
                      uint tid [[thread_position_in_grid]]);              \
  template [[host_name("rand_normal_" #DTYPE)]] kernel void              \
  rand_normal<DTYPE>(device DTYPE * out [[buffer(0)]],                    \
                     constant uint& seed [[buffer(1)]],                \
                     constant uint& numel [[buffer(2)]],                 \
                     uint tid [[thread_position_in_grid]])

REGISTER_RAND(float)
REGISTER_RAND(half)
REGISTER_RAND(bfloat)