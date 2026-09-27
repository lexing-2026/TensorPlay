#pragma once

// Stateless Philox-4x32 PRNG.
//
// Unlike the stateful generator the rest of these kernels use, this is a pure
// function: given (seed, offset) it returns four pseudo-random 32-bit values
// with nothing carried between calls.  That is what makes it usable from a
// graph: a value read at a position does not depend on what was read before
// it, so the same graph produces the same values, and two graphs that read the
// same positions produce the same values as each other.
//
// The cipher works on a 128-bit counter whose full form is
// (offset_lo, offset_hi, subsequence_lo, subsequence_hi).  The subsequence
// halves are held at zero so that the whole counter is addressed by the offset
// alone.  Subsequence numbers taken from a thread index or a device's thread
// count would spread reads differently across devices, so the values would
// differ between two devices for the same seed and offset; holding them at
// zero is what makes the answer a function of the offset and nothing else.

#include <cstdint>

namespace tensorplay {
namespace cuda {

__device__ __forceinline__ uint2 mulhilo32(uint32_t a, uint32_t b) {
  return {a * b, __umulhi(a, b)};
}

__device__ __forceinline__ uint4 philox_round(uint4 ctr, uint2 key) {
  constexpr uint32_t kPhiloxSA = 0xD2511F53;
  constexpr uint32_t kPhiloxSB = 0xCD9E8D57;
  uint2 r0 = mulhilo32(kPhiloxSA, ctr.x);
  uint2 r1 = mulhilo32(kPhiloxSB, ctr.z);
  return {r1.y ^ ctr.y ^ key.x, r1.x, r0.y ^ ctr.w ^ key.y, r0.x};
}

// Four pseudo-random 32-bit values (128 bits) determined entirely by
// (seed, offset).  Every distinct offset gives a distinct 128-bit output.
template <int N_ROUNDS = 10>
__device__ __forceinline__ uint4 philox_4x32(uint64_t seed, uint64_t offset) {
  uint2 key = {
      static_cast<uint32_t>(seed),
      static_cast<uint32_t>(seed >> 32)};
  uint4 ctr = {
      static_cast<uint32_t>(offset),
      static_cast<uint32_t>(offset >> 32),
      // The two subsequence halves stay at zero, so that the offset alone
      // addresses the whole counter.
      0,
      0};

  constexpr uint32_t kPhilox10A = 0x9E3779B9;
  constexpr uint32_t kPhilox10B = 0xBB67AE85;

  #pragma unroll
  for (int i = 0; i < N_ROUNDS - 1; i++) {
    ctr = philox_round(ctr, key);
    key.x += kPhilox10A;
    key.y += kPhilox10B;
  }
  return philox_round(ctr, key);
}

// A new (seed, offset) from four random 32-bit values: two for the seed and
// two for the offset, low half first.
__device__ __forceinline__ void philox_derive_key(
    uint4 r,
    uint64_t* out_seed,
    uint64_t* out_offset) {
  *out_seed = static_cast<uint64_t>(r.x) | (static_cast<uint64_t>(r.y) << 32);
  *out_offset = static_cast<uint64_t>(r.z) | (static_cast<uint64_t>(r.w) << 32);
}

} // namespace cuda
} // namespace tensorplay
