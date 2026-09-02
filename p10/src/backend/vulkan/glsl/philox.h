/*
 * Philox4x32-10 counter-based random number generator for compute shaders.
 * Counter-style generators suit GPU evaluation: every invocation derives its
 * random words from (seed, offset, invocation id) without shared state, so
 * results are reproducible for a given seed and independent of scheduling.
 *
 * The generator follows the reference 10-round specification with the
 * standard multipliers and Weyl constants.  Vulkan 1.0 compute has no 64-bit
 * integer arithmetic, so the 32x32->64 multiplication is built from 16-bit
 * halves.
 */

// 32x32 -> 64 multiply expressed through 16-bit halves; returns (hi, lo).
uvec2 mulhilo(const uint a, const uint b) {
  const uint a_lo = a & 0xffffu;
  const uint a_hi = a >> 16u;
  const uint b_lo = b & 0xffffu;
  const uint b_hi = b >> 16u;

  const uint ll = a_lo * b_lo;
  const uint lh = a_lo * b_hi;
  const uint hl = a_hi * b_lo;
  const uint hh = a_hi * b_hi;

  const uint mid = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
  return uvec2(
      hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u),
      (mid << 16u) | (ll & 0xffffu));
}

// One Philox round: two 32x32 multiplies feeding a high/low shuffle.
uvec4 philox_round(uvec4 ctr, const uvec2 key) {
  const uvec2 prod0 = mulhilo(0xD2511F53u, ctr.x);
  const uvec2 prod1 = mulhilo(0xCD9E8D57u, ctr.z);
  return uvec4(
      prod1.y ^ ctr.y ^ key.x,
      prod1.x,
      prod0.y ^ ctr.w ^ key.y,
      prod0.x);
}

// Full 10-round Philox4x32 with the reference Weyl key sequence.
uvec4 philox4x32_10(const uvec4 ctr, const uvec2 key) {
  uvec2 k = key;
  uvec4 c = ctr;
  for (int i = 0; i < 9; ++i) {
    c = philox_round(c, k);
    k += uvec2(0x9E3779B9u, 0xBB67AE85u);
  }
  return philox_round(c, k);
}

// Maps a 32-bit word onto [0, 1) with a 24-bit mantissa.
float word_to_uniform(const uint w) {
  return float(w >> 8u) * (1.0f / 16777216.0f);
}

// Two standard normal draws from two uniforms via the Box-Muller transform.
// The log argument is mapped into (0, 1] so the transform stays finite.
vec2 box_muller(const float u1_raw, const float u2) {
  const float u1 = 1.0f - u1_raw; // (0, 1]
  const float r = sqrt(-2.0f * log(u1));
  const float theta = 6.283185307179586f * u2;
  return vec2(r * cos(theta), r * sin(theta));
}
