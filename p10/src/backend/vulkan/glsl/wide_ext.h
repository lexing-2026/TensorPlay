/*
 * A count wider than a word has no register to hold it and no texel format to
 * name, so every step below keeps the value as a pair of 32-bit words, low one
 * first, and hands the carry between them by hand.  Nothing here asks a shader
 * for 64-bit arithmetic, which a phone driver is not obliged to provide.
 *
 * The words are read as unsigned: two's complement addition, subtraction and
 * the low half of a product come out the same whichever sign the payload
 * carries, and the sign is only ever the caller's interpretation.
 */

// Sum of a pair.  A low word that wraps backwards has handed one to the high
// word; a total past every word lands where the counts it added would land.
uvec2 wide_add(const uvec2 x, const uvec2 y) {
  const uint low = x.x + y.x;
  const uint high = x.y + y.y + (low < x.x ? 1u : 0u);
  return uvec2(low, high);
}

// Difference of a pair, borrowing the other way: a low word that goes below
// nothing has taken one from the high word.
uvec2 wide_sub(const uvec2 x, const uvec2 y) {
  const uint low = x.x - y.x;
  const uint high = x.y - y.y - (x.x < y.x ? 1u : 0u);
  return uvec2(low, high);
}

// Product of two single-word factors as a pair.  Each of the four 16-bit
// partial products fits a word exactly, and they are combined 16 bits at a
// time, so no step of the walk ever needs more than a word.
uvec2 wide_mul_words(const uint x, const uint y) {
  const uint xl = x & 0xFFFFu;
  const uint xh = x >> 16u;
  const uint yl = y & 0xFFFFu;
  const uint yh = y >> 16u;
  const uint w0 = xl * yl;
  const uint w1 = xl * yh;
  const uint w2 = xh * yl;
  const uint w3 = xh * yh;
  // The two middle terms share the bits 16..31; carrying them by hand keeps
  // their sum inside a word, which the widest case proves it can be.
  const uint k1 = (w0 >> 16u) + (w2 & 0xFFFFu) + w1;
  const uint k2 = (k1 >> 16u) + (w2 >> 16u) + w3;
  return uvec2((w0 & 0xFFFFu) | (k1 << 16u), k2);
}

// Product of two pairs, keeping the low 64 bits.  The two cross terms are each
// weighted by 2^32, so only their low halves reach the answer, and anything
// weighted by 2^64 or beyond leaves it entirely.
uvec2 wide_mul(const uvec2 x, const uvec2 y) {
  const uvec2 base = wide_mul_words(x.x, y.x);
  const uint cross = x.x * y.y + x.y * y.x;
  return uvec2(base.x, base.y + cross);
}

// Two's complement negation: every bit turns, then one count comes back.
uvec2 wide_neg(const uvec2 x) {
  return wide_add(~x, uvec2(1u, 0u));
}

// Magnitude of a pair: a negative count answers with its negation, and the
// high word alone carries the sign.
uvec2 wide_abs(const uvec2 x) {
  return (x.y >> 31u) == 0u ? x : wide_neg(x);
}
