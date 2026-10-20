#version 450 core

layout(std430) buffer;

/*
 * Single precision to 64-bit codes over linear storage.  A wide element owns a
 * pair of words, so one invocation reads one 16-byte float word and writes the
 * four elements it names: 32 bytes.  Each element is guarded on its own because
 * an allocation only rounds up to a multiple of 16 bytes and a wide element at
 * the end of a payload needs all eight of its bytes.
 *
 * The fraction is taken away first, so a value lands on the whole number nearest
 * its magnitude on the side of nothing, which is the rule a host cast follows.
 * The magnitude is then split at two to the thirty-second into the two words of
 * the code, and a negative one is complemented and advanced by one: the pair is
 * the number itself, not a sign kept beside a magnitude.
 *
 * Both splits are exact.  A float carries at most twenty-four significant bits,
 * so the high part is a shift and the leftover sits inside the reach of a word
 * with no bits of its own to lose.
 *
 * A magnitude the pair cannot name keeps the widest code on its side, and a
 * number that is not a number keeps nothing: the conversion of either into a
 * word is not defined by the shader language, so both are answered here rather
 * than left to it.
 */
layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  uvec2 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer highp restrict readonly InBuffer {
  vec4 data[];
}
uInput;
layout(set = 0, binding = 2) uniform highp restrict Block {
  uint buf_length;  // element count, shared by both payloads
  uint fill0;
  uint fill1;
  uint fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

uvec2 wide_of(const float value) {
  if (isnan(value)) {
    return uvec2(0u);
  }
  const float whole = trunc(value);
  if (whole >= 9223372036854775808.0) {
    // The most positive code a pair of words names: two to the sixty-third,
    // less one.
    return uvec2(0xFFFFFFFFu, 0x7FFFFFFFu);
  }
  if (whole < -9223372036854775808.0) {
    // Two words all set above the sign, which is the most negative code.
    return uvec2(0u, 0x80000000u);
  }
  const bool negative = whole < 0.0;
  const float magnitude = negative ? -whole : whole;
  const float high = floor(magnitude * (1.0 / 4294967296.0));
  const float low = magnitude - high * 4294967296.0;
  uint low_word = uint(low);
  uint high_word = uint(high);
  if (negative) {
    // The carry out of the low word only happens when the complemented low word
    // was all ones, that is when the magnitude's own low word was nothing.
    const bool carries = (low_word == 0u);
    low_word = ~low_word + 1u;
    high_word = ~high_word + (carries ? 1u : 0u);
  }
  return uvec2(low_word, high_word);
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint base = word << 2u;
  if (base >= uBlock.buf_length) {
    return;
  }
  const vec4 values = uInput.data[word];
  for (uint i = 0u; i < 4u; ++i) {
    const uint e = base + i;
    if (e >= uBlock.buf_length) {
      return;
    }
    uOutput.data[e] = wide_of(values[i]);
  }
}
