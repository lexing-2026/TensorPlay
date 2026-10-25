#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Per-tensor affine quantization over linear storage:
 *   code = clamp(roundEven(x * inv_scale) + zero_point, quant_min, quant_max)
 *
 * A single-precision payload fills one word with four lanes while an Int8 code
 * payload keeps four lanes to a word, so one fetch and one store name the same
 * four positions on both sides.  The codes are merged with masks and shifts
 * because a buffer view of 8-bit elements needs device features this baseline
 * does not assume.
 *
 * Buffer allocations round up to 16 bytes, so a payload whose element count
 * does not divide by four ends in a partly used word; the guard below keeps the
 * invocations whose first lane lies past the payload out of the run, while the
 * spare lanes of the last kept word take whatever the padded bytes say.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  vec4 data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  float inv_scale; // 1 / scale
  int zero_point;
  int quant_min;
  int quant_max;
  uint buf_length; // element count, shared by both payloads
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// A clamped code needs the eight bits its lane keeps, and the low eight bits of
// the two's-complement form are exactly those bits.
uint code_lane(const int clamped_code) {
  return uint(clamped_code & 0xFF);
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint first = word << 2u;
  if (first >= uBlock.buf_length) {
    return;
  }

  const ivec4 rounded = ivec4(roundEven(uInput.data[word] * uBlock.inv_scale)) +
      ivec4(uBlock.zero_point);
  const ivec4 q =
      clamp(rounded, ivec4(uBlock.quant_min), ivec4(uBlock.quant_max));

  uOutput.data[word] = code_lane(q.x) | (code_lane(q.y) << 8u) |
      (code_lane(q.z) << 16u) | (code_lane(q.w) << 24u);
}
