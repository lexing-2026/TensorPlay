#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Quantized clamp over linear storage: dequantize, clamp against the float
 * bounds, requantize with round-to-nearest-even into [-128, 127].
 *
 * A code payload keeps four lanes to a 32-bit word, and the fold is elementwise
 * on the payload's own order, so the four lanes of one word are the four
 * elements of one output word: a single fetch and a single store name the same
 * positions on both sides.  Lanes are extracted with shifts and masks because a
 * buffer view of 8-bit elements needs device features this baseline does not
 * assume, and merged back the same way.
 *
 * Buffer allocations round up to 16 bytes, so a payload whose element count does
 * not divide by four ends in a partly used word; the guard below keeps the
 * invocations whose first lane lies past the payload out of the run.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  float in_scale;
  int in_zero_point;
  float inv_out_scale;
  int out_zero_point;
  int has_min;
  int has_max;
  float min_value;
  float max_value;
  uint buf_length; // element count, shared by both payloads
  int fill0;
  int fill1;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Bit 7 of a lane is the sign of a two's-complement int8 code, so a byte of 128
// or more names a negative value and the remaining seven bits name its magnitude.
int code_lane(const uint byte_value) {
  return int(byte_value & 0x7Fu) - int(byte_value & 0x80u);
}

// A clamped code needs the eight bits its lane keeps, and the low eight bits of
// the two's-complement form are exactly those bits.
uint packed_lane(const int clamped_code) {
  return uint(clamped_code & 0xFF);
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint first = word << 2u;
  if (first >= uBlock.buf_length) {
    return;
  }

  const uint packed = uInput.data[word];
  vec4 y = vec4(
      float(code_lane(packed & 0xFFu)),
      float(code_lane((packed >> 8u) & 0xFFu)),
      float(code_lane((packed >> 16u) & 0xFFu)),
      float(code_lane((packed >> 24u) & 0xFFu)));
  y = (y - vec4(float(uBlock.in_zero_point))) * vec4(uBlock.in_scale);

  if (uBlock.has_min != 0) {
    y = max(y, vec4(uBlock.min_value));
  }
  if (uBlock.has_max != 0) {
    y = min(y, vec4(uBlock.max_value));
  }

  const ivec4 rounded =
      ivec4(roundEven(y * uBlock.inv_out_scale)) +
      ivec4(uBlock.out_zero_point);
  const ivec4 q = clamp(rounded, ivec4(-128), ivec4(127));

  uOutput.data[word] = packed_lane(q.x) | (packed_lane(q.y) << 8u) |
      (packed_lane(q.z) << 16u) | (packed_lane(q.w) << 24u);
}
