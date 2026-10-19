#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Undoing a per-tensor affine quantization over linear storage:
 *   out = scale * (code - zero_point)
 *
 * An Int8 code payload keeps four lanes to a 32-bit word while a single-precision
 * payload fills one word with four, so one fetch and one store name the same
 * four positions on both sides.  Byte lanes are extracted with shifts and masks
 * because a buffer view of 8-bit elements needs device features this baseline
 * does not assume.
 *
 * Buffer allocations round up to 16 bytes, so a payload whose element count
 * does not divide by four ends in a partly used word that the guard below keeps
 * out of the run.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  float scale;
  int zero_point;
  uint buf_length; // element count, shared by both payloads
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Bit 7 of a lane is the sign of a two's-complement int8 code, so a byte of 128
// or more names a negative value and the remaining seven bits name its magnitude.
float code_to_float(const uint byte_value) {
  return float(int(byte_value & 0x7Fu) - int(byte_value & 0x80u));
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint first = word << 2u;
  if (first >= uBlock.buf_length) {
    return;
  }

  const uint packed = uInput.data[word];
  const float scale = uBlock.scale;
  const float zero_point = float(uBlock.zero_point);
  uOutput.data[word] = vec4(
      scale * (code_to_float(packed & 0xFFu) - zero_point),
      scale * (code_to_float((packed >> 8u) & 0xFFu) - zero_point),
      scale * (code_to_float((packed >> 16u) & 0xFFu) - zero_point),
      scale * (code_to_float((packed >> 24u) & 0xFFu) - zero_point));
}
