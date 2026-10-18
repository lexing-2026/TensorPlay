#version 450 core

layout(std430) buffer;

/*
 * Byte-packed linear payload to single precision.  An Int8, UInt8 or Bool
 * payload keeps one byte per element, four lanes to a 32-bit word, so one
 * invocation owns one source word and writes the four destination floats it
 * covers.  Byte lanes are extracted with shifts and masks because a buffer
 * view of 8-bit elements needs device features this baseline does not assume.
 *
 * Buffer allocations round up to 16 bytes, so a tail word past the logical
 * length is still inside the allocation; writing its lanes costs nothing and
 * nothing reads them back.
 */
layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer highp restrict readonly InBuffer {
  uint data[];
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

$if SIGNED:
  float byte_to_float(const uint byte_value) {
    // Sign-extend: bit 7 of the byte is the sign of a two's-complement int8.
    return float(int(byte_value & 0xFFu) - int((byte_value & 0x80u) << 1u));
  }
$else:
  float byte_to_float(const uint byte_value) {
    return float(byte_value & 0xFFu);
  }
// End of the byte-lane rules.

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint first = word << 2u;
  if (first >= uBlock.buf_length) {
    return;
  }

  const uint packed = uInput.data[word];
  uOutput.data[word] = vec4(
      byte_to_float(packed & 0xFFu),
      byte_to_float((packed >> 8u) & 0xFFu),
      byte_to_float((packed >> 16u) & 0xFFu),
      byte_to_float((packed >> 24u) & 0xFFu));
}
