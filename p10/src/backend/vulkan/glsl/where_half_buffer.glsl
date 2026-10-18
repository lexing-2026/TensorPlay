#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer where over half-precision payloads.  A half payload packs
// two lanes per 32-bit word and the Bool condition one byte per element, four
// lanes per word, so one condition word feeds two result words and which pair
// of bytes a result word draws on follows the half of the word it sits in.
// Selection moves the 16-bit fields themselves: nothing is widened and
// re-narrowed, so a rejected lane's infinity or not-a-number cannot reach the
// result.
//
// One invocation owns one whole result word, which is what keeps two
// invocations from writing different lanes of the same word.  An odd length
// leaves the second lane of the final word past the logical end; the read
// stays inside the allocation, because buffer sizes round up to 16 bytes, and
// no read-back reaches that lane.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Condition {
  uint data[];
}
uCond;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly Other {
  uint data[];
}
uOther;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  uint buf_length; // element count
  uint fill0;      // std140 padding
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint word = gl_GlobalInvocationID.x;
  if (word >= ((uBlock.buf_length + 1u) >> 1u)) {
    return;
  }

  const uint flags = uCond.data[word >> 1u];
  // A condition byte occupies eight bits of its word, so a result word's two
  // bytes start at bit 0 or bit 16, whichever half of the word they sit in.
  const uint base = (word & 1u) * 16u;
  const bool first = ((flags >> base) & 0xffu) != 0u;
  const bool second = ((flags >> (base + 8u)) & 0xffu) != 0u;

  const uint a = uInput.data[word];
  const uint b = uOther.data[word];
  const uint low = first ? (a & 0xffffu) : (b & 0xffffu);
  const uint high = second ? (a >> 16u) : (b >> 16u);
  uOutput.data[word] = low | (high << 16u);
}
