#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Apply pass of a scatter that moves a half payload over linear storage: one
// invocation owns one whole 32-bit word, which carries two lanes, and rewrites
// it in place.  The value a lane takes comes out of the half field of a source
// word, so the pair beside it stays untouched in both payloads, and it goes back
// through one rounding because it never leaves the width its storage is in.
layout(set = 0, binding = 0) buffer PRECISION restrict InOut {
  uint data[];
}
uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Marks {
  int data[];
}
uMarks;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Src {
  uint data[];
}
uSrc;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int count; // elements of the destination
  int fill;
  int fill2;
  int fill3;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uint(uBlock.count);
  if (word >= ((n + 1u) >> 1u)) {
    return;
  }

  uint packed = uSelf.data[word];
  bool touched = false;
  for (uint lane = 0u; lane < 2u; ++lane) {
    const uint element = (word << 1u) + lane;
    if (element >= n) {
      break;
    }
    const int entry = uMarks.data[element];
    if (entry > 0) {
      const int source = entry - 1;
      const float incoming =
          unpackHalf2x16(uSrc.data[source >> 1])[source & 1];
      const uint field = packHalf2x16(vec2(incoming, 0.0)) & 0xffffu;
      const uint shift = 16u * lane;
      packed = (packed & ~(0xffffu << shift)) | (field << shift);
      touched = true;
    }
  }
  if (touched) {
    uSelf.data[word] = packed;
  }
}
