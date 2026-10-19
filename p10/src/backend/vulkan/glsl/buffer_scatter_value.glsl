#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Apply pass of a scatter over linear storage: one invocation owns one whole
// word of the destination, four single-precision lanes, and rewrites it in
// place.  Owning the word is what makes the store safe without a
// read-modify-write of a narrower field; a word with nothing marked in it is
// left unstored, so the padding of a payload whose count is not a multiple of
// four keeps whatever the allocation put there.
layout(set = 0, binding = 0) buffer PRECISION restrict InOut {
  vec4 data[];
}
uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Marks {
  int data[];
}
uMarks;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int count;   // elements of the destination
  float value;  // the number written into every marked lane
  int fill;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uint(uBlock.count);
  if (word >= ((n + 3u) >> 2u)) {
    return;
  }

  vec4 value = uSelf.data[word];
  bool touched = false;
  for (uint lane = 0u; lane < 4u; ++lane) {
    const uint element = (word << 2u) + lane;
    if (element >= n) {
      break;
    }
    if (uMarks.data[element] > 0) {
      value[int(lane)] = uBlock.value;
      touched = true;
    }
  }
  if (touched) {
    uSelf.data[word] = value;
  }
}
