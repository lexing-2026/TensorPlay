#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Apply pass of a row fill over linear storage: one invocation owns one whole
// word of the destination, four single-precision lanes, and rewrites it in
// place.  Owning the word is what makes the store safe, since a row whose count
// is not a multiple of four ends in a word whose remaining lanes belong to the
// row below - and a row is named by a code on its own, so the two lanes answer
// to different marks.  Each lane therefore looks up its own row, which the
// destination's width and the count of its rows place without any list to walk.
layout(set = 0, binding = 0) buffer PRECISION restrict InOut {
  vec4 data[];
}
uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Marks {
  int data[];
}
uMarks;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int count;  // elements of the destination
  int width;  // elements of one row, and the stride between rows
  int rows;   // entries the destination offers along the axis
  float value; // the number written into a named row
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

  vec4 held = uSelf.data[word];
  bool touched = false;
  for (uint lane = 0u; lane < 4u; ++lane) {
    const uint element = (word << 2u) + lane;
    if (element >= n) {
      break;
    }
    if (uMarks.data[(element / uint(uBlock.width)) % uint(uBlock.rows)] != 0) {
      held[int(lane)] = uBlock.value;
      touched = true;
    }
  }
  if (touched) {
    uSelf.data[word] = held;
  }
}
