#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Accumulation pass of a scatter that moves a half payload, over linear storage:
// one invocation owns one whole 32-bit word, which carries two lanes, and grows
// each lane by the source value of every entry of the list that names it.  The
// pair sharing a word is why the ownership is the word and not the lane: an odd
// row length puts the last lane of one element beside the first of the next, so
// a lane-only store would hand a stale half to whoever owns the other.
//
// The running total is kept at full width and rounded on the way out, which is
// how the row accumulation already treats a half payload.
layout(set = 0, binding = 0) buffer PRECISION restrict InOut {
  uint data[];
}
uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Names {
  int data[];
}
uNames;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Src {
  uint data[];
}
uSrc;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int count; // elements of the destination
  int n;     // entries of the index list
  int fill;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// The 16-bit field holding one element of a dense half payload.
float half_field(uint word, uint lane) {
  return unpackHalf2x16(word)[lane];
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uint(uBlock.count);
  if (word >= ((n + 1u) >> 1u)) {
    return;
  }

  vec2 held = unpackHalf2x16(uSelf.data[word]);
  bool touched = false;
  for (int k = 0; k < uBlock.n; ++k) {
    const int named = uNames.data[k];
    if (named < 0 || (uint(named) >> 1u) != word) {
      continue;
    }
    held[named & 1] += half_field(uSrc.data[k >> 1], k & 1);
    touched = true;
  }
  if (touched) {
    uSelf.data[word] = packHalf2x16(held);
  }
}
