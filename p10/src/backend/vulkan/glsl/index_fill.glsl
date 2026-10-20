#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Apply pass of a row fill over image storage, along the axis the texture's y
// extent carries: one invocation owns one destination texel and rewrites it in
// whole when the mark of its row names it.  A texel's four lanes are four
// elements of the same row - the row axis is a line of whole texels - so the
// one store covers them all, and an unnamed row's texel is left unstored rather
// than handed back as what it already holds.  No two invocations touch one
// texel, which is what lets the pass read and write the same image.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Marks {
  int data[];
}
uMarks;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents; // (texel width, destination rows, texel planes, unused)
  int rows;      // entries the destination offers along the axis
  float value;   // the number written into a named row
  int fill;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.extents.x || pos.y >= uBlock.rows ||
      pos.z >= uBlock.extents.z) {
    return;
  }
  if (uMarks.data[pos.y] == 0) {
    return;
  }
  imageStore(uSelf, pos, vec4(uBlock.value));
}
