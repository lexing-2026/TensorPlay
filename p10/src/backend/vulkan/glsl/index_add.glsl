#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// In-place accumulation of whole rows along the axis just above the last one of
// a texture payload: one invocation owns one destination texel, walks the Int32
// index payload and adds every source row the list names onto it.  That axis is
// the texture's y extent, so a row is a line of whole texels and both the fetch
// and the store stay texel-aligned whatever the lanes and the depth planes hold.
//
// The walk runs from the destination rather than from the source, so a code the
// list repeats is added by the single invocation that owns the row it names, in
// the order the list states.  No two invocations touch one texel, which is what
// lets the accumulation read and write the same image without a lock.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uSelf;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uSource;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 extents; // (texel width, destination rows, texel planes, unused)
  int n_idx;     // rows of the source and entries of the payload
  int fill;
  int fill2;
  int fill3;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.extents.x || pos.y >= uBlock.extents.y ||
      pos.z >= uBlock.extents.z) {
    return;
  }

  vec4 total = imageLoad(uSelf, pos);
  for (int k = 0; k < uBlock.n_idx; ++k) {
    // A code outside the destination adds nothing; addressing an image beyond
    // its extents is undefined, so the row it would have landed on keeps its
    // own value untouched.
    if (texelFetch(uIndices, ivec3(k, 0, 0), 0).x != pos.y) {
      continue;
    }
    total += texelFetch(uSource, ivec3(pos.x, k, pos.z), 0);
  }
  imageStore(uSelf, pos, total);
}
