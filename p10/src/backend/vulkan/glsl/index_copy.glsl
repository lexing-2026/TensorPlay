#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// In-place scatter of whole rows along the sequence axis of a channel-packed
// [N, C, S, W] texture: one invocation copies one source row texel into its
// destination row, addressed through an Int32 index payload.  Rows are the
// texture's y extent, so the copy stays texel-aligned whatever the channel
// lanes hold.  Nothing is read from the destination, which keeps the write
// third the only memory access besides the source fetch.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uSelf;
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
  if (pos.x >= uBlock.extents.x || pos.y >= uBlock.n_idx ||
      pos.z >= uBlock.extents.z) {
    return;
  }

  const int dest_row = texelFetch(uIndices, ivec3(pos.y, 0, 0), 0).x;
  // A code naming no row of the destination writes nothing, the same way a
  // read-side lookup returns zeros for one.  Addressing an image outside its
  // extents is undefined, so the row it would have landed on is left alone.
  if (dest_row < 0 || dest_row >= uBlock.extents.y) {
    return;
  }
  imageStore(
      uSelf,
      ivec3(pos.x, dest_row, pos.z),
      texelFetch(uSource, ivec3(pos.x, pos.y, pos.z), 0));
}
