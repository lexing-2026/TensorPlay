#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define REPLICATE ${REPLICATE}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  // x: left pad, y: top pad
  ivec2 padding;
  int c_depth; // ceil(C / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * 2D spatial padding.  REPLICATE selects replication (edge clamp);
 * otherwise reflection pads without repeating the edge element.  Channels
 * and batches pass through untouched, so the texel z coordinate carries
 * over directly.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  int iw = pos.x - uBlock.padding.x;
  int ih = pos.y - uBlock.padding.y;

  // clang-format off
  $if REPLICATE:
    iw = clamp(iw, 0, uBlock.in_sizes.x - 1);
    ih = clamp(ih, 0, uBlock.in_sizes.y - 1);
  $else:
    if (iw < 0) {
      iw = -iw;
    }
    if (iw >= uBlock.in_sizes.x) {
      iw = 2 * (uBlock.in_sizes.x - 1) - iw;
    }
    if (ih < 0) {
      ih = -ih;
    }
    if (ih >= uBlock.in_sizes.y) {
      ih = 2 * (uBlock.in_sizes.y - 1) - ih;
    }
  // clang-format on

  const vec4 v = texelFetch(uInput, ivec3(iw, ih, pos.z), 0);
  imageStore(uOutput, pos, v);
}
