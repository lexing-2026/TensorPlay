#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 sizes; // {N, C, H, W} logical sizes of the source operand
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Packs the right operand of a batched product into per-batch height-packed
 * planes: output texel (n, j, b) carries M[b][4j .. 4j + 3][n] in its lanes,
 * aligning the reduction axis with the vec4 lanes.
 *
 * The source is the channel-packed operand texture, where element (b, k, n)
 * lives at texel (n, k, b / 4) in lane b % 4.  A single-batch source
 * broadcasts across every output plane.  K elements past the edge are stored
 * as zeros so the packed tail never carries garbage into downstream
 * reductions.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int k_texels = (uBlock.sizes.z + 3) / 4;

  if (pos.x >= uBlock.sizes.w || pos.y >= k_texels) {
    return;
  }

  const int src_b = (uBlock.sizes.y == 1) ? 0 : pos.z;

  vec4 out_t = vec4(0.0f);
  for (int lane = 0; lane < 4; ++lane) {
    const int k = pos.y * 4 + lane;
    if (k < uBlock.sizes.z) {
      out_t[lane] = texelFetch(uInput, ivec3(pos.x, k, src_b / 4), 0)[src_b % 4];
    }
  }

  imageStore(uOutput, pos, out_t);
}
