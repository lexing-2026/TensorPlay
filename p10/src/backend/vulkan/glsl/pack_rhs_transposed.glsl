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
  ivec4 sizes; // {N, C, H, W} logical sizes of the dense source
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Packs an (N x K) dense matrix holding B^T directly into height-packed
 * (K x N) planes for B: output texel (n, j) lane l gathers B[4j + l][n],
 * i.e. source element (n, 4j + l).  The transpose never materializes; the
 * gather folds it into the lane compaction, so the reduction axis runs
 * through the vec4 lanes of the packed planes.  Elements past the K edge
 * are stored as zeros so the packed tail never carries garbage into
 * downstream reductions.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int k_texels = (uBlock.sizes.w + 3) / 4;

  if (pos.x >= uBlock.sizes.z || pos.y >= k_texels) {
    return;
  }

  vec4 out_t = vec4(0.0f);
  for (int lane = 0; lane < 4; ++lane) {
    const int k = pos.y * 4 + lane;
    if (k < uBlock.sizes.w) {
      out_t[lane] = texelFetch(uInput, ivec3(k, pos.x, 0), 0).x;
    }
  }

  imageStore(uOutput, pos, out_t);
}
