#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define HAS_BIAS ${HAS_BIAS}
#define TILE ${OUTPUT_TILE}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 in_sizes; // (W, H, C, N) logical sizes
  ivec4 out_sizes; // (OW, OH, C, N) logical sizes
  ivec4 weight_sizes; // (KW, KH, ..., ...) with .z = KH, .w = KW
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int c_depth; // ceil(C / 4)
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Depthwise 2D convolution (groups == channels): every output channel filters
 * its own input channel with a 2D kernel.
 *
 * The packed weight texture stores four consecutive channels per texel:
 * texel (kx, ky, z) lanes hold w[c = 4 * (z % c_depth) + lane][ky][kx], so
 * one fetch covers a whole channel group of the kernel.  One invocation
 * computes a TILE x TILE block of output positions; the weight fetch is
 * shared across the block, and input fetches at positions beyond the output
 * edges return zeros whose stores are discarded.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x * TILE >= uBlock.out_sizes.x ||
      pos.y * TILE >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const int n = pos.z / uBlock.c_depth;
  const int c4 = pos.z % uBlock.c_depth;

  vec4 acc[TILE][TILE];
  for (int i = 0; i < TILE; ++i) {
    for (int j = 0; j < TILE; ++j) {
      acc[i][j] = vec4(0.0f);
    }
  }

  for (int ky = 0; ky < uBlock.weight_sizes.z; ++ky) {
    int ih[TILE];
    bool ih_ok[TILE];
    for (int i = 0; i < TILE; ++i) {
      ih[i] = (pos.y * TILE + i) * uBlock.stride.y - uBlock.padding.y +
          ky * uBlock.dilation.y;
      ih_ok[i] = ih[i] >= 0 && ih[i] < uBlock.in_sizes.y;
    }
    for (int kx = 0; kx < uBlock.weight_sizes.w; ++kx) {
      // Four consecutive channels of one kernel tap.
      const vec4 w = texelFetch(uWeight, ivec3(kx, ky, pos.z), 0);
      int iw[TILE];
      bool iw_ok[TILE];
      for (int j = 0; j < TILE; ++j) {
        iw[j] = (pos.x * TILE + j) * uBlock.stride.x - uBlock.padding.x +
            kx * uBlock.dilation.x;
        iw_ok[j] = iw[j] >= 0 && iw[j] < uBlock.in_sizes.x;
      }
      for (int i = 0; i < TILE; ++i) {
        for (int j = 0; j < TILE; ++j) {
          if (ih_ok[i] && iw_ok[j]) {
            const vec4 v = texelFetch(
                uInput, ivec3(iw[j], ih[i], n * uBlock.c_depth + c4), 0);
            acc[i][j] += v * w;
          }
        }
      }
    }
  }

  // clang-format off
  $if HAS_BIAS:
    const vec4 b = param_vec(uBias, c4, uBlock.in_sizes.z);
    for (int i = 0; i < TILE; ++i) {
      for (int j = 0; j < TILE; ++j) {
        acc[i][j] += b;
      }
    }
  // clang-format on

  for (int i = 0; i < TILE; ++i) {
    for (int j = 0; j < TILE; ++j) {
      imageStore(
          uOutput,
          ivec3(pos.x * TILE + j, pos.y * TILE + i, pos.z),
          acc[i][j]);
    }
  }
}
