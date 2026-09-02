#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define HAS_BIAS ${HAS_BIAS}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 in_sizes; // (W, H, C, N) logical sizes
  ivec4 out_sizes; // (OW, OH, O, N) logical sizes
  ivec4 weight_sizes; // (KW, KH, ..., ...) with .z = KH, .w = KW
  ivec2 stride;
  ivec2 padding;
  ivec2 output_padding;
  int in_c_depth; // ceil(C / 4)
  int out_c_depth; // ceil(O / 4)
  int weight_c_depth; // ceil(O / 4), channel depth of the weight texture
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Transposed 2D convolution in gather form.  Each output element sums the
 * input positions whose scatter lands on it: kernel taps are visited and
 * the implied input coordinate must divide the stride evenly and stay in
 * bounds.  One invocation computes the four output channels of one texel.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x ||
      pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  const int ow = pos.x;
  const int oh = pos.y;
  const int n = pos.z / uBlock.out_c_depth;
  const int o4 = pos.z % uBlock.out_c_depth;
  const int o = o4 * 4;

  vec4 acc = vec4(0.0f);

  for (int ky = 0; ky < uBlock.weight_sizes.z; ++ky) {
    // oh = ih * stride_h - padding_h + ky  =>  ih = (oh + padding_h - ky) / stride_h
    const int ty = oh + uBlock.padding.y - ky;
    if (ty < 0 || ty % uBlock.stride.y != 0) {
      continue;
    }
    const int ih = ty / uBlock.stride.y;
    if (ih < 0 || ih >= uBlock.in_sizes.y) {
      continue;
    }
    for (int kx = 0; kx < uBlock.weight_sizes.w; ++kx) {
      const int tx = ow + uBlock.padding.x - kx;
      if (tx < 0 || tx % uBlock.stride.x != 0) {
        continue;
      }
      const int iw = tx / uBlock.stride.x;
      if (iw < 0 || iw >= uBlock.in_sizes.x) {
        continue;
      }
      for (int ci = 0; ci < uBlock.in_sizes.z; ++ci) {
        const float v = texelFetch(
            uInput,
            ivec3(iw, ih, n * uBlock.in_c_depth + ci / 4),
            0)[ci % 4];
        // Weight texture: {C, O, KH, KW} logical sizes map to
        // (W=KW, H=KH, C=O, N=C); z carries output channel and input
        // channel group, the lane the output channel within its group.
        const vec4 w = vec4(
            texelFetch(
                uWeight,
                ivec3(kx, ky, ci * uBlock.weight_c_depth + (o + 0) / 4),
                0)[(o + 0) % 4],
            texelFetch(
                uWeight,
                ivec3(kx, ky, ci * uBlock.weight_c_depth + (o + 1) / 4),
                0)[(o + 1) % 4],
            texelFetch(
                uWeight,
                ivec3(kx, ky, ci * uBlock.weight_c_depth + (o + 2) / 4),
                0)[(o + 2) % 4],
            texelFetch(
                uWeight,
                ivec3(kx, ky, ci * uBlock.weight_c_depth + (o + 3) / 4),
                0)[(o + 3) % 4]);
        acc += v * w;
      }
    }
  }

  // clang-format off
  $if HAS_BIAS:
    acc += param_vec(uBias, o4, uBlock.out_sizes.z);
  // clang-format on

  imageStore(uOutput, pos, acc);
}
