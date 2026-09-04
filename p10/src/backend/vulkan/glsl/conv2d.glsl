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
  ivec4 weight_sizes; // (KW, KH, z_extent, KW) with .z = KH, .w = KW
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int in_c_depth; // ceil(C / 4)
  int out_c_depth; // ceil(O / 4)
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Grouped (single group) 2D convolution in gather form: one invocation
 * computes the four output channels covered by one texel at one spatial
 * position.
 *
 * Packed weight layout: texel (kx, ky, (o4 * C4 + ic4) * 4 + lane) carries,
 * in its four components, w[oc = 4*o4 + comp][ic = 4*ic4 + lane][ky][kx].
 * One fetch therefore covers one kernel tap of one packed input channel
 * against all four output channels, and the inner loop reduces to four fma
 * sweeps per input-channel group (the unpacked form needed four fetches per
 * single input channel).  The four weight texels of one tap sit at
 * consecutive z, so they stream from the same cache line.
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

  vec4 acc = vec4(0.0f);

  for (int ky = 0; ky < uBlock.weight_sizes.z; ++ky) {
    const int ih =
        oh * uBlock.stride.y - uBlock.padding.y + ky * uBlock.dilation.y;
    if (ih < 0 || ih >= uBlock.in_sizes.y) {
      continue;
    }
    for (int kx = 0; kx < uBlock.weight_sizes.w; ++kx) {
      const int iw =
          ow * uBlock.stride.x - uBlock.padding.x + kx * uBlock.dilation.x;
      if (iw < 0 || iw >= uBlock.in_sizes.x) {
        continue;
      }
      for (int ci4 = 0; ci4 < uBlock.in_c_depth; ++ci4) {
        const int in_z = n * uBlock.in_c_depth + ci4;
        const vec4 v = texelFetch(uInput, ivec3(iw, ih, in_z), 0);
        const int wz = (o4 * uBlock.in_c_depth + ci4) * 4;
        acc = fma(v.xxxx, texelFetch(uWeight, ivec3(kx, ky, wz + 0), 0), acc);
        acc = fma(v.yyyy, texelFetch(uWeight, ivec3(kx, ky, wz + 1), 0), acc);
        acc = fma(v.zzzz, texelFetch(uWeight, ivec3(kx, ky, wz + 2), 0), acc);
        acc = fma(v.wwww, texelFetch(uWeight, ivec3(kx, ky, wz + 3), 0), acc);
      }
    }
  }

  // clang-format off
  $if HAS_BIAS:
    acc += param_vec(uBias, o4, uBlock.out_sizes.z);
  // clang-format on

  imageStore(uOutput, pos, acc);
}
