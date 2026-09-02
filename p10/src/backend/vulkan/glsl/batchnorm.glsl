#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define HAS_WEIGHT ${HAS_WEIGHT}
#define HAS_BIAS ${HAS_BIAS}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION sampler3D uRunningMean;
layout(set = 0, binding = 5) uniform PRECISION sampler3D uRunningVar;
layout(set = 0, binding = 6) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  int c_depth; // ceil(C / 4)
  int channels;
  float eps;
  int fill0;
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Inference-mode batch normalization: (x - mean) / sqrt(var + eps) * weight
 * + bias with per-channel statistics and affine parameters broadcast along
 * width and height.  HAS_WEIGHT / HAS_BIAS fold the affine terms away when
 * the corresponding tensor is absent.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int c4 = pos.z % uBlock.c_depth;

  const vec4 v = texelFetch(uInput, pos, 0);
  const vec4 mean = param_vec(uRunningMean, c4, uBlock.channels);
  const vec4 var = param_vec(uRunningVar, c4, uBlock.channels);

  vec4 out_texel = (v - mean) * inversesqrt(var + uBlock.eps);

  // clang-format off
  $if HAS_WEIGHT:
    out_texel *= param_vec(uWeight, c4, uBlock.channels);
  // clang-format on

  // clang-format off
  $if HAS_BIAS:
    out_texel += param_vec(uBias, c4, uBlock.channels);
  // clang-format on

  imageStore(uOutput, pos, out_texel);
}
