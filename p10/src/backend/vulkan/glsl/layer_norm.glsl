#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define WEIGHT_MODE ${WEIGHT_MODE}
#define BIAS_MODE ${BIAS_MODE}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  int c_depth; // ceil(C / 4)
  int channels;
  // Elements of the normalized span along W and H, per channel
  int span;
  // 1 when the normalized span includes the channel axis: statistics are
  // then shared by every lane of a batch, computed with a scalar collapse.
  int norm_channels;
  float eps;
  int weight_len; // element count of a 1d affine parameter
  int fill0;
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Layer normalization over the trailing normalized axes with optional
 * affine parameters.  Affine parameter addressing follows the parameter
 * rank: a full-shape parameter ({C,H,W}) is fetched per channel lane at the
 * invocation position; a 2d ({H,W}) or 1d ({W}) parameter is scalar for the
 * lane group and replicated; a 1d parameter is fetched through the texel x
 * axis with clamping.
 *
 * When the span excludes the channel axis, statistics are per texel lane;
 * when it includes channels, the statistics are shared by the whole batch
 * position: lanes are masked, collapsed into one scalar mean/variance pair,
 * and replicated for the transform.
 */

vec4 fetch_weight(const ivec3 pos, const int c4) {
  // clang-format off
  $if WEIGHT_MODE == 1:
    return texelFetch(uWeight, pos, 0);
  $elif WEIGHT_MODE == 2:
    return vec4(texelFetch(uWeight, ivec3(pos.x, pos.y, 0), 0).x);
  $elif WEIGHT_MODE == 3:
    return param_vec(uWeight, c4, uBlock.weight_len);
  $else:
    return vec4(1.0f);
  // clang-format on
}

vec4 fetch_bias(const ivec3 pos, const int c4) {
  // clang-format off
  $if BIAS_MODE == 1:
    return texelFetch(uBias, pos, 0);
  $elif BIAS_MODE == 2:
    return vec4(texelFetch(uBias, ivec3(pos.x, pos.y, 0), 0).x);
  $elif BIAS_MODE == 3:
    return param_vec(uBias, c4, uBlock.weight_len);
  $else:
    return vec4(0.0f);
  // clang-format on
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const vec4 v = texelFetch(uInput, pos, 0);
  const int c4 = pos.z % uBlock.c_depth;
  const vec4 w = fetch_weight(pos, c4);
  const vec4 b = fetch_bias(pos, c4);

  vec4 mean;
  vec4 var;

  if (uBlock.norm_channels == 0) {
    // The span maps onto the texel plane starting at this invocation's
    // position: a width-only span keeps y (and z) fixed; a (H, W) span
    // walks every y of the same z.
    vec4 acc = vec4(0.0f);
    vec4 sq = vec4(0.0f);
    for (int i = 0; i < uBlock.span; ++i) {
      const int x = i % uBlock.in_sizes.x;
      const int y = (uBlock.span == uBlock.in_sizes.x)
          ? pos.y
          : i / uBlock.in_sizes.x;
      const vec4 vi = texelFetch(uInput, ivec3(x, y, pos.z), 0);
      acc += vi;
      sq += vi * vi;
    }
    mean = acc / float(uBlock.span);
    var = sq / float(uBlock.span) - mean * mean;
  } else {
    const int n = pos.z / uBlock.c_depth;
    const vec4 lane_mask = vec4(lessThan(
        ivec4(0, 1, 2, 3), ivec4(uBlock.channels)));
    vec4 acc = vec4(0.0f);
    vec4 sq = vec4(0.0f);
    for (int c4i = 0; c4i < uBlock.c_depth; ++c4i) {
      for (int i = 0; i < uBlock.span; ++i) {
        const int x = i % uBlock.in_sizes.x;
        const int y = i / uBlock.in_sizes.x;
        const vec4 vi = lane_mask *
            texelFetch(
                uInput, ivec3(x, y, n * uBlock.c_depth + c4i), 0);
        acc += vi;
        sq += vi * vi;
      }
    }
    const float count = float(uBlock.span * uBlock.channels);
    const float m = dot(acc, vec4(1.0f)) / count;
    const float m2 = dot(sq, vec4(1.0f)) / count;
    mean = vec4(m);
    var = vec4(m2 - m * m);
  }

  imageStore(
      uOutput, pos, (v - mean) * inversesqrt(var + uBlock.eps) * w + b);
}
