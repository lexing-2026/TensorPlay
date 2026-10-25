#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;
#include "channel_lanes.h"

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 in_sizes; // (W, H, C, N) of the input
  ivec4 out_sizes; // (W, H, C, N) of the output
  int axis; // 0: W, 1: H, 2: C, 3: N
  int in_c_depth; // ceil(in C / 4)
  int out_c_depth; // ceil(out C / 4)
  int count; // elements along the reduced axis
  int correction; // delta degrees of freedom
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Single-axis variance: each invocation accumulates the line mean and mean
 * square, then combines them into sum((x - mean)^2) / (count - correction).
 * Lane treatment follows the matching sum shader: channel reductions
 * collapse all lanes, every other axis reduces lanes independently.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  // Resolve the reduce step/length for this axis.
  ivec3 step;
  if (uBlock.axis == 0) {
    step = ivec3(1, 0, 0);
  } else if (uBlock.axis == 1) {
    step = ivec3(0, 1, 0);
  } else if (uBlock.axis == 2) {
    step = ivec3(0, 0, 1);
  } else {
    step = ivec3(0, 0, uBlock.in_c_depth);
  }

  vec4 mean = vec4(0.0f);
  vec4 sq = vec4(0.0f);

  if (uBlock.axis == 2) {
    // Channel: every lane of every channel texel feeds the same line, so the
    // line mean is collapsed across the lanes before it is squared.  Squaring
    // the per-lane means would drop the cross-lane terms of the deviation sum.
    const int n = pos.z;
    vec4 lane_sum = vec4(0.0f);
    for (int c4 = 0; c4 < uBlock.in_c_depth; ++c4) {
      lane_sum += guarded_lanes(
          texelFetch(
              uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + c4), 0),
          c4 * 4, uBlock.in_sizes.z, 0.0);
    }
    const float line_mean =
        (lane_sum.x + lane_sum.y + lane_sum.z + lane_sum.w) /
        float(uBlock.count);
    float dev_sq = 0.0f;
    for (int c4 = 0; c4 < uBlock.in_c_depth; ++c4) {
      const vec4 d = guarded_lanes(
          texelFetch(
              uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + c4), 0) -
          vec4(line_mean),
          c4 * 4, uBlock.in_sizes.z, 0.0);
      dev_sq += dot(d, d);
    }
    const float var =
        dev_sq / float(max(uBlock.count - uBlock.correction, 1));
    imageStore(uOutput, ivec3(pos.x, pos.y, n), vec4(var));
    return;
  }

  for (int i = 0; i < uBlock.count; ++i) {
    const vec4 v = texelFetch(uInput, pos + step * i, 0);
    mean += v;
    sq += v * v;
  }
  mean /= float(uBlock.count);
  const vec4 var = (sq / float(uBlock.count) - mean * mean) *
      float(uBlock.count) /
      float(max(uBlock.count - uBlock.correction, 1));
  imageStore(uOutput, pos, var);
}
