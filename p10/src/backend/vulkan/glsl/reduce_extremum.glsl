#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, Y) ${OPERATOR}
#define COLLAPSE(V) ${COLLAPSE}
#define NEUTRAL ${NEUTRAL}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 in_sizes; // (W, H, C, N) of the input
  ivec4 out_sizes; // (W, H, C, N) of the output
  int axis; // 0: W, 1: H, 2: C, 3: N
  int in_c_depth;
  int out_c_depth;
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Single-axis reduction with a caller-selected lane operator.  Width,
 * height, and batch reductions fold lane-wise; the texel z coordinate
 * carries over directly.  A channel reduction walks the texel depth of
 * one batch: padded lanes are substituted with the operator's neutral
 * element so they never contaminate the fold, and the four lanes then
 * collapse into one value that is replicated across the output lanes.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  vec4 acc;

  if (uBlock.axis == 0) {
    acc = texelFetch(uInput, ivec3(0, pos.y, pos.z), 0);
    for (int x = 1; x < uBlock.in_sizes.x; ++x) {
      acc = OP(acc, texelFetch(uInput, ivec3(x, pos.y, pos.z), 0));
    }
    imageStore(uOutput, pos, acc);
  } else if (uBlock.axis == 1) {
    acc = texelFetch(uInput, ivec3(pos.x, 0, pos.z), 0);
    for (int y = 1; y < uBlock.in_sizes.y; ++y) {
      acc = OP(acc, texelFetch(uInput, ivec3(pos.x, y, pos.z), 0));
    }
    imageStore(uOutput, pos, acc);
  } else if (uBlock.axis == 2) {
    // Channel: pos.z addresses (n * in_c_depth + c4) of the input and
    // (n * out_c_depth + c4) of the output.  All channels of the batch
    // collapse into one value per texel position.
    const int n = pos.z / uBlock.out_c_depth;
    const vec4 lane_mask = vec4(lessThan(
        ivec4(0, 1, 2, 3), ivec4(uBlock.in_sizes.z)));
    const vec4 first_texel =
        mix(vec4(NEUTRAL), texelFetch(uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth), 0),
            lane_mask);
    acc = first_texel;
    for (int z = n * uBlock.in_c_depth + 1;
         z < (n + 1) * uBlock.in_c_depth; ++z) {
      const vec4 v = mix(
          vec4(NEUTRAL), texelFetch(uInput, ivec3(pos.x, pos.y, z), 0),
          vec4(lessThan(
              ivec4(0, 1, 2, 3) + (z - n * uBlock.in_c_depth) * 4,
              ivec4(uBlock.in_sizes.z))));
      acc = OP(acc, v);
    }
    imageStore(
        uOutput, ivec3(pos.x, pos.y, n), vec4(COLLAPSE(acc)));
  } else {
    // Batch: pos.z addresses (n0 * in_c_depth + c4) of the input.
    const int c4 = pos.z % uBlock.in_c_depth;
    acc = texelFetch(uInput, ivec3(pos.x, pos.y, c4), 0);
    for (int n = 1; n < uBlock.in_sizes.w; ++n) {
      acc = OP(
          acc,
          texelFetch(uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + c4), 0));
    }
    imageStore(uOutput, pos, acc);
  }
}
