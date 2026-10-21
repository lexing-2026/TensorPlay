#version 450 core
// clang-format off
#define OP(X, Y) ${OPERATOR}
#define NEUTRAL ${NEUTRAL}
#define COLLAPSE(V) ${COLLAPSE}
// clang-format on

layout(std430) buffer;

/*
 * Single-axis extremum of a signed-word payload held in an image.  The fold
 * keeps the width the payload names: a count past the exact reach of single
 * precision must not be rounded on its way into the answer.  Width, height and
 * batch axes fold lane-wise; a channel axis walks the texel depth of one batch
 * and collapses four lanes into one value, and lanes past the channel count
 * fold the walk's identity, so padding never wins.
 */
layout(set = 0, binding = 0, rgba32i) uniform highp restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, rgba32i) uniform highp restrict readonly iimage3D uInput;
layout(set = 0, binding = 2) uniform highp restrict Block {
  ivec4 in_sizes;
  ivec4 out_sizes;
  int axis; // 0: W, 1: H, 2: C, 3: N
  int in_c_depth;
  int out_c_depth;
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  ivec4 acc = ivec4(NEUTRAL);
  if (uBlock.axis == 0) {
    for (int x = 0; x < uBlock.in_sizes.x; ++x) {
      acc = OP(acc, imageLoad(uInput, ivec3(x, pos.y, pos.z)));
    }
    imageStore(uOutput, pos, acc);
  } else if (uBlock.axis == 1) {
    for (int y = 0; y < uBlock.in_sizes.y; ++y) {
      acc = OP(acc, imageLoad(uInput, ivec3(pos.x, y, pos.z)));
    }
    imageStore(uOutput, pos, acc);
  } else if (uBlock.axis == 2) {
    // Channel: every batch's lanes fold into one answer that the output texel
    // replicates, so a lane past the channel count has to fold the identity.
    const int n = pos.z / uBlock.out_c_depth;
    for (int c4 = 0; c4 < uBlock.in_c_depth; ++c4) {
      ivec4 value = imageLoad(
          uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + c4));
      const int base = c4 * 4;
      for (int lane = 0; lane < 4; ++lane) {
        if (base + lane >= uBlock.in_sizes.z) {
          value[lane] = NEUTRAL;
        }
      }
      acc = OP(acc, value);
    }
    imageStore(uOutput, ivec3(pos.x, pos.y, n), ivec4(COLLAPSE(acc)));
  } else {
    // Batch: the texel depth carries the channels, so the walk steps over the
    // batches that share this texel.
    const int c4 = pos.z % uBlock.in_c_depth;
    for (int n = 0; n < uBlock.in_sizes.w; ++n) {
      acc = OP(
          acc,
          imageLoad(uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + c4)));
    }
    imageStore(uOutput, pos, acc);
  }
}
