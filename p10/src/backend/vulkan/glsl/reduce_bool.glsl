#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, Y) ${OPERATOR}
#define COLLAPSE(V) ${COLLAPSE}
#define NEUTRAL ${NEUTRAL}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, FORMAT) uniform PRECISION restrict readonly iimage3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 in_sizes;
  ivec4 out_sizes;
  int axis;
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
    const int n = pos.z / uBlock.out_c_depth;
    for (int c4 = 0; c4 < uBlock.in_c_depth; ++c4) {
      ivec4 v = imageLoad(
          uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + c4));
      const int base = c4 * 4;
      for (int lane = 0; lane < 4; ++lane) {
        if (base + lane >= uBlock.in_sizes.z) {
          v[lane] = NEUTRAL;
        }
      }
      acc = OP(acc, v);
    }
    imageStore(uOutput, ivec3(pos.x, pos.y, n), ivec4(COLLAPSE(acc)));
  } else {
    const int c4 = pos.z % uBlock.in_c_depth;
    for (int n = 0; n < uBlock.in_sizes.w; ++n) {
      acc = OP(
          acc,
          imageLoad(uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + c4)));
    }
    imageStore(uOutput, pos, acc);
  }
}
