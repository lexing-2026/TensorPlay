#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 in_sizes;
  int axis;
  int c_depth;
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * One invocation scans one packed line in linear work. Channel scans carry
 * the prefix between texels and between the four lanes of each texel.
 */
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  int depth = uBlock.c_depth;
  if (pos.x >= uBlock.in_sizes.x || pos.y >= uBlock.in_sizes.y ||
      pos.z >= uBlock.in_sizes.w * depth) return;
  vec4 acc = vec4(1);
  if (uBlock.axis == 0) {
    if (pos.x != 0) return;
    for (int x = 0; x < uBlock.in_sizes.x; ++x) {
      ivec3 at = ivec3(x, pos.y, pos.z);
      acc *= texelFetch(uInput, at, 0);
      imageStore(uOutput, at, acc);
    }
  } else if (uBlock.axis == 1) {
    if (pos.y != 0) return;
    for (int y = 0; y < uBlock.in_sizes.y; ++y) {
      ivec3 at = ivec3(pos.x, y, pos.z);
      acc *= texelFetch(uInput, at, 0);
      imageStore(uOutput, at, acc);
    }
  } else if (uBlock.axis == 2) {
    int n = pos.z;
    if (n >= uBlock.in_sizes.w) return;
    float prefix = 1.0;
    for (int c4 = 0; c4 < depth; ++c4) {
      ivec3 at = ivec3(pos.x, pos.y, n * depth + c4);
      vec4 value = texelFetch(uInput, at, 0);
      vec4 result = vec4(0);
      for (int lane = 0; lane < 4 && c4 * 4 + lane < uBlock.in_sizes.z; ++lane) {
        prefix *= value[lane];
        result[lane] = prefix;
      }
      imageStore(uOutput, at, result);
    }
  } else {
    if (pos.z >= depth) return;
    for (int n = 0; n < uBlock.in_sizes.w; ++n) {
      ivec3 at = ivec3(pos.x, pos.y, n * depth + pos.z);
      acc *= texelFetch(uInput, at, 0);
      imageStore(uOutput, at, acc);
    }
  }
}
