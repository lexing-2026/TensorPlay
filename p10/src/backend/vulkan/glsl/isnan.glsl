#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
  int fill0;
  int fill1;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * NaN probe: writes 1 where the payload is not equal to itself, 0 otherwise,
 * one boolean lane per element.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  const vec4 v = texelFetch(uInput, pos, 0);
  imageStore(uOutput, pos, ivec4(notEqual(v, v)));
}
