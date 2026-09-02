#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
  float p0;
  float p1;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Dead-band indicator for the hardshrink gradient: 1 where the input
 * magnitude is within the threshold (the shrunken band), 0 outside it.
 * Multiplying the incoming gradient by (1 - mask) zeroes the band and
 * passes everything else through unchanged.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  const vec4 x = texelFetch(uInput, pos, 0);
  const vec4 p0 = vec4(uBlock.p0);
  const vec4 mask =
      mix(vec4(0.0f), vec4(1.0f), lessThanEqual(abs(x), p0));
  imageStore(uOutput, pos, mask);
}
