#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Signed-byte texture in, float texture out.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uFloatOutput;
layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uQuantizedInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
  float scale;
  int zero_point;
  int fill0;
  int fill1;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Per-tensor affine dequantization: recovers the real value from the stored
 * byte as (q - zero_point) * scale.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  const ivec4 q = imageLoad(uQuantizedInput, pos);
  const vec4 x = (vec4(q) - vec4(float(uBlock.zero_point))) * uBlock.scale;
  imageStore(uFloatOutput, pos, x);
}
