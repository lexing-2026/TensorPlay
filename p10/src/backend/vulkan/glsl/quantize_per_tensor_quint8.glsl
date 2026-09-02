#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Unsigned-byte variant of the per-tensor affine quantization: rounds the
// scaled value to the nearest integer (ties-to-even) with the caller's
// clamping range and stores the code into a uint-byte texture.
layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D uQuantizedOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uFloatInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
  float inv_scale;
  int zero_point;
  int quant_min;
  int quant_max;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  const vec4 x = texelFetch(uFloatInput, pos, 0);
  const ivec4 rounded =
      ivec4(roundEven(x * uBlock.inv_scale)) + ivec4(uBlock.zero_point);
  const ivec4 q =
      clamp(rounded, ivec4(uBlock.quant_min), ivec4(uBlock.quant_max));
  imageStore(uQuantizedOutput, pos, q);
}
