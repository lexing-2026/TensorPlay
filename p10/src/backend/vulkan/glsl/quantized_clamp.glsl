#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Signed-byte textures in and out; the clamp bounds are given in the
// dequantized domain and the result is requantized into the output qparams.
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
  float in_scale;
  int in_zero_point;
  float inv_out_scale;
  int out_zero_point;
  int has_min;
  int has_max;
  float min_value;
  float max_value;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Quantized clamp: dequantize, clamp against the float bounds, requantize
 * with round-to-nearest-even into [-128, 127].
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  const ivec4 q = imageLoad(uInput, pos);
  vec4 y = (vec4(q) - vec4(float(uBlock.in_zero_point))) * uBlock.in_scale;

  if (uBlock.has_min != 0) {
    y = max(y, vec4(uBlock.min_value));
  }
  if (uBlock.has_max != 0) {
    y = min(y, vec4(uBlock.max_value));
  }

  const ivec4 rounded =
      ivec4(roundEven(y * uBlock.inv_out_scale)) + ivec4(uBlock.out_zero_point);
  imageStore(uOutput, pos, clamp(rounded, ivec4(-128), ivec4(127)));
}
