#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uMask;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  int c_depth;
  float value; // scalar fill; the tensor form fetches uValue instead
  int fill;
}
uBlock;

// clang-format off
$if VALUE_FROM_TENSOR:
  layout(set = 0, binding = 4) uniform PRECISION sampler3D uValue;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * masked_fill: every output texel copies the input texel, with lanes where
 * the mask byte is nonzero replaced by the fill value.  The mask is a Bool
 * payload in a signed-byte texture; its channels share the input's packing,
 * so a texel-coordinate lookup yields one comparison per output lane.  The
 * fill value either rides the uniform block (scalar form) or comes from a
 * same-shape value tensor fetched at the same position (tensor form).
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const vec4 v = texelFetch(uInput, pos, 0);
  const ivec4 m = texelFetch(uMask, pos, 0);
  vec4 replacement = vec4(uBlock.value);
  // clang-format off
  $if VALUE_FROM_TENSOR:
    replacement = texelFetch(uValue, pos, 0);
  // clang-format on
  const vec4 selected = mix(v, replacement, vec4(notEqual(m, ivec4(0))));
  imageStore(uOutput, pos, selected);
}
