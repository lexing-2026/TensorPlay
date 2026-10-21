#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

#include "texel_access.h"

layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
$if DTYPE == "float":
  layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
  layout(set = 0, binding = 2) uniform PRECISION sampler3D uOther;
$elif DTYPE == "uint8":
  layout(set = 0, binding = 1) uniform PRECISION usampler3D uInput;
  layout(set = 0, binding = 2) uniform PRECISION usampler3D uOther;
$else:
  layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;
  layout(set = 0, binding = 2) uniform PRECISION isampler3D uOther;
// End of typed inputs.
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  // (W, H, C, N) sizes of each operand, so an operand is addressed through its
  // own geometry rather than the output's
  ivec4 input_sizes;
  ivec4 other_sizes;
  int c_depth;
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Elementwise comparison: the result is a Bool payload in a signed-byte
 * texture, one 0/1 byte per lane.  OP returns a boolean vector stored as
 * integer 0/1 lanes.
 *
 * The operands need not share the output's shape: each one is fetched at the
 * position its own sizes give it, so a dimension of one repeats across the
 * matching output range and a missing leading dimension repeats the whole
 * plane.  An operand with a single logical channel carries its value in the
 * first lane of its texel, which then answers for every lane of the output.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const ivec3 input_pos =
      map_output_pos_to_input_pos(pos, uBlock.out_sizes, uBlock.input_sizes);
  const ivec3 other_pos =
      map_output_pos_to_input_pos(pos, uBlock.out_sizes, uBlock.other_sizes);

$if DTYPE == "float":
  const vec4 a = (uBlock.input_sizes.z == 1 ||
                  uBlock.out_sizes.z != uBlock.input_sizes.z)
      ? texelFetch(uInput, input_pos, 0).xxxx
      : texelFetch(uInput, input_pos, 0);
  const vec4 b = (uBlock.other_sizes.z == 1 ||
                  uBlock.out_sizes.z != uBlock.other_sizes.z)
      ? texelFetch(uOther, other_pos, 0).xxxx
      : texelFetch(uOther, other_pos, 0);
$elif DTYPE == "uint8":
  const uvec4 a = (uBlock.input_sizes.z == 1 ||
                   uBlock.out_sizes.z != uBlock.input_sizes.z)
      ? texelFetch(uInput, input_pos, 0).xxxx
      : texelFetch(uInput, input_pos, 0);
  const uvec4 b = (uBlock.other_sizes.z == 1 ||
                   uBlock.out_sizes.z != uBlock.other_sizes.z)
      ? texelFetch(uOther, other_pos, 0).xxxx
      : texelFetch(uOther, other_pos, 0);
$else:
  const ivec4 a = (uBlock.input_sizes.z == 1 ||
                   uBlock.out_sizes.z != uBlock.input_sizes.z)
      ? ivec4(texelFetch(uInput, input_pos, 0).x)
      : texelFetch(uInput, input_pos, 0);
  const ivec4 b = (uBlock.other_sizes.z == 1 ||
                   uBlock.out_sizes.z != uBlock.other_sizes.z)
      ? ivec4(texelFetch(uOther, other_pos, 0).x)
      : texelFetch(uOther, other_pos, 0);
// End of the operand loads.

  imageStore(uOutput, pos, ivec4(OP(a, b)));
}
