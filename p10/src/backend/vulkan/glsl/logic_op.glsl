#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

#include "texel_access.h"

/*
 * Elementwise logic over Bool planes.
 *
 * A Bool payload occupies one signed byte per element in a four-lane texel, so
 * every operator works lane-wise on the texel as an integer vector and the
 * result stays a Bool texel of the same shape.  A lane that is not exactly zero
 * counts as true: each operand is first reduced to 0 or 1, and the operators
 * are arithmetic on those reduced lanes, which spells and, or, exclusive or and
 * negation without depending on integer bit operations.
 *
 * The operands need not share the output's shape: each one is addressed through
 * its own sizes, so a dimension of one repeats across the corresponding output
 * range and a missing leading dimension repeats the whole plane.  An operand
 * with a single logical channel carries its value in the first lane of its
 * texel, which is then the operand of every lane of the output.
 */
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
$if UNARY:
  layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uInput;
  layout(set = 0, binding = 2) uniform PRECISION restrict Block {
    ivec4 output_sizes;
    ivec4 input_sizes;
  }
  uArgs;
$else:
  layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uInput;
  layout(set = 0, binding = 2, rgba8i) uniform PRECISION restrict readonly iimage3D uOther;
  layout(set = 0, binding = 3) uniform PRECISION restrict Block {
    ivec4 output_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
  }
  uArgs;
// End of the operand bindings.

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Whether each lane of a texel carries a nonzero byte, as 0 or 1.
ivec4 truth(const ivec4 lanes) {
  return clamp(abs(lanes), ivec4(0), ivec4(1));
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const ivec3 output_extents = ivec3(
      uArgs.output_sizes.xy,
      uArgs.output_sizes.w * int(ceil(uArgs.output_sizes.z / 4.0)));

  if (any(greaterThanEqual(pos, output_extents))) {
    return;
  }

  const ivec3 input_pos =
      map_output_pos_to_input_pos(pos, uArgs.output_sizes, uArgs.input_sizes);
  const ivec4 input_raw = imageLoad(uInput, input_pos);
  const ivec4 input_texel = (uArgs.output_sizes.z != uArgs.input_sizes.z)
      ? truth(ivec4(input_raw.x))
      : truth(input_raw);

$if UNARY:
  imageStore(uOutput, pos, OP(input_texel, ivec4(0)));
$else:
  const ivec3 other_pos =
      map_output_pos_to_input_pos(pos, uArgs.output_sizes, uArgs.other_sizes);
  const ivec4 other_raw = imageLoad(uOther, other_pos);
  const ivec4 other_texel = (uArgs.output_sizes.z != uArgs.other_sizes.z)
      ? truth(ivec4(other_raw.x))
      : truth(other_raw);
  imageStore(uOutput, pos, OP(input_texel, other_texel));
}
