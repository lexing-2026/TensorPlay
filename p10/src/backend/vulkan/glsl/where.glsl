#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define DTYPE ${DTYPE}
// clang-format on

layout(std430) buffer;

#include "texel_access.h"

$if DTYPE == "int":
  // Signed-word payload planes; the condition stays a signed-byte texture.
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
  layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uCondition;
  layout(set = 0, binding = 2, FORMAT) uniform PRECISION restrict readonly iimage3D uInput;
  layout(set = 0, binding = 3, FORMAT) uniform PRECISION restrict readonly iimage3D uOther;
  layout(set = 0, binding = 4) uniform PRECISION restrict Block {
    // (W, H, C, N) sizes of the output and of each operand
    ivec4 out_sizes;
    ivec4 cond_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
    int c_depth;
    int fill;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
  layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uCondition;
  layout(set = 0, binding = 2) uniform PRECISION sampler3D uInput;
  layout(set = 0, binding = 3) uniform PRECISION sampler3D uOther;
  layout(set = 0, binding = 4) uniform PRECISION restrict Block {
    // (W, H, C, N) sizes of the output and of each operand
    ivec4 out_sizes;
    ivec4 cond_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
    int c_depth;
    int fill;
  }
  uBlock;
// end-of-dtype-branches

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * where(): the condition is a Bool payload in a signed-byte texture, and the
 * two operands share the output's texel geometry (scalar variants were
 * folded into tensors by the caller).  Lanes with a nonzero condition byte
 * take the self operand, the rest take other.
 *
 * The three planes need not carry the output's shape: each is addressed
 * through its own sizes, so a dimension of one repeats across the matching
 * output range and a missing leading dimension repeats the whole plane.  An
 * operand with a single logical channel keeps its value in the first lane of
 * its texel, which then feeds every lane of the output.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const ivec3 cond_pos = map_output_pos_to_input_pos(
      pos, uBlock.out_sizes, uBlock.cond_sizes);
  const ivec4 cond_raw = imageLoad(uCondition, cond_pos);
  const ivec4 cond_lanes = (uBlock.out_sizes.z != uBlock.cond_sizes.z)
      ? ivec4(cond_raw.x)
      : cond_raw;

  const ivec3 input_pos = map_output_pos_to_input_pos(
      pos, uBlock.out_sizes, uBlock.input_sizes);
  const ivec3 other_pos = map_output_pos_to_input_pos(
      pos, uBlock.out_sizes, uBlock.other_sizes);

  // clang-format off
  $if DTYPE == "int":
    const ivec4 cond = ivec4(notEqual(cond_lanes, ivec4(0)));
    const ivec4 a_raw = imageLoad(uInput, input_pos);
    const ivec4 b_raw = imageLoad(uOther, other_pos);
    const ivec4 a =
        (uBlock.out_sizes.z != uBlock.input_sizes.z) ? ivec4(a_raw.x) : a_raw;
    const ivec4 b =
        (uBlock.out_sizes.z != uBlock.other_sizes.z) ? ivec4(b_raw.x) : b_raw;
    imageStore(uOutput, pos, a * cond + b * (1 - cond));
  $else:
    const bvec4 cond = notEqual(cond_lanes, ivec4(0));
    const vec4 a_raw = texelFetch(uInput, input_pos, 0);
    const vec4 b_raw = texelFetch(uOther, other_pos, 0);
    const vec4 a =
        (uBlock.out_sizes.z != uBlock.input_sizes.z) ? vec4(a_raw.x) : a_raw;
    const vec4 b =
        (uBlock.out_sizes.z != uBlock.other_sizes.z) ? vec4(b_raw.x) : b_raw;
    // Selects per lane.  An arithmetic recombination (a*m - b*(1-m)) cannot
    // represent infinities: the rejected operand multiplied by zero yields
    // 0 * 0 or 0 * minus_infinity forms that turn into not-a-number and
    // poison the surviving lanes, so the selection must be exact.
    imageStore(
        uOutput,
        pos,
        vec4(cond.x ? a.x : b.x, cond.y ? a.y : b.y, cond.z ? a.z : b.z, cond.w ? a.w : b.w));
  // clang-format on
}
