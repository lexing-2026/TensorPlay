#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define MAX_POOLING ${MAX_POOLING}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  ivec2 kernel;
  ivec2 stride;
  ivec2 padding;
  int c_depth; // ceil(C / 4)
  int count_include_pad; // divisor: full window area vs in-bounds count
  float divisor_override; // 0 keeps the computed divisor
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * 2D pooling over the width and height axes with a fixed window and stride.
 * MAX_POOLING selects the lane-wise maximum; otherwise the window average is
 * computed with the requested divisor: a caller-provided override, the full
 * window area (count includes padded positions), or the in-bounds element
 * count.  The batch axis maps straight onto the z coordinate.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const ivec2 start =
      ivec2(pos.x, pos.y) * uBlock.stride - uBlock.padding;

  // clang-format off
  $if MAX_POOLING:
    vec4 acc = vec4(-3.0e38f);
  $else:
    vec4 acc = vec4(0.0f);
    int count = 0;
  // clang-format on

  for (int ky = 0; ky < uBlock.kernel.y; ++ky) {
    for (int kx = 0; kx < uBlock.kernel.x; ++kx) {
      const ivec2 in_pos = start + ivec2(kx, ky);
      const bool inside = all(greaterThanEqual(in_pos, ivec2(0))) &&
          in_pos.x < uBlock.in_sizes.x &&
          in_pos.y < uBlock.in_sizes.y;
      const vec4 v = inside
          ? texelFetch(
                uInput, ivec3(in_pos.x, in_pos.y, pos.z), 0)
          : vec4(0.0f);
      // clang-format off
      $if MAX_POOLING:
        acc = max(acc, v);
      $else:
        count += inside ? 1 : 0;
        acc += v;
      // clang-format on
    }
  }

  // clang-format off
  $if MAX_POOLING:
    imageStore(uOutput, pos, acc);
  $else:
    float divisor = (uBlock.divisor_override > 0.0f)
        ? uBlock.divisor_override
        : (uBlock.count_include_pad != 0
            ? float(uBlock.kernel.x * uBlock.kernel.y)
            : float(max(count, 1)));
    imageStore(uOutput, pos, acc / divisor);
  // clang-format on
}
