#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Signed-byte textures in and out.  The window maximum is order-preserving
// in the quantized domain, so no requantization is needed and the output
// inherits the input qparams.
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  ivec2 kernel;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int c_depth; // ceil(C / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Quantized 2D max pooling over the width and height axes: each output
 * position takes the lane-wise maximum over the in-bounds window elements;
 * padded positions are skipped, matching the float kernel's boundary rule.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const ivec2 start = ivec2(pos.x, pos.y) * uBlock.stride - uBlock.padding;

  ivec4 acc = ivec4(-128);

  for (int ky = 0; ky < uBlock.kernel.y; ++ky) {
    for (int kx = 0; kx < uBlock.kernel.x; ++kx) {
      const ivec2 in_pos =
          start + ivec2(kx, ky) * uBlock.dilation;
      const bool inside = all(greaterThanEqual(in_pos, ivec2(0))) &&
          in_pos.x < uBlock.in_sizes.x &&
          in_pos.y < uBlock.in_sizes.y;
      if (inside) {
        acc = max(acc, imageLoad(uInput, ivec3(in_pos.x, in_pos.y, pos.z)));
      }
    }
  }

  imageStore(uOutput, pos, acc);
}
