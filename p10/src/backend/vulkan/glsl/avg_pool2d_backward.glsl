#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uGradOut;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  // (W, H, C, N) sizes of the grad output (pooled) tensor
  ivec4 out_sizes;
  ivec2 kernel;
  ivec2 stride;
  ivec2 padding;
  int c_depth; // ceil(C / 4)
  int count_include_pad;
  float divisor_override;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Gradient of 2D average pooling.  Each input position accumulates the
 * incoming gradient of every pooling window that covers it, scaled by the
 * same divisor the forward kernel used for that window (an explicit
 * override, the full window area when padded positions count, or the number
 * of in-bounds elements otherwise).
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.in_sizes.x || pos.y >= uBlock.in_sizes.y ||
      pos.z >= uBlock.in_sizes.w * uBlock.c_depth) {
    return;
  }

  const ivec2 in_p = ivec2(pos.x, pos.y);

  const int oh_lo = max(
      0, (in_p.y + uBlock.padding.y - uBlock.kernel.y + uBlock.stride.y) /
             uBlock.stride.y);
  const int oh_hi = min(
      uBlock.out_sizes.y - 1, (in_p.y + uBlock.padding.y) / uBlock.stride.y);
  const int ow_lo = max(
      0, (in_p.x + uBlock.padding.x - uBlock.kernel.x + uBlock.stride.x) /
             uBlock.stride.x);
  const int ow_hi = min(
      uBlock.out_sizes.x - 1, (in_p.x + uBlock.padding.x) / uBlock.stride.x);

  vec4 grad_acc = vec4(0.0f);

  for (int oh = oh_lo; oh <= oh_hi; ++oh) {
    for (int ow = ow_lo; ow <= ow_hi; ++ow) {
      const ivec2 start = ivec2(ow, oh) * uBlock.stride - uBlock.padding;

      float divisor = uBlock.divisor_override;
      if (divisor <= 0.0f) {
        if (uBlock.count_include_pad != 0) {
          divisor = float(uBlock.kernel.x * uBlock.kernel.y);
        } else {
          int count = 0;
          for (int ky = 0; ky < uBlock.kernel.y; ++ky) {
            for (int kx = 0; kx < uBlock.kernel.x; ++kx) {
              const ivec2 win_p = start + ivec2(kx, ky);
              const bool inside = all(greaterThanEqual(win_p, ivec2(0))) &&
                  win_p.x < uBlock.in_sizes.x &&
                  win_p.y < uBlock.in_sizes.y;
              count += inside ? 1 : 0;
            }
          }
          divisor = float(max(count, 1));
        }
      }

      const vec4 g = texelFetch(uGradOut, ivec3(ow, oh, pos.z), 0);
      grad_acc += g / divisor;
    }
  }

  imageStore(uOutput, pos, grad_acc);
}