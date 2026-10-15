#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uGradOut;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  // (W, H, C, N) sizes of the grad output (pooled) tensor
  ivec4 out_sizes;
  ivec2 kernel;
  ivec2 stride;
  ivec2 padding;
  int c_depth; // ceil(C / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Gradient of 2D max pooling.  Each input position examines every pooling
 * window that covers it, recomputes the lane-wise window maximum, and
 * accumulates the incoming gradient only when the position is the first
 * window element (in row-major scan order) to reach that maximum, matching
 * the argmax convention of the forward kernel.  All four texel lanes are
 * resolved independently because the forward reduces each channel lane
 * separately.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.in_sizes.x || pos.y >= uBlock.in_sizes.y ||
      pos.z >= uBlock.in_sizes.w * uBlock.c_depth) {
    return;
  }

  const ivec2 in_p = ivec2(pos.x, pos.y);
  const vec4 in_val = texelFetch(uInput, pos, 0);

  // Windows whose [start, start+kernel) span contains in_p:
  //   start = out * stride - padding  =>  out in
  //   [(in_p + padding - kernel + 1 + stride - 1) / stride,
  //    (in_p + padding) / stride]
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
      const ivec2 start =
          ivec2(ow, oh) * uBlock.stride - uBlock.padding;

      // First recompute the lane-wise maximum of this window, ignoring
      // padded positions exactly like the forward kernel.
      vec4 wmax = vec4(-3.0e38f);
      for (int ky = 0; ky < uBlock.kernel.y; ++ky) {
        for (int kx = 0; kx < uBlock.kernel.x; ++kx) {
          const ivec2 win_p = start + ivec2(kx, ky);
          const bool inside = all(greaterThanEqual(win_p, ivec2(0))) &&
              win_p.x < uBlock.in_sizes.x &&
              win_p.y < uBlock.in_sizes.y;
          const vec4 v = inside
              ? texelFetch(uInput, ivec3(win_p.x, win_p.y, pos.z), 0)
              : vec4(-3.0e38f);
          wmax = max(wmax, v);
        }
      }

      // Whether this input position is the first element (row-major scan
      // order) in the window that reaches wmax, per lane.
      const int cur_off =
          (in_p.y - start.y) * uBlock.kernel.x + (in_p.x - start.x);
      bvec4 first = bvec4(true);
      int off = 0;
      for (int ky = 0; ky < uBlock.kernel.y && any(first); ++ky) {
        for (int kx = 0; kx < uBlock.kernel.x && any(first); ++kx) {
          if (off >= cur_off) break;
          const ivec2 win_p = start + ivec2(kx, ky);
          const bool inside = all(greaterThanEqual(win_p, ivec2(0))) &&
              win_p.x < uBlock.in_sizes.x &&
              win_p.y < uBlock.in_sizes.y;
          const vec4 v = inside
              ? texelFetch(uInput, ivec3(win_p.x, win_p.y, pos.z), 0)
              : vec4(-3.0e38f);
          first = bvec4(
              first.x && v.x < wmax.x,
              first.y && v.y < wmax.y,
              first.z && v.z < wmax.z,
              first.w && v.w < wmax.w);
          ++off;
        }
      }

      const vec4 g = texelFetch(uGradOut, ivec3(ow, oh, pos.z), 0);
      // The window gradient belongs to the first position that actually
      // reaches the maximum; the "first" flags only say no earlier window
      // element matched, so they must be ANDed with an equality check on
      // the current input value.
      const bvec4 is_argmax = bvec4(
          first.x && in_val.x == wmax.x,
          first.y && in_val.y == wmax.y,
          first.z && in_val.z == wmax.z,
          first.w && in_val.w == wmax.w);
      grad_acc += vec4(is_argmax) * g;
    }
  }

  imageStore(uOutput, pos, grad_acc);
}