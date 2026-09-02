#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  int c_depth; // ceil(C / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Adaptive 2D average pooling: every output position covers the window
 * [floor(i * H / OH), ceil((i+1) * H / OH)) on each spatial axis, so the
 * window geometry follows the requested output resolution instead of a
 * fixed kernel.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const int in_h = uBlock.in_sizes.y;
  const int in_w = uBlock.in_sizes.x;
  const int out_h = uBlock.out_sizes.y;
  const int out_w = uBlock.out_sizes.x;

  const int y0 = (pos.y * in_h) / out_h;
  const int y1 = ((pos.y + 1) * in_h + out_h - 1) / out_h;
  const int x0 = (pos.x * in_w) / out_w;
  const int x1 = ((pos.x + 1) * in_w + out_w - 1) / out_w;

  vec4 acc = vec4(0.0f);
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      acc += texelFetch(uInput, ivec3(x, y, pos.z), 0);
    }
  }

  acc /= float(max((y1 - y0) * (x1 - x0), 1));
  imageStore(uOutput, pos, acc);
}
