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
  float scale_w; // in_w / out_w
  float scale_h; // in_h / out_h
  int c_depth; // ceil(C / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Nearest-neighbor 2D upsampling: each output position copies the input
 * element at floor(scale * position), clamped into range.  The batch axis
 * maps straight onto the z coordinate.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const int in_x = min(
      int(pos.x * uBlock.scale_w), uBlock.in_sizes.x - 1);
  const int in_y = min(
      int(pos.y * uBlock.scale_h), uBlock.in_sizes.y - 1);

  const vec4 v =
      texelFetch(uInput, ivec3(in_x, in_y, pos.z), 0);
  imageStore(uOutput, pos, v);
}
