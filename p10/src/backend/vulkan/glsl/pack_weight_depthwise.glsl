#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 weight_sizes; // (C, KH, KW, unused) logical sizes of the source
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Regroups a depthwise weight payload for the tiled depthwise kernel: the
 * source texture stores one channel per texel (lane x); the destination
 * texel (kx, ky, c4) carries the four channels 4*c4 .. 4*c4 + 3 of the same
 * kernel tap in its lanes.  Channels past the edge are written as zeros so
 * the packed tail contributes nothing.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int c_depth = (uBlock.weight_sizes.x + 3) / 4;
  if (pos.x >= uBlock.weight_sizes.z || pos.y >= uBlock.weight_sizes.y ||
      pos.z >= c_depth) {
    return;
  }

  vec4 out_t = vec4(0.0f);
  for (int lane = 0; lane < 4; ++lane) {
    const int c = pos.z * 4 + lane;
    if (c < uBlock.weight_sizes.x) {
      out_t[lane] = texelFetch(uInput, ivec3(pos.x, pos.y, c), 0).x;
    }
  }

  imageStore(uOutput, pos, out_t);
}
