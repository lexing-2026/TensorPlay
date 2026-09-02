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
  // One flag per input axis, nonzero when that axis is flipped
  ivec4 flip_axes;
  int c_depth; // ceil(C / 4)
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Reverses element order along each flagged axis.  Width and height flip in
 * texel coordinates; flipping the batch axis steps by c_depth texels along
 * z.  Flipping the channel axis reverses the channel index itself, so every
 * output lane resolves its source texel and lane individually and channel
 * padding stays out of the mapping.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  ivec3 src = pos;
  if (uBlock.flip_axes.x != 0) {
    src.x = uBlock.in_sizes.x - 1 - pos.x;
  }
  if (uBlock.flip_axes.y != 0) {
    src.y = uBlock.in_sizes.y - 1 - pos.y;
  }

  const int n = pos.z / uBlock.c_depth;
  const int c4 = pos.z % uBlock.c_depth;

  int src_n = n;
  if (uBlock.flip_axes.w != 0) {
    src_n = uBlock.in_sizes.w - 1 - n;
  }

  vec4 out_texel;
  if (uBlock.flip_axes.z != 0) {
    const int channels = uBlock.in_sizes.z;
    for (int i = 0; i < 4; ++i) {
      const int c = c4 * 4 + i;
      if (c < channels) {
        const int src_c = channels - 1 - c;
        out_texel[i] = texelFetch(
            uInput,
            ivec3(src.x, src.y, src_n * uBlock.c_depth + src_c / 4),
            0)[src_c % 4];
      }
    }
  } else {
    out_texel = texelFetch(
        uInput, ivec3(src.x, src.y, src_n * uBlock.c_depth + c4), 0);
  }

  imageStore(uOutput, pos, out_texel);
}
