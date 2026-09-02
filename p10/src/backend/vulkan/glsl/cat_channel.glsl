#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  ivec4 in_sizes;
  int axis; // 0: W, 1: H, 2: C, 3: N (the concatenated axis)
  int offset; // element offset of this input along the concat axis
  int in_c_depth; // ceil(in C / 4)
  int out_c_depth; // ceil(out C / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Concatenation along the channel axis.  Every other axis has matching
 * extents, so the invocation copies the texel it lands on, remapping the
 * channel coordinate through the offset and masking the lanes that fall
 * outside this input.  Concatenation along width, height, and batch uses
 * plain texture copies with offsets instead of this shader.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  const int n_out = pos.z / uBlock.out_c_depth;
  const int c4 = pos.z % uBlock.out_c_depth;

  vec4 out_texel = vec4(0.0f);
  for (int i = 0; i < 4; ++i) {
    const int out_c = c4 * 4 + i;
    const int in_c = out_c - uBlock.offset;
    if (in_c >= 0 && in_c < uBlock.in_sizes.z &&
        n_out < uBlock.in_sizes.w) {
      out_texel[i] = texelFetch(
          uInput,
          ivec3(pos.x, pos.y, n_out * uBlock.in_c_depth + in_c / 4),
          0)[in_c % 4];
    }
  }

  imageStore(uOutput, pos, out_texel);
}
