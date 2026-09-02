#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // Logical sizes of the input, prepadded to four axes {d0, d1, d2, d3}
  ivec4 in_logical;
  // Logical sizes of the output, prepadded to four axes {d0, d1, d2, d3}
  ivec4 out_logical;
  // out axis k reads in axis perm[k]
  ivec4 perm;
  int in_c_depth; // ceil(in C-slot / 4)
  int out_c_depth; // ceil(out C-slot / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Generic 4D permutation.  The output invocation reconstructs its logical
 * coordinates (including the channel lane), permutes them back to input
 * coordinates, and fetches one element per lane; lane order follows the
 * output channel axis, so a permuted channel axis reassigns lanes
 * individually.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_logical.w || pos.y >= uBlock.out_logical.z ||
      pos.z >= uBlock.out_logical.x * uBlock.out_c_depth) {
    return;
  }

  const int c4 = pos.z % uBlock.out_c_depth;
  const int n_out = pos.z / uBlock.out_c_depth;

  vec4 out_texel;
  for (int i = 0; i < 4; ++i) {
    const int c_out = c4 * 4 + i;
    if (c_out < uBlock.out_logical.y) {
      // Output logical coordinates {d0, d1, d2, d3} = {n, c, h, w}
      const int out_coord[4] =
          int[4](n_out, c_out, pos.y, pos.x);
      int in_coord[4];
      for (int k = 0; k < 4; ++k) {
        // Output axis k reads input axis perm[k]: walk output axes and
        // scatter each coordinate into its input slot.
        in_coord[uBlock.perm[k]] = out_coord[k];
      }
      out_texel[i] = texelFetch(
          uInput,
          ivec3(
              in_coord[3],
              in_coord[2],
              in_coord[0] * uBlock.in_c_depth + in_coord[1] / 4),
          0)[in_coord[1] % 4];
    }
  }

  imageStore(uOutput, pos, out_texel);
}
