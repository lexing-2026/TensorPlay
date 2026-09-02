#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  ivec4 in_sizes;
  // concatenated axis, counted from the innermost (0: W, 1: H, 2: C, 3: N)
  int axis;
  // element offset of this input along the concatenated axis
  int offset;
  int in_c_depth; // ceil(in C / 4)
  int out_c_depth; // ceil(out C / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Concatenation along any axis, addressed in innermost-first axis order
 * (0: width, 1: height, 2: channel, 3: batch).  Inputs and output share the
 * rank, so the invocation maps its output coordinates to input coordinates
 * by shifting only the concatenated axis back through the running offset;
 * every other coordinate carries over.  Elements are fetched per channel
 * lane, so a channel-axis concatenation reroutes lanes and z positions
 * correctly, and batch-axis concatenation of inputs whose outer slots sit
 * inside another input's texel (3d inputs) still reaches the right planes.
 * A lane whose shifted coordinate falls outside this input belongs to a
 * neighboring input: its current output value is loaded and written back
 * untouched instead of being zeroed.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  const int n_out = pos.z / uBlock.out_c_depth;
  const int c4 = pos.z % uBlock.out_c_depth;

  vec4 out_texel = imageLoad(uOutput, pos);
  for (int i = 0; i < 4; ++i) {
    // Output coordinates in innermost-first order.
    int o[4];
    o[0] = pos.x;
    o[1] = pos.y;
    o[2] = c4 * 4 + i;
    o[3] = n_out;

    // Input coordinates: only the concatenated axis moves.
    int s[4];
    for (int q = 0; q < 4; ++q) {
      s[q] = (q == uBlock.axis) ? (o[q] - uBlock.offset) : o[q];
    }

    if (s[0] >= 0 && s[0] < uBlock.in_sizes.x && s[1] >= 0 &&
        s[1] < uBlock.in_sizes.y && s[2] >= 0 && s[2] < uBlock.in_sizes.z &&
        s[3] >= 0 && s[3] < uBlock.in_sizes.w) {
      out_texel[i] = texelFetch(
          uInput,
          ivec3(s[0], s[1], s[3] * uBlock.in_c_depth + s[2] / 4),
          0)[s[2] % 4];
    }
  }

  imageStore(uOutput, pos, out_texel);
}
