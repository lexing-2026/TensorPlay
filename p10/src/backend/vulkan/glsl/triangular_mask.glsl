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
  // lower triangular when 1, upper triangular when 0
  int lower;
  int k; // diagonal offset
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Triangular mask.  The two-dimensional {H, W} payload addresses rows
 * through H and columns through W with one channel lane, so the predicate
 * compares the row and column indices directly.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.in_sizes.x || pos.y >= uBlock.in_sizes.y ||
      pos.z >= uBlock.in_sizes.w * ((uBlock.in_sizes.z + 3) / 4)) {
    return;
  }

  const int row = pos.y;
  const int col = pos.x;
  float keep;
  if (uBlock.lower != 0) {
    keep = (col <= row + uBlock.k) ? 1.0f : 0.0f;
  } else {
    keep = (col >= row + uBlock.k) ? 1.0f : 0.0f;
  }
  // Elements inside the triangle keep their values; everything else is
  // zeroed.  The fetch addresses the same {W, H, N*C/4} texel grid the
  // invocation covers.
  const vec4 in_texel = texelFetch(uInput, ivec3(pos.x, pos.y, pos.z), 0);
  imageStore(uOutput, pos, in_texel * vec4(keep));
}
