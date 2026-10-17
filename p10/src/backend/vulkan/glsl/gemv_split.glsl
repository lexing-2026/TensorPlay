#version 450 core
#define FOUR 4
layout(std430) buffer;

// The partial arena is always Float32 regardless of the operand precision,
// so its image format is fixed; the packed operands stream through plain
// samplers, whose binary contract carries no format and binds either
// rgba32f or rgba16f views.
layout(set = 0, binding = 0, rgba32f) uniform highp restrict writeonly image3D uPartials;
layout(set = 0, binding = 1) uniform highp sampler3D uMat1;
layout(set = 0, binding = 2) uniform highp sampler3D uMat2;
layout(set = 0, binding = 3) uniform highp restrict Block {
  ivec4 out_sizes; // (W=N, H=M, C=1, N=1) logical product sizes
  int k_texels;    // number of K texels: ceil(K / 4)
  int chunk;       // K texels covered by one split
  int splits;
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * First pass of the two-pass decode matrix-vector product.  A long serial
 * reduction is latency-limited whenever the output width alone cannot feed
 * the device, so the K axis is divided into a fixed number of contiguous
 * chunks: one invocation owns one output texel (four consecutive columns)
 * and one chunk, reducing the lane-aligned packed operands independently.
 * Both packed operands fuse four consecutive K elements into their texel
 * lanes under matching index maps, so a single float4 dot folds one texel
 * pair and the four accumulator chains stay independent.
 *
 * Each split writes a disjoint row of the partial arena, so the fold pass
 * needs nothing beyond job ordering to observe every partial.  Columns past
 * the output width read zero texels and store into the aligned tail lanes,
 * which the fold pass masked-stores never read back.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int n_texel = pos.x;  // covers four consecutive output columns
  const int split = pos.y;
  const int row = pos.z;

  if (FOUR * n_texel >= uBlock.out_sizes.x || split >= uBlock.splits ||
      row >= uBlock.out_sizes.y) {
    return;
  }

  const int j_begin = split * uBlock.chunk;
  const int j_end = min(j_begin + uBlock.chunk, uBlock.k_texels);

  // The packed rhs plane is one texel per output column, so the tail texel
  // of a width that is not a multiple of four reads past the image: those
  // columns never contribute, and guarded fetches keep the undefined-read
  // case out of the loop.
  const int n_base = FOUR * n_texel;
  const int n = uBlock.out_sizes.x;

  vec4 acc = vec4(0.0);
  for (int j = j_begin; j < j_end; ++j) {
    const vec4 a = texelFetch(uMat1, ivec3(j, row, 0), 0);
    acc.x += dot(a, texelFetch(uMat2, ivec3(n_base + 0, j, 0), 0));
    if (n_base + 1 < n) {
      acc.y += dot(a, texelFetch(uMat2, ivec3(n_base + 1, j, 0), 0));
    }
    if (n_base + 2 < n) {
      acc.z += dot(a, texelFetch(uMat2, ivec3(n_base + 2, j, 0), 0));
    }
    if (n_base + 3 < n) {
      acc.w += dot(a, texelFetch(uMat2, ivec3(n_base + 3, j, 0), 0));
    }
  }

  // The partial arena is width-packed: texel (x, s * M + row) lane l holds
  // the partial of output column FOUR * x + l.
  imageStore(
      uPartials,
      ivec3(n_texel, split * uBlock.out_sizes.y + row, 0),
      acc);
}
