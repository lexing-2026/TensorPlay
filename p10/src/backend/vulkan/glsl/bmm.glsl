#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

// A named constant (instead of a literal) lets the SPIR-V compiler fully
// unroll every loop below.
#define FOUR 4

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uMat2;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_sizes; // (W=N, H=M, C=B, N=1) logical sizes of the batched result
  int step_size;   // number of K texels: ceil(K / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Batched tiled matrix product: one invocation computes a 4x4 output block
 * for up to four batches at once.
 *
 * The result is stored in the channel-packed batched layout, where one texel
 * carries four batches in its lanes: texel (n, m, g) component l holds
 * out[b = 4g + l][m][n].  Each invocation therefore owns one texel z slot and
 * produces the full vec4 of batch values per output element, so every store
 * lands complete.
 *
 * Operand layout (produced by the pack passes in the op): both operands keep
 * one texture plane per batch (texel z equals the batch index).
 *  - uMat1 plane b is the B-th matrix width-packed: texel (j, m) carries
 *    M1[b][m][4j .. 4j + 3] in its lanes.
 *  - uMat2 plane b is the B-th matrix height-packed: texel (n, j) carries
 *    M2[b][4j .. 4j + 3][n].
 *
 * Per K step the invocation caches four adjacent rows of M1[b] and four
 * adjacent columns of M2[b], then performs sixteen four-lane dot products,
 * for each active batch in turn.  Batch groups past the batch count are
 * skipped entirely (their lanes would land on padding positions the dense
 * readback ignores).  The tail texel of a non-multiple-of-4 K holds zeros
 * inserted by the packing pass; reads of rows/columns beyond the matrix
 * edges also return zeros, and the corresponding stores land outside the
 * output image where they are discarded.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (FOUR * pos.x >= uBlock.out_sizes.x ||
      FOUR * pos.y >= uBlock.out_sizes.y ||
      FOUR * pos.z >= uBlock.out_sizes.z) {
    return;
  }

  float results[FOUR][FOUR][FOUR];
  for (int b = 0; b < FOUR; ++b) {
    for (int r = 0; r < FOUR; ++r) {
      for (int c = 0; c < FOUR; ++c) {
        results[b][r][c] = 0.0f;
      }
    }
  }

  for (int b_i = 0; b_i < FOUR; ++b_i) {
    if (FOUR * pos.z + b_i >= uBlock.out_sizes.z) {
      break;
    }
    const int b = FOUR * pos.z + b_i;

    for (int j = 0; j < uBlock.step_size; ++j) {
      vec4 rows[FOUR];
      vec4 cols[FOUR];

      for (int k = 0; k < FOUR; ++k) {
        rows[k] = texelFetch(uMat1, ivec3(j, FOUR * pos.y + k, b), 0);
      }
      for (int k = 0; k < FOUR; ++k) {
        cols[k] = texelFetch(uMat2, ivec3(FOUR * pos.x + k, j, b), 0);
      }

      for (int r = 0; r < FOUR; ++r) {
        for (int c = 0; c < FOUR; ++c) {
          results[b_i][r][c] += dot(rows[r], cols[c]);
        }
      }
    }
  }

  for (int r = 0; r < FOUR; ++r) {
    for (int c = 0; c < FOUR; ++c) {
      const ivec3 out_pos = ivec3(FOUR * pos.x + c, FOUR * pos.y + r, pos.z);
      imageStore(
          uOutput,
          out_pos,
          vec4(
              results[0][r][c],
              results[1][r][c],
              results[2][r][c],
              results[3][r][c]));
    }
  }
}
