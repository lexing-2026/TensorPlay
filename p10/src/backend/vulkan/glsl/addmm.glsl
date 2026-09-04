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
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 out_sizes;  // (W=N, H=M, C=1, N=1) logical sizes
  ivec4 bias_sizes; // (W, H, C, N) logical sizes of the addend
  int step_size;    // number of K texels: ceil(K / 4)
  float alpha;
  float beta;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Tiled matrix product with a fused affine epilogue:
 *   out = beta * bias + alpha * (M1 x M2)
 *
 * Operand layout (produced by the packing pass in the op):
 *  - uMat1 (M x K) is width-packed: texel (j, m) carries M1[m][4j .. 4j + 3]
 *    in its lanes, so four consecutive K elements cost one fetch.
 *  - uMat2 (K x N) is height-packed: texel (n, j) carries M2[4j .. 4j + 3][n],
 *    aligning the reduction axis with the same lanes.
 *
 * Per K step the invocation caches four adjacent rows of M1 and four adjacent
 * columns of M2, then performs sixteen four-lane dot products.  The tail
 * texel of a non-multiple-of-4 K holds zeros inserted by the packing pass,
 * so it contributes nothing to the accumulators.  The addend is broadcast
 * over singleton width/height dimensions and folded out entirely when beta
 * is zero.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (FOUR * pos.x >= uBlock.out_sizes.x ||
      FOUR * pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w) {
    return;
  }

  float results[FOUR][FOUR];
  for (int i = 0; i < FOUR; ++i) {
    for (int j = 0; j < FOUR; ++j) {
      results[i][j] = 0.0f;
    }
  }

  for (int j = 0; j < uBlock.step_size; ++j) {
    vec4 rows[FOUR];
    vec4 cols[FOUR];

    for (int k = 0; k < FOUR; ++k) {
      rows[k] = texelFetch(uMat1, ivec3(j, FOUR * pos.y + k, pos.z), 0);
    }
    for (int k = 0; k < FOUR; ++k) {
      cols[k] = texelFetch(uMat2, ivec3(FOUR * pos.x + k, j, pos.z), 0);
    }

    for (int r = 0; r < FOUR; ++r) {
      for (int c = 0; c < FOUR; ++c) {
        results[r][c] += dot(rows[r], cols[c]);
      }
    }
  }

  for (int r = 0; r < FOUR; ++r) {
    for (int c = 0; c < FOUR; ++c) {
      float bias_val = 0.0f;
      if (uBlock.beta != 0.0f) {
        const int row_eff =
            (uBlock.bias_sizes.y == 1) ? 0 : (FOUR * pos.y + r);
        const int col_eff =
            (uBlock.bias_sizes.x == 1) ? 0 : (FOUR * pos.x + c);
        bias_val = texelFetch(uBias, ivec3(col_eff, row_eff, 0), 0).x;
      }
      const ivec3 out_pos = ivec3(FOUR * pos.x + c, FOUR * pos.y + r, pos.z);
      imageStore(
          uOutput,
          out_pos,
          vec4(uBlock.beta * bias_val + uBlock.alpha * results[r][c]));
    }
  }
}
