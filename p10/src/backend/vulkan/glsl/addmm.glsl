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
 * A workgroup of 8x8 threads, each producing a 4x4 output tile, covers a 32x32
 * output block.  Per K tile of 32 elements it cooperatively stages the 32x32
 * A and B sub-blocks into shared memory, one fetch per texel reused by every
 * thread of the workgroup, and then accumulates from shared memory.  Walking
 * the operands from their images at every K step instead costs the workgroup
 * eight texel fetches per four lanes of the reduction, so an operand element
 * is read once per output row that needs it rather than once per tile, and the
 * reduction stalls on each of those fetches.  The K tile stride is 33 floats
 * to avoid shared-memory bank conflicts.
 *
 * Operand layout (produced by the packing pass in the op):
 *  - uMat1 (M x K) is width-packed: texel (j, m) carries M1[m][4j .. 4j + 3]
 *    in its lanes, so four consecutive K elements cost one fetch.
 *  - uMat2 (K x N) is height-packed: texel (n, j) carries M2[4j .. 4j + 3][n],
 *    aligning the reduction axis with the same lanes.
 *
 * The tail texel of a non-multiple-of-4 K holds zeros inserted by the packing
 * pass, so it contributes nothing to the accumulators.  Reads of rows/columns
 * beyond the matrix edges also return zeros, and the corresponding stores land
 * outside the output image where they are discarded.
 *
 * The addend is a plane of the output, so it is read once per output element
 * after the reduction rather than inside it, is broadcast over singleton
 * width/height dimensions, and is folded out entirely when beta is zero.
 */
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define K_TEXELS (TILE_K / 4)
#define TILE_PAD 1

shared float sA[TILE_M][TILE_K + TILE_PAD];
shared float sB[TILE_N][TILE_K + TILE_PAD];

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Whole workgroups past the batch bound have no data at all.
  if (pos.z >= uBlock.out_sizes.w) {
    return;
  }

  // This shader is dispatched with an 8x8 workgroup.
  const int tid = int(gl_LocalInvocationIndex);
  const int wg_base_m = (pos.y / 8) * 8 * FOUR;
  const int wg_base_n = (pos.x / 8) * 8 * FOUR;

  float results[FOUR][FOUR];
  for (int i = 0; i < FOUR; ++i) {
    for (int j = 0; j < FOUR; ++j) {
      results[i][j] = 0.0f;
    }
  }

  for (int k_block = 0; k_block < uBlock.step_size; k_block += K_TEXELS) {
    // Stage the A sub-block (TILE_M rows x TILE_K cols) into shared memory.
    for (int i = tid; i < TILE_M * K_TEXELS; i += 64) {
      const int row = i / K_TEXELS;
      const int jtex = k_block + (i % K_TEXELS);
      const vec4 v = texelFetch(uMat1, ivec3(jtex, wg_base_m + row, pos.z), 0);
      const int kk = (i % K_TEXELS) * 4;
      sA[row][kk + 0] = v.x;
      sA[row][kk + 1] = v.y;
      sA[row][kk + 2] = v.z;
      sA[row][kk + 3] = v.w;
    }

    // Stage the B sub-block (TILE_N rows x TILE_K cols) into shared memory.
    for (int i = tid; i < TILE_N * K_TEXELS; i += 64) {
      const int row = i / K_TEXELS;
      const int jtex = k_block + (i % K_TEXELS);
      const vec4 v = texelFetch(uMat2, ivec3(wg_base_n + row, jtex, pos.z), 0);
      const int kk = (i % K_TEXELS) * 4;
      sB[row][kk + 0] = v.x;
      sB[row][kk + 1] = v.y;
      sB[row][kk + 2] = v.z;
      sB[row][kk + 3] = v.w;
    }

    barrier();

    const int local_m = (pos.y % 8) * 4;
    const int local_n = (pos.x % 8) * 4;
    for (int k = 0; k < TILE_K; ++k) {
      for (int r = 0; r < FOUR; ++r) {
        const float arow = sA[local_m + r][k];
        for (int c = 0; c < FOUR; ++c) {
          results[r][c] += arow * sB[local_n + c][k];
        }
      }
    }

    barrier();
  }

  // Store the 4x4 output tile; out-of-range stores are discarded.
  const int out_m = (pos.y % 8) * 4;
  const int out_n = (pos.x % 8) * 4;
  for (int r = 0; r < FOUR; ++r) {
    const int m = wg_base_m + out_m + r;
    if (m >= uBlock.out_sizes.y) {
      continue;
    }
    const int row_eff = (uBlock.bias_sizes.y == 1) ? 0 : m;
    for (int c = 0; c < FOUR; ++c) {
      const int n = wg_base_n + out_n + c;
      if (n >= uBlock.out_sizes.x) {
        continue;
      }
      float addend = 0.0f;
      if (uBlock.beta != 0.0f) {
        const int col_eff = (uBlock.bias_sizes.x == 1) ? 0 : n;
        addend = texelFetch(uBias, ivec3(col_eff, row_eff, 0), 0).x;
      }
      imageStore(
          uOutput,
          ivec3(n, m, pos.z),
          vec4(uBlock.beta * addend + uBlock.alpha * results[r][c]));
    }
  }
}
