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
  ivec4 out_sizes; // (W=N, H=M, C=1, N=1) logical sizes
  int step_size;   // number of K texels: ceil(K / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Tiled matrix product with shared-memory staging.  A workgroup (8x8
 * threads, each producing a 4x4 output tile) covers a 32x32 output block.
 * Per K tile of 32 elements it cooperatively loads the 32x32 A and B
 * sub-blocks into shared memory (one fetch per texel, reused by every thread
 * in the workgroup), then accumulates from shared memory.  This removes the
 * per-step texture latency of the previous direct-fetch loop; the K tile
 * stride is 33 floats to avoid shared-memory bank conflicts.
 *
 * Operand layout (produced by the packing pass in the op):
 *  - uMat1 (M x K) is width-packed: texel (j, m) carries M1[m][4j .. 4j + 3]
 *    in its lanes, so four consecutive K elements cost one fetch.
 *  - uMat2 (K x N) is height-packed: texel (n, j) carries M2[4j .. 4j + 3][n],
 *    aligning the reduction axis with the same lanes.
 *
 * The tail texel of a non-multiple-of-4 K holds zeros inserted by the
 * packing pass, so it contributes nothing to the accumulators.  Reads of
 * rows/columns beyond the matrix edges also return zeros, and the
 * corresponding stores land outside the output image where they are
 * discarded.
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
    for (int c = 0; c < FOUR; ++c) {
      const ivec3 out_pos =
          ivec3(wg_base_n + out_n + c, wg_base_m + out_m + r, pos.z);
      if (out_pos.x < uBlock.out_sizes.x && out_pos.y < uBlock.out_sizes.y) {
        imageStore(uOutput, out_pos, vec4(results[r][c]));
      }
    }
  }
}