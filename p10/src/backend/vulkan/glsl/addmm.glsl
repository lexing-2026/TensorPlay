#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

// A named constant (instead of a literal) lets the SPIR-V compiler fully
// unroll every loop below.  TT is the number of output entries a thread owns
// along each axis: a workgroup of 8x8 threads covers 8*TT of each output axis,
// and the reduction tile is sized so that both staged operands stay inside the
// shared memory a baseline device has to offer.
#define TT ${THREAD_TILE}
#if TT > 4
#define TILE_K 16
#else
#define TILE_K 32
#endif

#define BLOCK (8 * TT)
#define K_TEXELS (TILE_K / 4)
#define TILE_PAD 1
#define THREADS 64

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
 * The product is staged exactly as the product with no addend stages it: one
 * workgroup of 8x8 threads covers a BLOCK x BLOCK output block, loads the two
 * BLOCK x TILE_K sub-blocks of the operands into shared memory per tile of
 * TILE_K reduction elements, and accumulates from there, so an operand texel
 * is read once per workgroup rather than once per tile.  The K tile stride is
 * one word longer than the tile to rotate rows across the shared memory banks,
 * and a thread's own rows and columns sit 8 elements apart so that the lanes of
 * a warp read distinct banks.
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
shared float sA[BLOCK][TILE_K + TILE_PAD];
shared float sB[BLOCK][TILE_K + TILE_PAD];

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Whole workgroups past the batch bound have no data at all.
  if (pos.z >= uBlock.out_sizes.w) {
    return;
  }

  // This shader is dispatched with an 8x8 workgroup.
  const int tid = int(gl_LocalInvocationIndex);
  const int wg_base_m = (pos.y / 8) * BLOCK;
  const int wg_base_n = (pos.x / 8) * BLOCK;
  const int lane_m = pos.y % 8;
  const int lane_n = pos.x % 8;

  float results[TT][TT];
  for (int i = 0; i < TT; ++i) {
    for (int j = 0; j < TT; ++j) {
      results[i][j] = 0.0f;
    }
  }

  for (int k_block = 0; k_block < uBlock.step_size; k_block += K_TEXELS) {
    // Stage the A sub-block (BLOCK rows x TILE_K cols) into shared memory.
    for (int i = tid; i < BLOCK * K_TEXELS; i += THREADS) {
      const int row = i / K_TEXELS;
      const int jtex = k_block + (i % K_TEXELS);
      const vec4 v = texelFetch(uMat1, ivec3(jtex, wg_base_m + row, pos.z), 0);
      const int kk = (i % K_TEXELS) * 4;
      sA[row][kk + 0] = v.x;
      sA[row][kk + 1] = v.y;
      sA[row][kk + 2] = v.z;
      sA[row][kk + 3] = v.w;
    }

    // Stage the B sub-block (BLOCK rows x TILE_K cols) into shared memory.
    for (int i = tid; i < BLOCK * K_TEXELS; i += THREADS) {
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

    for (int k = 0; k < TILE_K; ++k) {
      float arow[TT];
      float bcol[TT];
      for (int r = 0; r < TT; ++r) {
        arow[r] = sA[lane_m + 8 * r][k];
      }
      for (int c = 0; c < TT; ++c) {
        bcol[c] = sB[lane_n + 8 * c][k];
      }
      for (int r = 0; r < TT; ++r) {
        for (int c = 0; c < TT; ++c) {
          results[r][c] += arow[r] * bcol[c];
        }
      }
    }

    barrier();
  }

  // Store this thread's block; out-of-range stores are discarded.
  const bool fold_addend = uBlock.beta != 0.0f;
  for (int r = 0; r < TT; ++r) {
    const int m = wg_base_m + lane_m + 8 * r;
    if (m >= uBlock.out_sizes.y) {
      continue;
    }
    const int row_eff = (uBlock.bias_sizes.y == 1) ? 0 : m;
    for (int c = 0; c < TT; ++c) {
      const int n = wg_base_n + lane_n + 8 * c;
      if (n >= uBlock.out_sizes.x) {
        continue;
      }
      float addend = 0.0f;
      if (fold_addend) {
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
