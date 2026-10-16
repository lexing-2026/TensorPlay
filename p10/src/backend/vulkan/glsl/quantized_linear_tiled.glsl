#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Tiled Int8 GEMM with per-output-channel requantization:
//   out[m, n] = clamp(roundEven((input_scale * w_scale[n] * acc + bias[n])
//                             / out_scale) + out_zero_point,
//                     quant_min, quant_max)
// where acc = sum_k (x[m,k] - x_zp) * (w[n,k] - w_zp[n]) accumulates in
// int32.  The activation is width-packed (texel (k/4, m) holds four
// consecutive K values of row m) and the weight is height-packed (texel
// (n, k/4) holds four consecutive K values of column n), both as Int8.
// An 8x8 workgroup stages 32x32 sub-blocks of both operands into shared
// memory per 32-element K tile, then each thread accumulates a 4x4 output
// tile from shared memory (33-int row stride avoids bank conflicts).
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION isampler3D uMat1;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uMat2;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uParams;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  int out_m;
  int out_n;
  int k; // logical reduction width
  float input_scale;
  int input_zero_point;
  float out_inv_scale; // 1 / out_scale
  int out_zero_point;
  int quant_min;
  int quant_max;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define K_TEXELS (TILE_K / 4)
#define TILE_PAD 1

shared int sA[TILE_M][TILE_K + TILE_PAD];
shared int sB[TILE_N][TILE_K + TILE_PAD];

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int tid = int(gl_LocalInvocationIndex);
  const int wg_base_m = (pos.y / 8) * TILE_M;
  const int wg_base_n = (pos.x / 8) * TILE_N;
  const int local_m = (pos.y % 8) * 4;
  const int local_n = (pos.x % 8) * 4;

  // Per-channel zero points for this thread's four output columns.
  int w_zp[4];
  for (int c = 0; c < 4; ++c) {
    const int n = wg_base_n + local_n + c;
    w_zp[c] = n < uBlock.out_n
        ? int(texelFetch(uParams, ivec3(n >> 2, 1, 0), 0)[n & 3])
        : 0;
  }

  int acc[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      acc[i][j] = 0;
    }
  }

  const int k_texels = (uBlock.k + 3) / 4;
  for (int k_block = 0; k_block < k_texels; k_block += K_TEXELS) {
    // Stage the activation sub-block (width-packed Int8).
    for (int i = tid; i < TILE_M * K_TEXELS; i += 64) {
      const int row = i / K_TEXELS;
      const int jtex = k_block + (i % K_TEXELS);
      const int g_row = wg_base_m + row;
      const ivec4 v = (jtex < k_texels && g_row < uBlock.out_m)
          ? texelFetch(uMat1, ivec3(jtex, g_row, 0), 0)
          : ivec4(0);
      const int kk = (i % K_TEXELS) * 4;
      sA[row][kk + 0] = v.x;
      sA[row][kk + 1] = v.y;
      sA[row][kk + 2] = v.z;
      sA[row][kk + 3] = v.w;
    }

    // Stage the weight sub-block (height-packed Int8).
    for (int i = tid; i < TILE_N * K_TEXELS; i += 64) {
      const int row = i / K_TEXELS;
      const int jtex = k_block + (i % K_TEXELS);
      const int g_col = wg_base_n + row;
      const ivec4 v = (jtex < k_texels && g_col < uBlock.out_n)
          ? texelFetch(uMat2, ivec3(g_col, jtex, 0), 0)
          : ivec4(0);
      const int kk = (i % K_TEXELS) * 4;
      sB[row][kk + 0] = v.x;
      sB[row][kk + 1] = v.y;
      sB[row][kk + 2] = v.z;
      sB[row][kk + 3] = v.w;
    }

    barrier();

    // Padding bytes must not contribute after affine zero-point subtraction.
    const int valid_k = min(TILE_K, uBlock.k - 4 * k_block);
    for (int k = 0; k < valid_k; ++k) {
      for (int r = 0; r < 4; ++r) {
        const int a = sA[local_m + r][k] - uBlock.input_zero_point;
        for (int c = 0; c < 4; ++c) {
          acc[r][c] += a * (sB[local_n + c][k] - w_zp[c]);
        }
      }
    }

    barrier();
  }

  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      const int n = wg_base_n + local_n + c;
      const int m = wg_base_m + local_m + r;
      if (n < uBlock.out_n && m < uBlock.out_m) {
        const float w_scale = texelFetch(uParams, ivec3(n >> 2, 0, 0), 0)[n & 3];
        const float bias = texelFetch(uParams, ivec3(n >> 2, 2, 0), 0)[n & 3];
        const float out_f =
            uBlock.input_scale * w_scale * float(acc[r][c]) + bias;
        const int q = clamp(
            int(roundEven(out_f * uBlock.out_inv_scale)) +
                uBlock.out_zero_point,
            uBlock.quant_min,
            uBlock.quant_max);
        imageStore(uOutput, ivec3(n, m, 0), ivec4(q, 0, 0, 0));
      }
    }
  }
}
