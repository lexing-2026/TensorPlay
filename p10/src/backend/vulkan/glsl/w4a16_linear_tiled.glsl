#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Tiled W4A16 GEMM: FP16 activations against weight-only 4-bit codes:
//   out[m, n] = w_scale[n] * sum_k act[m, k] * w4[n, k] + bias[n]
// The activation is width-packed FP16 (texel (k/4, m) holds four
// consecutive half K values) and the weight is height-packed Int8 (texel
// (n, k/8) holds eight signed 4-bit codes, two per byte, little nibble
// first).  An 8x8 workgroup stages 32x32 sub-blocks of both operands into
// shared memory per 32-element K tile, then each thread accumulates a 4x4
// output tile from shared memory (33-element row stride avoids bank
// conflicts).  Accumulation runs in FP32; the output is stored as FP16.
layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uMat2;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uParams;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  int out_m;
  int out_n;
  int k; // logical reduction width
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define ACT_TEXELS (TILE_K / 4)
#define W_TEXELS (TILE_K / 8)
#define TILE_PAD 1

shared float sA[TILE_M][TILE_K + TILE_PAD];
shared float sB[TILE_N][TILE_K + TILE_PAD];

int nib0(int byte) {
  // Convert through uint so the nibble extraction is well-defined for the
  // negative bytes produced by int8 texture loads.
  const uint u = uint(byte);
  const int v = int(u & 0xFu);
  return (v >= 8) ? (v - 16) : v;
}

int nib1(int byte) {
  const uint u = uint(byte);
  const int v = int((u >> 4u) & 0xFu);
  return (v >= 8) ? (v - 16) : v;
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int tid = int(gl_LocalInvocationIndex);
  const int wg_base_m = (pos.y / 8) * TILE_M;
  const int wg_base_n = (pos.x / 8) * TILE_N;
  const int local_m = (pos.y % 8) * 4;
  const int local_n = (pos.x % 8) * 4;

  float acc[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      acc[i][j] = 0.0;
    }
  }

  const int k_texels_act = (uBlock.k + 3) / 4;
  const int k_texels_w = (uBlock.k + 7) / 8;

  for (int k_block = 0; k_block < uBlock.k; k_block += TILE_K) {
    // Stage the activation sub-block (width-packed FP16).
    for (int i = tid; i < TILE_M * ACT_TEXELS; i += 64) {
      const int row = i / ACT_TEXELS;
      const int jtex = (k_block / 4) + (i % ACT_TEXELS);
      const int g_row = wg_base_m + row;
      vec4 v = (jtex < k_texels_act && g_row < uBlock.out_m)
          ? texelFetch(uMat1, ivec3(jtex, g_row, 0), 0)
          : vec4(0.0);
      // Zero padded lanes so they stay neutral under the FP32 accumulation.
      for (int lane = 0; lane < 4; ++lane) {
        if (4 * jtex + lane >= uBlock.k || g_row >= uBlock.out_m) {
          v[lane] = 0.0;
        }
      }
      const int kk = (i % ACT_TEXELS) * 4;
      sA[row][kk + 0] = v.x;
      sA[row][kk + 1] = v.y;
      sA[row][kk + 2] = v.z;
      sA[row][kk + 3] = v.w;
    }

    // Stage the weight sub-block (packed 4-bit codes).
    for (int i = tid; i < TILE_N * W_TEXELS; i += 64) {
      const int row = i / W_TEXELS;
      const int jtex = (k_block / 8) + (i % W_TEXELS);
      const int g_col = wg_base_n + row;
      const ivec4 b = (jtex < k_texels_w && g_col < uBlock.out_n)
          ? texelFetch(uMat2, ivec3(g_col, jtex, 0), 0)
          : ivec4(0);
      const int kk = (i % W_TEXELS) * 8;
      // Decode straight to float so the inner K loop only does float
      // multiplies; the int->float conversion happens once per staged
      // element instead of once per (thread, K) pair.
      sB[row][kk + 0] = float(nib0(b.x));
      sB[row][kk + 1] = float(nib1(b.x));
      sB[row][kk + 2] = float(nib0(b.y));
      sB[row][kk + 3] = float(nib1(b.y));
      sB[row][kk + 4] = float(nib0(b.z));
      sB[row][kk + 5] = float(nib1(b.z));
      sB[row][kk + 6] = float(nib0(b.w));
      sB[row][kk + 7] = float(nib1(b.w));
    }

    barrier();

    for (int k = 0; k < TILE_K; ++k) {
      for (int r = 0; r < 4; ++r) {
        const float a = sA[local_m + r][k];
        for (int c = 0; c < 4; ++c) {
          acc[r][c] += a * sB[local_n + c][k];
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
        const float out_f = w_scale * acc[r][c] + bias;
        imageStore(uOutput, ivec3(n, m, 0), vec4(out_f, 0, 0, 0));
      }
    }
  }
}