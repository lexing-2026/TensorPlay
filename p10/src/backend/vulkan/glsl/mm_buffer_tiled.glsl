#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FLOAT_IMAGE_FORMAT}
// clang-format on

// A named constant (instead of a literal) lets the SPIR-V compiler fully
// unroll every loop below.
#define FOUR 4
#define TILE 32
#define TK 32
#define KVEC (TK / FOUR)
#define PAD 1

layout(std430) buffer;

/*
 * Tiled matrix product for operands that live in linear storage.  A matrix
 * with an axis longer than the device's image limit has no texture form at all,
 * which is where every large activation, weight and logit product ends up.
 *
 * A workgroup of 8x8 threads, each owning a 4x4 output tile, covers a 32x32
 * output block.  Per block of 32 reduction elements both sub-blocks are staged
 * cooperatively into shared memory, so an operand word is read from device
 * memory once however many output entries need it, and the accumulation loop
 * runs entirely out of shared memory.  Staging hands the consecutive lanes of
 * an item to whichever axis of the right operand runs along memory, so a
 * reduction-major weight and a column-major one are both streamed without
 * striding a warp across the whole matrix.
 *
 * Staging writes zero outside the operand extents, which is what keeps a
 * reduction length that is not a multiple of the tile size, and an output edge
 * that is not a multiple of four, from contributing anything.  The footprint is
 * 2 * 32 * 33 floats of shared memory, and the kernel needs no subgroup
 * operation, no image operand and no 16-bit storage.
 */
// clang-format off
$if OUTPUT_BUFFER == 1:
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
    float data[];
  }
  uOutput;
$else:
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
// clang-format on
// clang-format off
$if A_BUFFER == 1:
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly Mat1 {
    float data[];
  }
  uMat1;
$else:
  layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
// clang-format on
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mat2 {
  float data[];
}
uMat2;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_sizes;   // (W=N, H=M, C=1, N=1) logical product sizes
  ivec4 bias_sizes;  // (W=n, H=m, C=1, N=1) broadcast extents of the offset
  int k_elements;    // length of the reduction axis
  int k_blocks;      // reduction tiles to walk
  int a_stride_m;    // element step to the next left row
  int a_stride_k;    // element step to the next left column
  int b_stride_n;    // element step to the next right column
  int b_stride_k;    // element step to the next right row
  float alpha;
  float beta;
}
uBlock;

// clang-format off
$if HAS_BIAS == 1 and OUTPUT_BUFFER == 1:
  layout(set = 0, binding = 4) buffer PRECISION restrict readonly BiasBuffer {
    float data[];
  }
  uBias;
$elif HAS_BIAS == 1:
  layout(set = 0, binding = 4) uniform PRECISION sampler3D uBias;
// clang-format on

shared float sA[TILE][TK + PAD];
shared float sB[TILE][TK + PAD];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// clang-format off
$if A_BUFFER == 1:
  // The left operand contributes one element per lane; its rows are the
  // contiguous side of a dense activation, so an item walks reduction elements.
  void stage_a(const int tid, const int k0, const int wg_base_m, const int m) {
    for (int i = tid; i < TILE * TK; i += 64) {
      const int row = i / TK;
      const int kk = i - row * TK;
      const int k = k0 + kk;
      const int m_glob = wg_base_m + row;
      float v = 0.0;
      if (m_glob < m && k < uBlock.k_elements) {
        v = uMat1.data[m_glob * uBlock.a_stride_m + k * uBlock.a_stride_k];
      }
      sA[row][kk] = v;
    }
  }
$else:
  // A width-packed row carries four reduction elements per texel, and the
  // packing pass leaves zeros in the tail texel of a length that is not a
  // multiple of four.
  void stage_a(const int tid, const int k0, const int wg_base_m, const int m) {
    for (int i = tid; i < TILE * KVEC; i += 64) {
      const int row = i / KVEC;
      const int j = i - row * KVEC;
      const vec4 v =
          texelFetch(uMat1, ivec3(k0 / FOUR + j, wg_base_m + row, 0), 0);
      const int kk = j * FOUR;
      sA[row][kk + 0] = v.x;
      sA[row][kk + 1] = v.y;
      sA[row][kk + 2] = v.z;
      sA[row][kk + 3] = v.w;
    }
  }
// clang-format on

void stage_b(const int tid, const int k0, const int wg_base_n, const int n) {
  // Whichever axis of the right operand runs along memory gets the consecutive
  // lanes: a reduction-major row and a column-major row are both streamed by
  // adjacent threads, just along different tile edges.
  if (uBlock.b_stride_k == 1) {
    for (int i = tid; i < TILE * TK; i += 64) {
      const int col = i / TK;
      const int kk = i - col * TK;
      const int k = k0 + kk;
      const int n_glob = wg_base_n + col;
      float v = 0.0;
      if (n_glob < n && k < uBlock.k_elements) {
        v = uMat2.data[n_glob * uBlock.b_stride_n + k * uBlock.b_stride_k];
      }
      sB[col][kk] = v;
    }
  } else {
    for (int i = tid; i < TILE * TK; i += 64) {
      const int kk = i / TILE;
      const int col = i - kk * TILE;
      const int k = k0 + kk;
      const int n_glob = wg_base_n + col;
      float v = 0.0;
      if (n_glob < n && k < uBlock.k_elements) {
        v = uMat2.data[n_glob * uBlock.b_stride_n + k * uBlock.b_stride_k];
      }
      sB[col][kk] = v;
    }
  }
}

// clang-format off
$if HAS_BIAS == 1 and OUTPUT_BUFFER == 1:
  float bias_at(const int row, const int col) {
    const int col_eff = (uBlock.bias_sizes.x == 1) ? 0 : col;
    const int row_eff = (uBlock.bias_sizes.y == 1) ? 0 : row;
    return uBias.data[row_eff * uBlock.bias_sizes.x + col_eff];
  }
$elif HAS_BIAS == 1:
  float bias_at(const int row, const int col) {
    const int col_eff = (uBlock.bias_sizes.x == 1) ? 0 : col;
    const int row_eff = (uBlock.bias_sizes.y == 1) ? 0 : row;
    return texelFetch(uBias, ivec3(col_eff, row_eff, 0), 0).x;
  }
$else:
  float bias_at(const int row, const int col) {
    return 0.0;
  }
// clang-format on

void store_at(const int row, const int col, const float value) {
  // clang-format off
  $if OUTPUT_BUFFER == 1:
    uOutput.data[row * uBlock.out_sizes.x + col] = value;
  $else:
    imageStore(uOutput, ivec3(col, row, 0), vec4(value));
  // clang-format on
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int tid = int(gl_LocalInvocationIndex);

  const int n = uBlock.out_sizes.x;
  const int m = uBlock.out_sizes.y;

  const int wg_base_m = (pos.y / 8) * TILE;
  const int wg_base_n = (pos.x / 8) * TILE;
  const int local_m = (pos.y % 8) * FOUR;
  const int local_n = (pos.x % 8) * FOUR;

  float results[FOUR][FOUR];
  for (int r = 0; r < FOUR; ++r) {
    for (int c = 0; c < FOUR; ++c) {
      results[r][c] = 0.0;
    }
  }

  for (int kb = 0; kb < uBlock.k_blocks; ++kb) {
    const int k0 = kb * TK;
    stage_a(tid, k0, wg_base_m, m);
    stage_b(tid, k0, wg_base_n, n);

    barrier();

    for (int k = 0; k < TK; ++k) {
      for (int r = 0; r < FOUR; ++r) {
        const float arow = sA[local_m + r][k];
        for (int c = 0; c < FOUR; ++c) {
          results[r][c] += arow * sB[local_n + c][k];
        }
      }
    }

    barrier();
  }

  for (int r = 0; r < FOUR; ++r) {
    const int out_m = wg_base_m + local_m + r;
    if (out_m >= m) {
      continue;
    }
    for (int c = 0; c < FOUR; ++c) {
      const int out_n = wg_base_n + local_n + c;
      if (out_n >= n) {
        continue;
      }
      const float acc = uBlock.alpha * results[r][c] +
          uBlock.beta * bias_at(out_m, out_n);
      store_at(out_m, out_n, acc);
    }
  }
}
