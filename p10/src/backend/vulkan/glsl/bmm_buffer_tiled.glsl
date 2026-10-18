#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
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
 * Batched tiled matrix product for a result with no texture form.  A batched
 * product whose row, column or reduction axis runs past the device's image
 * limit cannot be expressed as one image per batch, and long-context attention
 * is exactly where every one of those axes grows.
 *
 * One workgroup of 8x8 threads, each owning a 4x4 output tile, covers a 32x32
 * output block of one batch; the z axis of the dispatch walks the batches, so
 * the whole product runs in a single submission.  An operand stays in whichever
 * form it already has: a linear payload is read through its own strides
 * (a_stride_b, b_stride_b, out_stride_b give the step to the next matrix),
 * while an operand that kept an image form is read from the per-batch planes
 * the pack passes produce, one texel z slot per matrix.  Per reduction tile
 * both sub-blocks are staged cooperatively into shared memory, so an operand
 * word is read from device memory once however many output entries need it,
 * and the accumulation loop runs entirely out of shared memory.
 *
 * The result is always written linearly, in (batch, row, column) order.
 * Staging writes zero outside the operand extents, which is what keeps a
 * reduction length that is not a multiple of the tile size, and an output edge
 * that is not a multiple of four, from contributing anything.  The footprint is
 * 2 * 32 * 33 floats of shared memory, and the kernel needs no subgroup
 * operation and no 16-bit storage.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;
// clang-format off
$if A_PACKED == 1:
  layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
$else:
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly Mat1 {
    float data[];
  }
  uMat1;
// clang-format on
// clang-format off
$if B_PACKED == 1:
  layout(set = 0, binding = 2) uniform PRECISION sampler3D uMat2;
$else:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mat2 {
    float data[];
  }
  uMat2;
// clang-format on
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_sizes;   // (W=N, H=M, C=B, N=1) logical product sizes
  int k_elements;    // length of the reduction axis
  int k_blocks;      // reduction tiles to walk
  int a_stride_m;    // element step to the next left row
  int a_stride_k;    // element step to the next left column
  int a_stride_b;    // element step to the next left matrix
  int b_stride_n;    // element step to the next right column
  int b_stride_k;    // element step to the next right row
  int b_stride_b;    // element step to the next right matrix
  int out_stride_b;  // element step to the next result matrix
}
uBlock;

shared float sA[TILE][TK + PAD];
shared float sB[TILE][TK + PAD];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// clang-format off
$if A_PACKED == 1:
  // One texture plane per matrix: a width-packed row carries four reduction
  // elements per texel, and the packing pass leaves zeros in the tail texel of
  // a length that is not a multiple of four.
  void stage_a(
      const int tid, const int k0, const int wg_base_m, const int batch) {
    for (int i = tid; i < TILE * KVEC; i += 64) {
      const int row = i / KVEC;
      const int j = i - row * KVEC;
      const vec4 v =
          texelFetch(uMat1, ivec3(k0 / FOUR + j, wg_base_m + row, batch), 0);
      const int kk = j * FOUR;
      sA[row][kk + 0] = v.x;
      sA[row][kk + 1] = v.y;
      sA[row][kk + 2] = v.z;
      sA[row][kk + 3] = v.w;
    }
  }
$else:
  // The left operand contributes one element per lane; its rows are the
  // contiguous side of a dense activation, so an item walks reduction elements.
  void stage_a(
      const int tid,
      const int k0,
      const int wg_base_m,
      const int m,
      const int batch) {
    const int base = batch * uBlock.a_stride_b;
    for (int i = tid; i < TILE * TK; i += 64) {
      const int row = i / TK;
      const int kk = i - row * TK;
      const int k = k0 + kk;
      const int m_glob = wg_base_m + row;
      float v = 0.0;
      if (m_glob < m && k < uBlock.k_elements) {
        v = uMat1.data[base + m_glob * uBlock.a_stride_m + k * uBlock.a_stride_k];
      }
      sA[row][kk] = v;
    }
  }
// clang-format on

// clang-format off
$if B_PACKED == 1:
  // A height-packed texel (n, j) of the matrix's plane carries its rows
  // 4j .. 4j + 3 at column n, so four consecutive reduction elements of one
  // output column arrive in one fetch.
  void stage_b(
      const int tid, const int k0, const int wg_base_n, const int batch) {
    for (int i = tid; i < TILE * KVEC; i += 64) {
      const int col = i / KVEC;
      const int j = i - col * KVEC;
      const vec4 v =
          texelFetch(uMat2, ivec3(wg_base_n + col, k0 / FOUR + j, batch), 0);
      const int kk = j * FOUR;
      sB[col][kk + 0] = v.x;
      sB[col][kk + 1] = v.y;
      sB[col][kk + 2] = v.z;
      sB[col][kk + 3] = v.w;
    }
  }
$else:
  // Whichever axis of the right operand runs along memory gets the consecutive
  // lanes: a reduction-major matrix and a column-major one are both streamed by
  // adjacent threads, just along different tile edges.
  void stage_b(
      const int tid,
      const int k0,
      const int wg_base_n,
      const int n,
      const int batch) {
    const int base = batch * uBlock.b_stride_b;
    if (uBlock.b_stride_k == 1) {
      for (int i = tid; i < TILE * TK; i += 64) {
        const int col = i / TK;
        const int kk = i - col * TK;
        const int k = k0 + kk;
        const int n_glob = wg_base_n + col;
        float v = 0.0;
        if (n_glob < n && k < uBlock.k_elements) {
          v = uMat2.data[base + n_glob * uBlock.b_stride_n + k * uBlock.b_stride_k];
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
          v = uMat2.data[base + n_glob * uBlock.b_stride_n + k * uBlock.b_stride_k];
        }
        sB[col][kk] = v;
      }
    }
  }
// clang-format on

void store_at(const int batch, const int row, const int col, const float value) {
  uOutput.data[batch * uBlock.out_stride_b + row * uBlock.out_sizes.x + col] =
      value;
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int tid = int(gl_LocalInvocationIndex);

  const int batch = pos.z;
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
    // clang-format off
    $if A_PACKED == 1:
      stage_a(tid, k0, wg_base_m, batch);
    $else:
      stage_a(tid, k0, wg_base_m, m, batch);
    // clang-format on
    // clang-format off
    $if B_PACKED == 1:
      stage_b(tid, k0, wg_base_n, batch);
    $else:
      stage_b(tid, k0, wg_base_n, n, batch);
    // clang-format on

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
      store_at(batch, out_m, out_n, results[r][c]);
    }
  }
}
