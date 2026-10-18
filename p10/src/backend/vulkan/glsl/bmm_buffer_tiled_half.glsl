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
 * Batched tiled matrix product over half-precision payloads written linearly.
 * A narrowed device runtime keeps its activations and weights half, and a
 * batched product with an axis past the image limit has no texture form for
 * them, so a payload word carries two lanes, low lane first.
 *
 * One workgroup of 8x8 threads, each owning a 4x4 output tile, covers a 32x32
 * output block of one matrix and the dispatch's z axis walks the matrices.
 * Both sub-blocks are staged cooperatively into shared memory, and staging is
 * the only place a lane is unpacked: the tiles hold floats, so the accumulation
 * loop never narrows between terms and never repeats the unpack.  A texel of a
 * packed operand already yields four floats, so only a linear read pairs lanes.
 *
 * The result is written as whole words, and a thread owns four consecutive
 * columns of each of its four rows, so pairing needs the result's row length to
 * be even: the host rejects an odd width rather than letting two matrices claim
 * one word.  Staging writes zero outside the operand extents, which is what
 * keeps a reduction length that is not a multiple of the tile size, and an
 * output edge, from contributing anything.  The footprint stays 2 * 32 * 33
 * floats, and the kernel needs no subgroup operation and no 16-bit storage.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  uint data[];
}
uOutput;
// clang-format off
$if A_PACKED == 1:
  layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
$else:
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly Mat1 {
    uint data[];
  }
  uMat1;
// clang-format on
// clang-format off
$if B_PACKED == 1:
  layout(set = 0, binding = 2) uniform PRECISION sampler3D uMat2;
$else:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mat2 {
    uint data[];
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

float lane_of(const in uint word, const in uint lane) {
  const vec2 pair = unpackHalf2x16(word);
  return lane == 0u ? pair.x : pair.y;
}

// clang-format off
$if A_PACKED == 0:
  float a_lane(const in int index) {
    return lane_of(uMat1.data[index >> 1], uint(index) & 1u);
  }
// clang-format on
// clang-format off
$if B_PACKED == 0:
  float b_lane(const in int index) {
    return lane_of(uMat2.data[index >> 1], uint(index) & 1u);
  }
// clang-format on

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
        v = a_lane(
            base + m_glob * uBlock.a_stride_m + k * uBlock.a_stride_k);
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
          v = b_lane(
              base + n_glob * uBlock.b_stride_n + k * uBlock.b_stride_k);
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
          v = b_lane(
              base + n_glob * uBlock.b_stride_n + k * uBlock.b_stride_k);
        }
        sB[col][kk] = v;
      }
    }
  }
// clang-format on

// One store covers the pair of neighbouring lanes inside a word.
void store_pair(
    const int batch, const int row, const int col, const vec2 value) {
  const int index =
      batch * uBlock.out_stride_b + row * uBlock.out_sizes.x + col;
  uOutput.data[index >> 1] = packHalf2x16(value);
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
    for (int c = 0; c < FOUR; c += 2) {
      const int out_n = wg_base_n + local_n + c;
      // An even row length makes every owned column run a whole number of
      // words, and an even matrix span keeps a pair inside one matrix.
      if (out_n >= n) {
        continue;
      }
      store_pair(
          batch,
          out_m,
          out_n,
          vec2(
              results[r][c],
              results[r][c + 1]));
    }
  }
}
