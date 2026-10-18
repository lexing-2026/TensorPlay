#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FLOAT_IMAGE_FORMAT}

#define A_BUFFER ${A_BUFFER}
#define OUTPUT_BUFFER ${OUTPUT_BUFFER}
#define HAS_BIAS ${HAS_BIAS}
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
 * Tiled matrix product for half-precision operands that live in linear storage.
 * A matrix with an axis past the device's image limit has no texture form, and
 * a device runtime keeps weights and activations narrowed to half, so a payload
 * word carries two lanes, low lane first.
 *
 * A workgroup of 8x8 threads, each owning a 4x4 output tile, covers a 32x32
 * output block.  Both sub-blocks are staged cooperatively into shared memory,
 * and staging is the only place a lane is unpacked: the tiles hold floats, so
 * the accumulation loop never narrows between terms and never repeats the
 * unpack.  The footprint stays 2 * 32 * 33 floats, and the kernel needs no
 * subgroup operation, no image operand and no 16-bit storage.
 *
 * A half result in linear storage writes whole words, and a thread owns four
 * consecutive columns of each of its four rows, so pairing needs the result's
 * row length to be even: the host rejects an odd width rather than letting two
 * threads claim one word.  Staging writes zero outside the operand extents,
 * which is what keeps a reduction length that is not a multiple of the tile
 * size, and an output edge, from contributing anything.
 */
// clang-format off
$if OUTPUT_BUFFER == 1:
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
    uint data[];
  }
  uOutput;
$else:
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
// clang-format on
// clang-format off
$if A_BUFFER == 1:
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly Mat1 {
    uint data[];
  }
  uMat1;
$else:
  layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
// clang-format on
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mat2 {
  uint data[];
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
    uint data[];
  }
  uBias;
$elif HAS_BIAS == 1:
  layout(set = 0, binding = 4) uniform PRECISION sampler3D uBias;
// clang-format on

shared float sA[TILE][TK + PAD];
shared float sB[TILE][TK + PAD];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float lane_of(const in uint word, const in uint lane) {
  const vec2 pair = unpackHalf2x16(word);
  return lane == 0u ? pair.x : pair.y;
}

float weight_at(const in int index) {
  return lane_of(uMat2.data[index >> 1], uint(index) & 1u);
}

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
        v = lane_of(
            uMat1.data[(m_glob * uBlock.a_stride_m + k * uBlock.a_stride_k) >> 1],
            uint(m_glob * uBlock.a_stride_m + k * uBlock.a_stride_k) & 1u);
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
        v = weight_at(n_glob * uBlock.b_stride_n + k * uBlock.b_stride_k);
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
        v = weight_at(n_glob * uBlock.b_stride_n + k * uBlock.b_stride_k);
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
    const int index = row_eff * uBlock.bias_sizes.x + col_eff;
    return lane_of(uBias.data[index >> 1], uint(index) & 1u);
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

// clang-format off
$if OUTPUT_BUFFER == 1:
  // One store covers the pair of neighbouring lanes inside a word.
  void store_pair(const int row, const int col, const vec2 value) {
    uOutput.data[(row * uBlock.out_sizes.x + col) >> 1] = packHalf2x16(value);
  }
$else:
  void store_one(const int row, const int col, const float value) {
    imageStore(uOutput, ivec3(col, row, 0), vec4(value));
  }
// clang-format on

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
    // clang-format off
    $if OUTPUT_BUFFER == 1:
      for (int c = 0; c < FOUR; c += 2) {
        const int out_n = wg_base_n + local_n + c;
        // An even row length makes every owned column run a whole number of
        // words, so a pair either sits entirely inside the result or entirely
        // past it.
        if (out_n >= n) {
          continue;
        }
        const vec2 pair = vec2(
            uBlock.alpha * results[r][c] +
                uBlock.beta * bias_at(out_m, out_n),
            uBlock.alpha * results[r][c + 1] +
                uBlock.beta * bias_at(out_m, out_n + 1));
        store_pair(out_m, out_n, pair);
      }
    $else:
      for (int c = 0; c < FOUR; ++c) {
        const int out_n = wg_base_n + local_n + c;
        if (out_n >= n) {
          continue;
        }
        store_one(out_m, out_n, uBlock.alpha * results[r][c] +
            uBlock.beta * bias_at(out_m, out_n));
      }
    // clang-format on
  }
}
