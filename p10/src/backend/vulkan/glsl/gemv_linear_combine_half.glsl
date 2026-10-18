#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OUTPUT_BUFFER ${OUTPUT_BUFFER}
#define HAS_BIAS ${HAS_BIAS}
// clang-format on

layout(std430) buffer;

/*
 * Second pass of the half-precision decode product: fold the split partial rows
 * of one output column and apply the epilogue.  The partials arrive as floats,
 * so the fold never re-accumulates narrowed terms, and only the result narrows.
 *
 * A half result in linear storage keeps two adjacent lanes in one 32-bit word,
 * so an invocation there owns a whole word: it folds the two output elements
 * that share it and stores both halves together, which leaves no read-modify-
 * write between neighbours.  The tail word of an odd element count repeats the
 * last element in its high lane, invisible because the logical extent stops
 * before it.  An image result needs no such pairing, since a texel channel
 * carries one element, so the two builds differ only in where the invocation's
 * output elements come from.
 */
// clang-format off
$if OUTPUT_BUFFER == 1:
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
    uint data[];
  }
  uOutput;
$else:
  layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D uOutput;
// clang-format on
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Partials {
  float data[];
}
uPartials;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 out_sizes;  // (W=N, H=M, C=1, N=1) logical product sizes
  ivec4 bias_sizes; // (W, H, C, N) logical sizes of the addend
  int splits;       // partial rows folded per output element
  int fill;
  float alpha;
  float beta;
}
uBlock;
// clang-format off
$if HAS_BIAS == 1:
  $if OUTPUT_BUFFER == 1:
    layout(set = 0, binding = 3) buffer PRECISION restrict readonly Bias {
      uint data[];
    }
    uBias;
  $else:
    layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// clang-format off
$if OUTPUT_BUFFER == 1 and HAS_BIAS == 1:
  float lane_of(const in uint word, const in uint lane) {
    const vec2 pair = unpackHalf2x16(word);
    return lane == 0u ? pair.x : pair.y;
  }
  float bias_lane(const in int index) {
    return lane_of(uBias.data[index >> 1], uint(index) & 1u);
  }
// clang-format on

float fold(const in int row, const in int col) {
  const int n = uBlock.out_sizes.x;

  float acc = 0.0;
  for (int s = 0; s < uBlock.splits; ++s) {
    acc += uPartials.data[(s * uBlock.out_sizes.y + row) * n + col];
  }
  acc = uBlock.alpha * acc;

  // clang-format off
  $if HAS_BIAS == 1:
    const int col_eff = (uBlock.bias_sizes.x == 1) ? 0 : col;
    const int row_eff = (uBlock.bias_sizes.y == 1) ? 0 : row;
    $if OUTPUT_BUFFER == 1:
      acc += uBlock.beta *
          bias_lane(row_eff * uBlock.bias_sizes.x + col_eff);
    $else:
      acc += uBlock.beta *
          texelFetch(uBias, ivec3(col_eff, row_eff, 0), 0).x;
  // clang-format on

  return acc;
}

void main() {
  const int n = uBlock.out_sizes.x;
  const int m = uBlock.out_sizes.y;

  // clang-format off
  $if OUTPUT_BUFFER == 1:
    const int first = 2 * int(gl_GlobalInvocationID.x);
    if (first >= n * m) {
      return;
    }
    const int second = min(first + 1, n * m - 1);
    const int row0 = first / n;
    const int row1 = second / n;
    uOutput.data[int(gl_GlobalInvocationID.x)] = packHalf2x16(vec2(
        fold(row0, first - row0 * n), fold(row1, second - row1 * n)));
  $else:
    const int col = int(gl_GlobalInvocationID.x);
    const int row = int(gl_GlobalInvocationID.y);
    if (col >= n || row >= m) {
      return;
    }
    imageStore(uOutput, ivec3(col, row, 0), vec4(fold(row, col)));
  // clang-format on
}
