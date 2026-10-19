#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OUTPUT_BUFFER ${OUTPUT_BUFFER}
// clang-format on

layout(std430) buffer;

/*
 * Weight-only 4-bit linear over operands held in linear storage:
 *   out[m, n] = w_scale[n] * sum_k act[m, k] * code[n, k] + bias[n]
 *
 * The activation payload keeps two halves in every 32-bit word, low lane first,
 * so it is read by element index through an unpack.  The weight arrives already
 * nibble-packed, eight codes to a word and one row per run of ceil(K / 8) words,
 * so one fetch names eight consecutive reduction terms and a row always starts
 * on a word edge.  Scale and bias ride in one single-precision payload: the
 * first `params_words` vec4 slots hold the per-output-channel scales, the next
 * that many hold the addends.  Both are applied after the fold, in single
 * precision, so the epilogue narrows once and the result depends on neither
 * which storage the operands drew nor which route ran.
 *
 * Accumulation stays in single precision across the whole reduction.  One
 * invocation owns one output element; a half result in linear storage instead
 * owns a whole word, because two adjacent lanes share it and an image store
 * cannot read-modify-write without racing its neighbours.  The tail word of an
 * odd element count repeats the last element in its high lane, invisible
 * because the logical extent stops before it.
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
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Act {
  uint data[];
}
uAct;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Weight {
  uint data[];
}
uWeight;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly Params {
  vec4 data[];
}
uParams;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 out_sizes; // (W=N, H=M, C=1, N=1) logical product sizes
  int k; // length of the reduction axis
  int k_words; // ceil(k / 8): packed weight words per row
  int params_words; // ceil(N / 4): words in one parameter row
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float act_at(const in int index) {
  const vec2 pair = unpackHalf2x16(uAct.data[index >> 1]);
  return (uint(index) & 1u) == 0u ? pair.x : pair.y;
}

// The unsigned nibbles 8..15 stand for the codes -8..-1.
float code_at(const in uint packed, const in int i) {
  const uint nib = (packed >> uint(i * 4)) & 0xFu;
  return float(int(nib) - int(nib >= 8u ? 16u : 0u));
}

float dot_row(const in int row, const in int col) {
  const int weight_base = col * uBlock.k_words;
  const int act_base = row * uBlock.k;

  float acc = 0.0;
  for (int j = 0; j < uBlock.k_words; ++j) {
    const uint packed = uWeight.data[weight_base + j];
    const int first = j * 8;
    for (int i = 0; i < 8; ++i) {
      const int kk = first + i;
      // A row whose length leaves pad nibbles must not carry their activation
      // terms in from the next row.
      if (kk < uBlock.k) {
        acc += act_at(act_base + kk) * code_at(packed, i);
      }
    }
  }
  return acc;
}

float product(const in int row, const in int col) {
  const float scale = uParams.data[col >> 2][col & 3];
  const float bias = uParams.data[uBlock.params_words + (col >> 2)][col & 3];
  return scale * dot_row(row, col) + bias;
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
        product(row0, first - row0 * n),
        product(row1, second - row1 * n)));
  $else:
    const int col = int(gl_GlobalInvocationID.x);
    const int row = int(gl_GlobalInvocationID.y);
    if (col >= n || row >= m) {
      return;
    }
    imageStore(uOutput, ivec3(col, row, 0), vec4(product(row, col)));
  // clang-format on
}
