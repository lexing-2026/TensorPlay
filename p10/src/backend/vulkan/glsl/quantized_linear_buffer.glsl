#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OUTPUT_BUFFER ${OUTPUT_BUFFER}
// clang-format on

layout(std430) buffer;

/*
 * Int8 linear over operands held in linear storage:
 *   out[m, n] = clamp(roundEven((input_scale * w_scale[n] * acc + bias[n])
 *                             / out_scale) + out_zero_point,
 *                     quant_min, quant_max)
 * where acc = sum_k (x[m,k] - x_zp) * (w[n,k] - w_zp[n]) accumulates in int32.
 *
 * Both operands are plain row-major bytes, four codes to a word, so the walk
 * needs no packed form: a row of the activation is a run of K bytes and a row
 * of the weight is a run of K bytes, each named by its own base.  A row whose
 * length divides by four therefore starts on a word edge and both operands give
 * four reduction terms per fetch; a row that does not splits its codes across
 * word boundaries, so each code is picked off its own byte lane instead.  The
 * per-channel scale, zero point and addend ride in one single-precision
 * payload: the first `params_words` vec4 slots hold the scales, the next that
 * many hold the zero points, the next hold the addends.
 *
 * One invocation owns one output element.  Four adjacent codes share a word in
 * linear storage and a texel store cannot read-modify-write without racing its
 * neighbours, so the buffer build takes four elements per invocation and writes
 * the whole word it fills; the image build takes one.  A tail word repeats the
 * last element in the lanes the logical extent does not name.
 */
// clang-format off
$if OUTPUT_BUFFER == 1:
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
    uint data[];
  }
  uOutput;
$else:
  layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
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
  int params_words; // ceil(N / 4): words in one parameter row
  float input_scale;
  int input_zero_point;
  float out_inv_scale; // 1 / out_scale
  int out_zero_point;
  int quant_min;
  int quant_max;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// A payload names its lanes as signed, so the high bit is the sign.
int signed_lane(const in uint byte) {
  return byte > 127u ? int(byte) - 256 : int(byte);
}

// The four consecutive codes one word carries, signed the same way.
ivec4 signed_lanes_of(const in uint word) {
  return ivec4(
      signed_lane(word & 0xFFu),
      signed_lane((word >> 8u) & 0xFFu),
      signed_lane((word >> 16u) & 0xFFu),
      signed_lane((word >> 24u) & 0xFFu));
}

// Eight bits of one word hold the code the linear position names.
int act_code(const in uint linear_offset) {
  return signed_lane(
      (uAct.data[linear_offset >> 2u] >> (8u * (linear_offset & 3u))) & 0xFFu);
}

int weight_code(const in uint linear_offset) {
  return signed_lane(
      (uWeight.data[linear_offset >> 2u] >> (8u * (linear_offset & 3u))) &
          0xFFu);
}

float scale_of(const in int col) {
  return uParams.data[col >> 2][col & 3];
}

int zero_point_of(const in int col) {
  return int(uParams.data[uBlock.params_words + (col >> 2)][col & 3]);
}

float bias_of(const in int col) {
  return uParams.data[2 * uBlock.params_words + (col >> 2)][col & 3];
}

int dot_row(const in int row, const in int col) {
  const int w_zp = zero_point_of(col);
  int acc = 0;
  if ((uBlock.k & 3) == 0) {
    const int words = uBlock.k >> 2;
    const int act_words = row * words;
    const int weight_words = col * words;
    for (int j = 0; j < words; ++j) {
      const ivec4 a = signed_lanes_of(uAct.data[act_words + j]) -
          ivec4(uBlock.input_zero_point);
      const ivec4 w =
          signed_lanes_of(uWeight.data[weight_words + j]) - ivec4(w_zp);
      acc += a.x * w.x + a.y * w.y + a.z * w.z + a.w * w.w;
    }
  } else {
    const int act_base = row * uBlock.k;
    const int weight_base = col * uBlock.k;
    for (int kk = 0; kk < uBlock.k; ++kk) {
      acc += (act_code(uint(act_base + kk)) - uBlock.input_zero_point) *
          (weight_code(uint(weight_base + kk)) - w_zp);
    }
  }
  return acc;
}

int requantize(const in int row, const in int col) {
  const float out_f =
      uBlock.input_scale * scale_of(col) * float(dot_row(row, col)) +
      bias_of(col);
  return clamp(
      int(roundEven(out_f * uBlock.out_inv_scale)) + uBlock.out_zero_point,
      uBlock.quant_min,
      uBlock.quant_max);
}

void main() {
  const int n = uBlock.out_sizes.x;
  const int m = uBlock.out_sizes.y;

  // clang-format off
  $if OUTPUT_BUFFER == 1:
    const int first = 4 * int(gl_GlobalInvocationID.x);
    if (first >= n * m) {
      return;
    }
    const int last = n * m - 1;
    uint word = 0u;
    for (int i = 0; i < 4; ++i) {
      const int e = min(first + i, last);
      const int row = e / n;
      const int q = requantize(row, e - row * n);
      word |= (uint(q & 0xFF) << (8u * uint(i)));
    }
    uOutput.data[int(gl_GlobalInvocationID.x)] = word;
  $else:
    const int col = int(gl_GlobalInvocationID.x);
    const int row = int(gl_GlobalInvocationID.y);
    if (col >= n || row >= m) {
      return;
    }
    imageStore(uOutput, ivec3(col, row, 0), ivec4(requantize(row, col), 0, 0, 0));
  // clang-format on
}
