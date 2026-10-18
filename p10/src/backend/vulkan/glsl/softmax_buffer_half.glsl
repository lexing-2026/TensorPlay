#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define LOG_MODE ${LOG_MODE}
// clang-format on

layout(std430) buffer;

/*
 * Row-wise softmax (or log-softmax) over a half-precision linear payload.  Two
 * adjacent elements share one 32-bit word, so an invocation owns one whole
 * output word: it derives the rows of the two elements it stores, takes their
 * statistics, and packs both lanes into a single write.  Neighbouring output
 * elements belong to different rows whenever the softmax axis is not the
 * innermost one, so the row statistics are looked up per element instead of
 * being walked once per row.
 *
 * The statistics themselves are the max-subtracted exponentials of the row:
 *   mx    = max over the axis
 *   total = sum of exp(x - mx)
 *   out   = exp(x - mx) / total          (LOG_MODE 0)
 *   out   = (x - mx) - log(total)        (LOG_MODE 1)
 * computed in single precision and narrowed once at the store.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int axis_size;
  int inner;
  int rows;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float element_at(const in int index) {
  const vec2 pair = unpackHalf2x16(uInput.data[index >> 1]);
  return (index & 1) == 0 ? pair.x : pair.y;
}

// The sort row an element belongs to: the outer block contributes the leading
// rows and the position inside the inner run picks one of them.
int row_of(const in int index) {
  const int outer = index / (uBlock.axis_size * uBlock.inner);
  const int rest = index - outer * (uBlock.axis_size * uBlock.inner);
  const int inner_pos = rest - (rest / uBlock.inner) * uBlock.inner;
  return outer * uBlock.inner + inner_pos;
}

void row_stats(const in int row, out float mx, out float total) {
  const int outer = row / uBlock.inner;
  const int inner_pos = row - outer * uBlock.inner;
  const int base = outer * (uBlock.axis_size * uBlock.inner);

  mx = element_at(base + inner_pos);
  for (int i = 1; i < uBlock.axis_size; ++i) {
    mx = max(mx, element_at(base + i * uBlock.inner + inner_pos));
  }

  total = 0.0;
  for (int i = 0; i < uBlock.axis_size; ++i) {
    total += exp(element_at(base + i * uBlock.inner + inner_pos) - mx);
  }
}

float normalized(const in int index, const in float mx, const in float total) {
  const float shifted = element_at(index) - mx;
  // clang-format off
  $if LOG_MODE == 1:
    return shifted - log(total);
  $else:
    return exp(shifted) / total;
  // clang-format on
}

void main() {
  const int word = int(gl_GlobalInvocationID.x);
  const int elements = uBlock.rows * uBlock.axis_size;
  const int first = 2 * word;
  if (first >= elements) {
    return;
  }
  // An odd element count repeats the last value in the tail lane; nothing reads
  // it, because the reduced extent stops before it.
  const int second = min(first + 1, elements - 1);

  const int row0 = row_of(first);
  const int row1 = row_of(second);

  float mx0 = 0.0;
  float total0 = 0.0;
  row_stats(row0, mx0, total0);

  float mx1 = mx0;
  float total1 = total0;
  if (row1 != row0) {
    row_stats(row1, mx1, total1);
  }

  uOutput.data[word] = packHalf2x16(
      vec2(normalized(first, mx0, total0), normalized(second, mx1, total1)));
}
