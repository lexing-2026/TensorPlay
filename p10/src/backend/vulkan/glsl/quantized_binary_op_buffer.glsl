#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define OPERATOR(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

/*
 * Quantized broadcast arithmetic over linear storage: each output element wraps
 * onto the operands through modulo on its own axes, a singleton axis repeating,
 * each operand byte is dequantized as (q - zero_point) * scale, the float
 * operation is applied, and the result is requantized with
 * round-to-nearest-even into [-128, 127] under the output qparams.
 *
 * The payload names its elements in the row-major order of its logical shape, so
 * the four lanes of one output word are four consecutive output positions whose
 * operands can sit anywhere else in their own storages.  Every byte is therefore
 * drawn by shifting and masking the word that holds it, and the four results are
 * merged back into one word the same way, because a buffer view of 8-bit
 * elements needs device features this baseline does not assume.
 *
 * Buffer allocations round up to 16 bytes, so a run stops before the invocations
 * whose first lane lies past the output payload.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly InputA {
  uint data[];
}
uInputA;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly InputB {
  uint data[];
}
uInputB;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_sizes; // (W, H, C, N) sizes of the output
  ivec4 out_strides; // the linear stride of each of those axes
  ivec4 a_sizes; // (W, H, C, N) sizes of operand A
  ivec4 a_strides;
  ivec4 b_sizes; // (W, H, C, N) sizes of operand B
  ivec4 b_strides;
  float a_scale;
  int a_zero_point;
  float b_scale;
  int b_zero_point;
  float inv_out_scale;
  int out_zero_point;
  uint buf_length; // output element count
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Bit 7 of a lane is the sign of a two's-complement int8 code, so a byte of 128
// or more names a negative value and the remaining seven bits name its magnitude.
int code_at(const uint linear_offset) {
  const uint byte =
      (uInputA.data[linear_offset >> 2u] >> (8u * (linear_offset & 3u))) &
      0xFFu;
  return int(byte & 0x7Fu) - int(byte & 0x80u);
}

int code_at_b(const uint linear_offset) {
  const uint byte =
      (uInputB.data[linear_offset >> 2u] >> (8u * (linear_offset & 3u))) &
      0xFFu;
  return int(byte & 0x7Fu) - int(byte & 0x80u);
}

// The position an output element draws from one operand: the output index is
// unwound against the output's own axes, from the fastest one outward, and each
// coordinate then wraps against that operand's extent before the operand's
// strides turn it back into a position in its storage.
int source_offset(const int out_index, const ivec4 sizes, const ivec4 strides) {
  int rest = out_index;
  const int w = rest % uBlock.out_sizes.x;
  rest /= uBlock.out_sizes.x;
  const int h = rest % uBlock.out_sizes.y;
  rest /= uBlock.out_sizes.y;
  const int c = rest % uBlock.out_sizes.z;
  const int n = rest / uBlock.out_sizes.z;
  return (w % sizes.x) * strides.x + (h % sizes.y) * strides.y +
      (c % sizes.z) * strides.z + (n % sizes.w) * strides.w;
}

// A clamped code needs the eight bits its lane keeps, and the low eight bits of
// the two's-complement form are exactly those bits.
uint packed_lane(const int clamped_code) {
  return uint(clamped_code & 0xFF);
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint first = word << 2u;
  if (first >= uBlock.buf_length) {
    return;
  }

  vec4 y;
  for (int i = 0; i < 4; ++i) {
    const int out_index = int(first) + i;
    const int a_offset = source_offset(
        out_index, uBlock.a_sizes, uBlock.a_strides);
    const int b_offset = source_offset(
        out_index, uBlock.b_sizes, uBlock.b_strides);
    const float xa =
        (float(code_at(uint(a_offset))) - float(uBlock.a_zero_point)) *
        uBlock.a_scale;
    const float xb =
        (float(code_at_b(uint(b_offset))) - float(uBlock.b_zero_point)) *
        uBlock.b_scale;
    y[i] = OPERATOR(xa, xb);
  }

  const ivec4 rounded =
      ivec4(roundEven(y * uBlock.inv_out_scale)) +
      ivec4(uBlock.out_zero_point);
  const ivec4 q = clamp(rounded, ivec4(-128), ivec4(127));

  uOutput.data[word] = packed_lane(q.x) | (packed_lane(q.y) << 8u) |
      (packed_lane(q.z) << 16u) | (packed_lane(q.w) << 24u);
}
