#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

layout(std430) buffer;
#include "pow_ext.h"

/*
 * Element-wise binary op over two half-precision linear payloads:
 * out = OP(in, other, alpha).  Two fp16 lanes share one 32-bit word, low half
 * first, and one invocation owns the same word of every operand, so a length
 * that is not even never leaves two writers racing over one word.  The
 * arithmetic is fp32 with a single rounding at the store.
 *
 * `broadcast` is a bitmask of which operand needs index remapping (bit 0:
 * input, bit 1: other).  A remapped lane is fetched from whichever word holds
 * it, so singleton dimensions expand without materializing the broadcast
 * tensor.  Padding lanes past the logical length are written but never read
 * back, since every linear kernel bounds its reads by buf_length.
 */
// clang-format off
$if not INPLACE:
  #define SRC_WORD(W) uInput.data[W]
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
    uint data[];
  }
  uOutput;
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
    uint data[];
  }
  uInput;
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer2 {
    uint data[];
  }
  uOther;
  layout(set = 0, binding = 3) uniform PRECISION Block {
    uint buf_length;
    uint fill0;
    float alpha;
    int broadcast;
    ivec4 out_strides;
    ivec4 input_sizes;
    ivec4 input_strides;
    ivec4 other_sizes;
    ivec4 other_strides;
  }
  uBlock;
$else:
  #define SRC_WORD(W) uOutput.data[W]
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    uint data[];
  }
  uOutput;
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer2 {
    uint data[];
  }
  uOther;
  // Numbered per variant: the descriptor set layout is rebuilt from the
  // ordered binding list, so the params block takes the first free number
  // rather than the slot reserved for the absent input buffer.
  layout(set = 0, binding = 2) uniform PRECISION Block {
    uint buf_length;
    uint fill0;
    float alpha;
    int broadcast;
    ivec4 out_strides;
    ivec4 input_sizes;
    ivec4 input_strides;
    ivec4 other_sizes;
    ivec4 other_strides;
  }
  uBlock;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

uint remap_index(
    uint i, ivec4 out_strides, ivec4 sizes, ivec4 strides) {
  uint rem = i;
  uint acc = 0u;
  for (int d = 0; d < 4; ++d) {
    const uint s = uint(out_strides[d]);
    if (s == 0u) {
      break;
    }
    const uint coord = rem / s;
    rem = rem % s;
    if (uint(sizes[d]) > 1u) {
      acc += coord * uint(strides[d]);
    }
  }
  return acc;
}

// The fp16 lane of `element`, addressed by its position inside the word.
float half_lane(const in uint word, const in uint lane) {
  const vec2 unpacked = unpackHalf2x16(word);
  return lane == 0u ? unpacked.x : unpacked.y;
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uBlock.buf_length;
  const uint i0 = word << 1u;

  if (i0 >= n) {
    return;
  }

  const bool tail = i0 + 1u >= n;
  // Clamped so a remapped read of a padding lane stays inside the operand; the
  // result of that lane is dropped below.
  const uint i1 = tail ? i0 : (i0 + 1u);

  if (uBlock.broadcast == 0) {
    const vec2 x = unpackHalf2x16(SRC_WORD(word));
    const vec2 y = unpackHalf2x16(uOther.data[word]);
    uOutput.data[word] = packHalf2x16(vec2(
        OP(x.x, y.x, uBlock.alpha),
        tail ? x.y : OP(x.y, y.y, uBlock.alpha)));
    return;
  }

  const bool remap_input = (uBlock.broadcast & 1) != 0;
  const bool remap_other = (uBlock.broadcast & 2) != 0;

  const uint a0 = remap_input
      ? remap_index(i0, uBlock.out_strides, uBlock.input_sizes,
                    uBlock.input_strides)
      : i0;
  const uint a1 = remap_input
      ? remap_index(i1, uBlock.out_strides, uBlock.input_sizes,
                    uBlock.input_strides)
      : i1;
  const uint b0 = remap_other
      ? remap_index(i0, uBlock.out_strides, uBlock.other_sizes,
                    uBlock.other_strides)
      : i0;
  const uint b1 = remap_other
      ? remap_index(i1, uBlock.out_strides, uBlock.other_sizes,
                    uBlock.other_strides)
      : i1;

  const vec2 x = vec2(
      half_lane(SRC_WORD(a0 >> 1u), a0 & 1u),
      half_lane(SRC_WORD(a1 >> 1u), a1 & 1u));
  const vec2 y = vec2(
      half_lane(uOther.data[b0 >> 1u], b0 & 1u),
      half_lane(uOther.data[b1 >> 1u], b1 & 1u));

  vec2 r = vec2(
      OP(x.x, y.x, uBlock.alpha),
      tail ? x.y : OP(x.y, y.y, uBlock.alpha));

  uOutput.data[word] = packHalf2x16(r);
}
