#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

/*
 * Elementwise logic over Bool payloads in linear storage.
 *
 * One byte per element, four lanes to a word: lane k of word i occupies bits
 * [8k, 8k + 8).  An invocation owns one destination word, so the four lanes it
 * writes never race with another invocation.  A word past the logical end of a
 * payload is still inside its allocation, which is rounded up to sixteen bytes,
 * and nothing reads those lanes back.
 *
 * Lanes are decided by "nonzero means true" and written as 0 or 1, so a payload
 * that carries a value other than one for true still behaves as a Bool.
 *
 * With `broadcast` set (bit 0: the first operand, bit 1: the second), an
 * element's flat index is recomputed from row-major strides, so a dimension of
 * one repeats without the operand being materialized at the output shape.
 */
uint remap_index(
    uint i,
    ivec4 out_strides,
    ivec4 sizes,
    ivec4 strides) {
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

// The four lanes of one word, each reduced to whether its byte is nonzero.
ivec4 unpack_lanes(const uint word) {
  return ivec4(
      (word & 255u) != 0u ? 1 : 0,
      ((word >> 8u) & 255u) != 0u ? 1 : 0,
      ((word >> 16u) & 255u) != 0u ? 1 : 0,
      ((word >> 24u) & 255u) != 0u ? 1 : 0);
}

int unpack_lane(const uint word, const uint k) {
  return ((word >> (k * 8u)) & 255u) != 0u ? 1 : 0;
}

uint pack_lanes(const ivec4 lanes) {
  return (uint(lanes.x) & 1u) | ((uint(lanes.y) & 1u) << 8u) |
      ((uint(lanes.z) & 1u) << 16u) | ((uint(lanes.w) & 1u) << 24u);
}

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
$if not UNARY:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly Other {
    uint data[];
  }
  uOther;

$if UNARY:
  layout(set = 0, binding = 2) uniform PRECISION restrict Block {
    uint buf_length; // element count
    uint fill0;
    int broadcast; // 1: the operand needs index remapping
    uint fill1;
    ivec4 out_strides; // row-major strides of the output shape
    ivec4 input_sizes; // the operand's own extents, with singletons
    ivec4 input_strides;
  }
  uBlock;
$else:
  layout(set = 0, binding = 3) uniform PRECISION restrict Block {
    uint buf_length; // element count
    uint fill0;
    int broadcast; // bit 0: the first operand, bit 1: the second
    uint fill1;
    ivec4 out_strides; // row-major strides of the output shape
    ivec4 input_sizes;
    ivec4 input_strides;
    ivec4 other_sizes;
    ivec4 other_strides;
  }
  uBlock;
// End of the operand bindings.

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= ((uBlock.buf_length + 3u) >> 2u)) {
    return;
  }

  ivec4 a = ivec4(0);
  ivec4 b = ivec4(0);

  if (uBlock.broadcast == 0) {
    // Both operands carry the output shape, so one word of each feeds one word
    // of the result.
    a = unpack_lanes(uInput.data[idx]);
    $if not UNARY:
      b = unpack_lanes(uOther.data[idx]);
  } else {
    for (uint k = 0u; k < 4u; ++k) {
      const uint i = idx * 4u + k;
      if (i >= uBlock.buf_length) {
        break;
      }
      const uint ai = (uBlock.broadcast & 1) != 0
          ? remap_index(i, uBlock.out_strides, uBlock.input_sizes,
                        uBlock.input_strides)
          : i;
      a[k] = unpack_lane(uInput.data[ai >> 2u], ai & 3u);
      $if not UNARY:
        const uint bi = (uBlock.broadcast & 2) != 0
            ? remap_index(i, uBlock.out_strides, uBlock.other_sizes,
                          uBlock.other_strides)
            : i;
        b[k] = unpack_lane(uOther.data[bi >> 2u], bi & 3u);
    }
  }

  uOutput.data[idx] = pack_lanes(OP(a, b));
}
