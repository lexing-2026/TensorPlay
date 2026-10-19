#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define DTYPE ${DTYPE}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

layout(std430) buffer;
#include "pow_ext.h"

$if DTYPE == "int":
  // Signed-word storage buffers for the integer variants.  Data is laid out
  // as 16-byte vectors so the bulk of the work uses aligned 128-bit loads.
  #define VECTOR ivec4
  $if not INPLACE:
    layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
      VECTOR data[];
    }
    uOutput;
    layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
      VECTOR data[];
    }
    uInput;
    layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer2 {
      VECTOR data[];
    }
    uOther;
    layout(set = 0, binding = 3) uniform PRECISION Block {
      uint buf_length;
      uint fill0;
      int alpha;
      int broadcast;
      ivec4 out_strides;
      ivec4 input_sizes;
      ivec4 input_strides;
      ivec4 other_sizes;
      ivec4 other_strides;
    }
    uBlock;
  $else:
    layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
      VECTOR data[];
    }
    uOutput;
    layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer2 {
      VECTOR data[];
    }
    uOther;
    layout(set = 0, binding = 2) uniform PRECISION Block {
      uint buf_length;
      uint fill0;
      int alpha;
      int broadcast;
      ivec4 out_strides;
      ivec4 input_sizes;
      ivec4 input_strides;
      ivec4 other_sizes;
      ivec4 other_strides;
    }
    uBlock;
$else:
  // clang-format off
  #define VECTOR vec4
  $if not INPLACE:
    layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
      VECTOR data[];
    }
    uOutput;
    layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
      VECTOR data[];
    }
    uInput;
    layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer2 {
      VECTOR data[];
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
    layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
      VECTOR data[];
    }
    uOutput;
    layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer2 {
      VECTOR data[];
    }
    uOther;
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

/*
 * Element-wise binary operation over two linear buffers:
 * out = OP(in, other, alpha).  Each invocation handles one 16-byte vector
 * (four elements); a small scalar tail covers lengths that are not a
 * multiple of four.  Buffer allocations are rounded up to 16 bytes so the
 * tail accesses never leave the allocation.  `broadcast` is a bitmask of
 * which operand needs index remapping (bit 0: input, bit 1: other); when
 * either is set, the flat index of that operand is recomputed per output
 * element from row-major strides, so singleton dimensions expand without
 * materializing the broadcast tensor.
 */
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

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  const uint n = uBlock.buf_length;
  const uint nvec = n >> 2u;
  const uint rem = n & 3u;

  if (uBlock.broadcast == 0) {
    if (idx < nvec) {
      // clang-format off
      $if not INPLACE:
        uOutput.data[idx] =
            OP(uInput.data[idx], uOther.data[idx], uBlock.alpha);
      $else:
        uOutput.data[idx] =
            OP(uOutput.data[idx], uOther.data[idx], uBlock.alpha);
      // clang-format on
    } else if (idx == nvec) {
      const uint j = nvec;
      // The tail operands are scalars; wrap them in a vector so operators
      // written in vector form (e.g. pow's integer-exponent branch) compile
      // unchanged, then take the lane being written.  One invocation owns the
      // whole partial word, so the dispatch stays at div_up(n, 4) groups.
      // clang-format off
      $if not INPLACE:
        for (uint i = 0u; i < rem; ++i) {
          uOutput.data[j][i] =
              (OP(VECTOR(uInput.data[j][i]), VECTOR(uOther.data[j][i]),
                  uBlock.alpha)).x;
        }
      $else:
        for (uint i = 0u; i < rem; ++i) {
          uOutput.data[j][i] =
              (OP(VECTOR(uOutput.data[j][i]), VECTOR(uOther.data[j][i]),
                  uBlock.alpha)).x;
        }
      // clang-format on
    }
    return;
  }

  const uint lo = idx << 2u;
  const uint hi = min(lo + 4u, n);
  for (uint i = lo; i < hi; ++i) {
    uint a = i;
    uint b = i;
    if ((uBlock.broadcast & 1) != 0) {
      a = remap_index(i, uBlock.out_strides, uBlock.input_sizes,
                      uBlock.input_strides);
    }
    if ((uBlock.broadcast & 2) != 0) {
      b = remap_index(i, uBlock.out_strides, uBlock.other_sizes,
                      uBlock.other_strides);
    }
    const uint j = i >> 2u;
    const uint k = i & 3u;
    // clang-format off
    $if not INPLACE:
      uOutput.data[j][k] =
          (OP(VECTOR(uInput.data[a >> 2u][a & 3u]),
              VECTOR(uOther.data[b >> 2u][b & 3u]), uBlock.alpha)).x;
    $else:
      uOutput.data[j][k] =
          (OP(VECTOR(uOutput.data[a >> 2u][a & 3u]),
              VECTOR(uOther.data[b >> 2u][b & 3u]), uBlock.alpha)).x;
    // clang-format on
  }
}