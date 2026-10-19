#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

layout(std430) buffer;
#include "pow_ext.h"

#include "math_ext.h"

/*
 * Unary op with one scalar operand over a half-precision linear payload:
 * out = OP(in, alpha).  The word packing follows buffer_half_unary_op: two
 * fp16 lanes per 32-bit word, low half first, one word per invocation, fp32
 * arithmetic with a single rounding at the store.
 */
// clang-format off
$if not INPLACE:
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
    uint data[];
  }
  uOutput;
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
    uint data[];
  }
  uInput;
  layout(set = 0, binding = 2) uniform PRECISION Block {
    uint buf_length;
    uint fill0;
    float alpha;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    uint data[];
  }
  uOutput;
  // Numbered per variant; see buffer_half_unary_op for the layout rule.
  layout(set = 0, binding = 1) uniform PRECISION Block {
    uint buf_length;
    uint fill0;
    float alpha;
  }
  uBlock;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint base = word << 1u;

  if (base >= uBlock.buf_length) {
    return;
  }

  // clang-format off
  $if not INPLACE:
    vec2 v = unpackHalf2x16(uInput.data[word]);
  $else:
    vec2 v = unpackHalf2x16(uOutput.data[word]);
  // clang-format on

  v.x = OP(v.x, uBlock.alpha, 1.0f);
  if (base + 1u < uBlock.buf_length) {
    v.y = OP(v.y, uBlock.alpha, 1.0f);
  }
  uOutput.data[word] = packHalf2x16(v);
}
