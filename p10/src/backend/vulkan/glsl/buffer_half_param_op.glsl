#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, P0, P1) ${OPERATOR}
// clang-format on

layout(std430) buffer;

#include "math_ext.h"

/*
 * Unary op with two affine parameters over a half-precision linear payload.
 * The word packing follows buffer_half_unary_op: two fp16 lanes per 32-bit
 * word, low half first, one word per invocation, and fp32 arithmetic with a
 * single rounding at the store.  This family covers the parameterized ops
 * (leaky relu, threshold, hard shrink) and the closed clip range, whose two
 * bounds arrive in the same two slots.
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
    float p0;
    float p1;
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
    float p0;
    float p1;
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

  const float p0 = uBlock.p0;
  const float p1 = uBlock.p1;

  // clang-format off
  $if not INPLACE:
    vec2 v = unpackHalf2x16(uInput.data[word]);
  $else:
    vec2 v = unpackHalf2x16(uOutput.data[word]);
  // clang-format on

  v.x = OP(v.x, p0, p1);
  if (base + 1u < uBlock.buf_length) {
    v.y = OP(v.y, p0, p1);
  }
  uOutput.data[word] = packHalf2x16(v);
}
