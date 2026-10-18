#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X) ${OPERATOR}
// clang-format on

layout(std430) buffer;

#include "math_ext.h"

/*
 * Unary elementwise op over a half-precision linear payload.  A storage buffer
 * has no texel format to carry the element type, so fp16 lives in its own
 * shader family: two lanes share one 32-bit word, and lane 2i is the low half
 * of word i.  One invocation owns one word, which keeps an odd-length tail
 * inside a single writer instead of racing two of them over a shared word.
 * The arithmetic runs in fp32 and rounds once at the store, the same accuracy
 * tier the rgba16f texture builds get from the hardware format conversion.
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
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    uint data[];
  }
  uOutput;
  // Bindings are numbered per variant: the descriptor set layout is rebuilt
  // from the ordered binding list, so the params block takes the first free
  // number rather than the slot reserved for the absent input buffer.
  layout(set = 0, binding = 1) uniform PRECISION Block {
    uint buf_length;
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

  v.x = OP(v.x);
  if (base + 1u < uBlock.buf_length) {
    v.y = OP(v.y);
  }
  uOutput.data[word] = packHalf2x16(v);
}
