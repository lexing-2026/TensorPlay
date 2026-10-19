#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer gather over single-precision payloads: the source, the index
// list and the output are plain row-major buffers, so any axis can be gathered -
// including the channel and batch axes, whose stride is four lanes of a texel on
// the image route.  One invocation owns one whole output word, which is four
// elements, and the output allocation is padded to a word, so the partial last
// word is written as a unit instead of being left to whichever invocation reached
// the tail.  A code naming no row of the source contributes a zero element.
#include "gather_mapping.h"

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  vec4 data[];
}
uInput;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int out_stride_0;   // strides of the output shape, left-padded to four axes
  int out_stride_1;
  int out_stride_2;
  int out_stride_3;
  int self_stride_0;  // strides of the source shape, in the same padding
  int self_stride_1;
  int self_stride_2;
  int self_stride_3;
  int axis;           // the gathered axis of the padded shape
  int self_rows;      // entries available along the gathered axis
  int count;          // elements of the output
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uint(uBlock.count);
  if (word >= ((n + 3u) >> 2u)) {
    return;
  }

  vec4 value = vec4(0.0);
  for (uint lane = 0u; lane < 4u; ++lane) {
    const uint element = (word << 2u) + lane;
    if (element >= n) {
      break;
    }
    const int source = gather_source_element(
        int(element),
        uIndices.data[element],
        uBlock.out_stride_0,
        uBlock.out_stride_1,
        uBlock.out_stride_2,
        uBlock.out_stride_3,
        uBlock.self_stride_0,
        uBlock.self_stride_1,
        uBlock.self_stride_2,
        uBlock.self_stride_3,
        uBlock.axis,
        uBlock.self_rows);
    if (source < 0) {
      continue;
    }
    value[int(lane)] =
        uInput.data[uint(source) >> 2u][uint(source) & 3u];
  }
  uOutput.data[word] = value;
}
