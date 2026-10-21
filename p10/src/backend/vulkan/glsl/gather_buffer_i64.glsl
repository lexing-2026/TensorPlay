#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer gather over a payload of wide counts: a code wider than one
// word has no texel form and no lane to split across, so one element owns a
// whole pair of words and one invocation moves one element.  Any axis can be
// gathered, since the walk is over the row-major order of both payloads.  A code
// naming no row of the source contributes a zero pair.
#include "gather_mapping.h"

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uvec2 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uvec2 data[];
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
  const uint element = gl_GlobalInvocationID.x;
  if (element >= uint(uBlock.count)) {
    return;
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
  uOutput.data[element] =
      source < 0 ? uvec2(0u, 0u) : uInput.data[uint(source)];
}
