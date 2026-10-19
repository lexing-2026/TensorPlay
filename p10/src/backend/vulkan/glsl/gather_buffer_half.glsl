#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer gather over half payloads: the source and the output are
// dense row-major buffers packing two lanes into each 32-bit word, the index
// list a buffer of Int32 words.  One invocation owns one whole output word, so a
// payload whose element count is odd still ends inside the allocation, which is
// padded to a word, and the two lanes of a value are moved as 16-bit fields
// without ever widening them.  A code naming no row of the source contributes a
// zero lane.
#include "gather_mapping.h"

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
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

// The 16-bit field holding one element of a dense half payload.
uint half_field(uint element) {
  return (uInput.data[element >> 1u] >> ((element & 1u) * 16u)) & 0xffffu;
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uint(uBlock.count);
  if (word >= ((n + 1u) >> 1u)) {
    return;
  }

  uint packed = 0u;
  for (uint lane = 0u; lane < 2u; ++lane) {
    const uint element = (word << 1u) + lane;
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
    packed |= half_field(source) << (16u * lane);
  }
  uOutput.data[word] = packed;
}
