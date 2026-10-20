#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Naming pass of an accumulating scatter: one invocation per entry of the index
// list records which element of the destination that entry asks to grow.  The
// answer is kept one word per entry rather than one per element, so the entries
// that name the same element each keep their own slot and the accumulation pass
// can add them in the order the list states - a mark of one word per element
// would let only the last of them through, which a rewrite tolerates and a total
// does not.
//
// The addressing is the gather mapping run the other way: an entry of the list
// decomposes over the list's own strides, its code replaces the coordinate on the
// scattered axis, and the coordinates recompose with the destination strides.  A
// code naming no row is written as -1, the same answer a row lookup gives to a
// request outside its rows, and the accumulation pass skips it.
#include "gather_mapping.h"

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Names {
  int data[];
}
uNames;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int index_stride_0;  // strides of the index shape, left-padded to four axes
  int index_stride_1;
  int index_stride_2;
  int index_stride_3;
  int self_stride_0;   // strides of the destination shape, in that padding
  int self_stride_1;
  int self_stride_2;
  int self_stride_3;
  int axis;            // the scattered axis of the padded shape
  int self_rows;       // entries available along the scattered axis
  int count;           // entries of the index list
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int element = int(gl_GlobalInvocationID.x);
  if (element >= uBlock.count) {
    return;
  }
  uNames.data[element] = gather_source_element(
      element,
      uIndices.data[element],
      uBlock.index_stride_0,
      uBlock.index_stride_1,
      uBlock.index_stride_2,
      uBlock.index_stride_3,
      uBlock.self_stride_0,
      uBlock.self_stride_1,
      uBlock.self_stride_2,
      uBlock.self_stride_3,
      uBlock.axis,
      uBlock.self_rows);
}
