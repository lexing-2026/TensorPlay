#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Mark pass of a scatter: one invocation per entry of the index list records
// which entry names which element of the destination.  The marks are a plain
// row-major payload with one word per element, so a mark is a whole-word store
// and the pass needs no read-modify-write, which a texel or a packed word would
// lose to a racing invocation.  Two entries that name the same element both
// record and the apply pass reads one of them, which is the order freedom a
// repeated code already carries.
//
// The addressing is the gather mapping run the other way: an entry of the index
// list decomposes over the list's own strides, its code replaces the coordinate
// on the scattered axis, and the coordinates recompose with the destination
// strides.  A code naming no row of the destination marks nothing, the same
// answer a row lookup gives to a request outside its rows.
#include "gather_mapping.h"

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Marks {
  int data[];
}
uMarks;
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
  const int destination = gather_source_element(
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
  if (destination < 0) {
    return;
  }
  // One is added so that a word the clearing pass left at zero stays
  // distinguishable from the first entry of the list.
  uMarks.data[destination] = element + 1;
}
