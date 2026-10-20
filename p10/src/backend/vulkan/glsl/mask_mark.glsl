#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Counts, per workgroup, how many elements of a mask hold a value.  One
 * workgroup covers `lane_words` packed mask words per invocation, so a payload
 * of any length is walked by blocks that each store one aggregate entry.
 *
 * The aggregate is handed to the host because the length of the compacted
 * result is only known once the whole mask has been read: the positions are
 * written by a second walk that offsets each block by the counts of the blocks
 * before it.  A sum is order-independent, so nothing here depends on how many
 * invocations a device gives the queue, and no block boundary splits a packed
 * word, so no element is counted twice.  The walk needs no subgroup operation.
 */
#include "mask_count.h"

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Agg {
  int data[];
}
uAgg;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Mask {
  uint data[];
}
uMask;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int elems;      // elements the mask covers
  int words;      // packed mask words: div_up(elems, 4)
  int lane_words; // mask words one invocation owns
  int fill;
}
uBlock;

shared int sCount[256];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int lane = int(gl_LocalInvocationID.x);
  const int block = int(gl_WorkGroupID.x);
  const int first_word = (block * 256 + lane) * uBlock.lane_words;

  int count = 0;
  for (int i = 0; i < uBlock.lane_words; ++i) {
    const int word = first_word + i;
    if (word >= uBlock.words) {
      break;
    }
    count += mask_word_count(uMask.data[word], 4 * word, uBlock.elems);
  }

  sCount[lane] = count;
  barrier();
  // Inclusive scan over the workgroup, so the top lane ends up holding its
  // total.  Every step reads what the lane below wrote in the step before it,
  // which the barrier around each write keeps ordered.
  for (int stride = 1; stride < 256; stride <<= 1) {
    const int below = lane >= stride ? sCount[lane - stride] : 0;
    barrier();
    sCount[lane] += below;
    barrier();
  }
  if (lane == 0) {
    uAgg.data[block] = sCount[255];
  }
}
