#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Writes the coordinates a mask selects, in the order the payload holds them.
 * The walk is the one the count pass ran: the same `lane_words` words per
 * invocation, the same workgroup.  A block therefore starts where the blocks
 * before it stopped, which the offsets the host derived from the per-block
 * counts say, and inside the block the same scan gives each invocation the rank
 * of its own selections.  A lane then emits its elements in payload order, so
 * the result is one fixed sequence however the work is spread.
 *
 * The coordinates are the row-major ones of the payload: the innermost axis is
 * the remainder, and each axis above it is what dividing by the axes below
 * leaves.  They land as words, which the widening pass turns into the counts
 * the caller returns.
 */
#include "mask_count.h"

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  int data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Mask {
  uint data[];
}
uMask;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Base {
  int data[];
}
uBase;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int elems;      // elements the mask covers
  int words;      // packed mask words: div_up(elems, 4)
  int lane_words; // mask words one invocation owns
  int ndim;       // axes the coordinates name: sizes[0 .. ndim) are real
  ivec4 sizes;
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
  for (int stride = 1; stride < 256; stride <<= 1) {
    const int below = lane >= stride ? sCount[lane - stride] : 0;
    barrier();
    sCount[lane] += below;
    barrier();
  }

  // The rank of this lane's first selection: every rank below it inside the
  // block, plus where the block starts in the result.
  int slot = uBase.data[block] + sCount[lane] - count;

  for (int i = 0; i < uBlock.lane_words; ++i) {
    const int word = first_word + i;
    if (word >= uBlock.words) {
      break;
    }
    const uint bits = uMask.data[word];
    for (int byte = 0; byte < 4; ++byte) {
      const int element = 4 * word + byte;
      if (element >= uBlock.elems) {
        break;
      }
      if (!mask_byte_set(bits, byte)) {
        continue;
      }
      int rest = element;
      const int row = slot * uBlock.ndim;
      for (int axis = uBlock.ndim - 1; axis >= 0; --axis) {
        const int size = uBlock.sizes[axis];
        uOutput.data[row + axis] = rest - (rest / size) * size;
        rest /= size;
      }
      ++slot;
    }
  }
}
