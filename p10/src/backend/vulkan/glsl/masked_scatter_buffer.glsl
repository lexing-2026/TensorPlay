#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Spreads the leading elements of a source payload over the positions a mask
 * selects, in the order the payload holds them, and carries the elements the
 * mask leaves alone straight through.
 *
 * The walk is the one the count pass ran: the same `lane_words` words per
 * invocation, the same workgroup.  A block therefore takes source elements
 * starting where the blocks before it stopped - the offsets the host derived
 * from the per-block counts say where - and inside the block the same scan
 * gives each invocation the rank of its own selections.  Every element belongs
 * to exactly one invocation, which reads it and writes it, so the payload needs
 * no lock and no subgroup operation.
 *
 * Elements are addressed as words of a four-lane group: one mask word covers
 * one payload word, so a block of the walk copies four elements at a time.
 */
#include "mask_count.h"

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  vec4 data[];
}
uInput;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mask {
  uint data[];
}
uMask;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly Base {
  int data[];
}
uBase;
layout(set = 0, binding = 4) buffer PRECISION restrict readonly Source {
  vec4 data[];
}
uSource;
layout(set = 0, binding = 5) uniform PRECISION restrict Block {
  int elems;      // elements the payload holds
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
  for (int stride = 1; stride < 256; stride <<= 1) {
    const int below = lane >= stride ? sCount[lane - stride] : 0;
    barrier();
    sCount[lane] += below;
    barrier();
  }

  // The source element this lane's first selection takes: every selection below
  // it inside the block, plus where the block starts in the source.
  int slot = uBase.data[block] + sCount[lane] - count;

  for (int i = 0; i < uBlock.lane_words; ++i) {
    const int word = first_word + i;
    if (word >= uBlock.words) {
      break;
    }
    const uint bits = uMask.data[word];
    const bool whole = 4 * word + 4 <= uBlock.elems;
    // A word no selection touches copies its four elements in one store, which
    // is the usual case: a mask marks the few positions it names.  A word with
    // a selection - and the last one, which the payload may cut short - is
    // written lane by lane, each selection taking the next source element.
    if (whole && bits == 0u) {
      uOutput.data[word] = uInput.data[word];
      continue;
    }
    const int bytes = whole ? 4 : uBlock.elems - 4 * word;
    for (int byte = 0; byte < bytes; ++byte) {
      if (!mask_byte_set(bits, byte)) {
        uOutput.data[word][byte] = uInput.data[word][byte];
        continue;
      }
      uOutput.data[word][byte] = uSource.data[slot >> 2][slot & 3];
      ++slot;
    }
  }
}
