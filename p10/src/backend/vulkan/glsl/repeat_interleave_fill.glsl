#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * The source position of every slot in an expansion.  One segment owns the
 * answer slots its count spans and writes its own number into all of them, so
 * no slot is written twice.  A segment is served by kLanesPerSegment
 * consecutive invocations that stride through its span, which keeps a run of
 * counts with a few wide segments from leaving the answer to one invocation at
 * a time; a segment counted fewer times than that simply has idle lanes.  A
 * span is a pair of 32-bit codes because the running totals that bound them
 * have already been read on the host, where an answer too long for one word is
 * refused before this walk starts.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Spans {
  // (start, end) per segment, in answer slots
  uvec2 data[];
}
uSpans;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // Segments, one per count
  int segments;
  // Answer slots, which is the total of the counts
  int total;
  // Segment-and-lane groups the grid covers, which is the stride a walk of more
  // segments than that has to step by
  int groups;
  int fill1;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Lanes serving one segment.  A workgroup of 256 covers eight segments, so the
// lanes of a segment stay together and their writes fall in order.
const int kLanesPerSegment = 32;

void main() {
  const int pos = int(gl_GlobalInvocationID.x);
  const int lane = pos % kLanesPerSegment;
  // The launch is bounded, so a walk of more segments than the grid has lanes
  // for visits the rest with the whole grid as its stride.
  for (int segment = pos / kLanesPerSegment; segment < uBlock.segments;
       segment += uBlock.groups) {
    const uvec2 span = uSpans.data[segment];
    // A segment counted zero times holds no slot, so its lanes all idle.
    for (uint at = span.x + uint(lane); at < span.y;
         at += uint(kLanesPerSegment)) {
      uOutput.data[at] = uint(segment);
    }
  }
}
