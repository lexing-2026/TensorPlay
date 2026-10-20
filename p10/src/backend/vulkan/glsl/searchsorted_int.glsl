#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * The insertion position of every query in a sorted sequence of codes.  The
 * walk is the halving search the single-precision payload runs too, with the
 * compares on the code width: an entry above two to the twenty-fourth means
 * nothing here can be rounded, which is why a payload counted in codes is
 * searched in codes.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  int data[];
} uOutput;

layout(set = 0, binding = 1) buffer PRECISION restrict readonly Boundaries {
  int data[];
} uBoundaries;

layout(set = 0, binding = 2) buffer PRECISION restrict readonly Queries {
  int data[];
} uQueries;

layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int queries;  // Queries, one per answer
  int row;      // Queries a row holds, which is how many share one table
  int span;     // Entries in one table
  int upper;    // Nonzero asks for the upper bound: the first entry above the query
  int one_d;    // Nonzero when one sequence answers every query
  int stride;   // Invocations the grid holds; a longer walk steps by this
  int fill0;
  int fill1;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  for (int pos = int(gl_GlobalInvocationID.x); pos < uBlock.queries;
       pos += uBlock.stride) {
    const int value = uQueries.data[pos];
    const int base = uBlock.one_d != 0
        ? 0
        : (pos / uBlock.row) * uBlock.span;
    int lo = base;
    int hi = base + uBlock.span;
    while (lo < hi) {
      const int mid = lo + ((hi - lo) >> 1);
      const int entry = uBoundaries.data[mid];
      const bool go_right =
          uBlock.upper != 0 ? !(entry > value) : !(entry >= value);
      if (go_right) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    uOutput.data[pos] = lo - base;
  }
}
