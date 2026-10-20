#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Mark pass of a row fill: one invocation per entry of the code list records
// that the row its code names is named.  A code given from the end names the
// row that many from it, so a negative one is read as an offset from the far
// side; a code outside the rows the destination offers names no row and marks
// nothing, which is the answer a row lookup already gives to a request beyond
// its rows.
//
// The mark is one word per row and every entry that names a row writes the same
// one into it, so a repeated code costs nothing beyond the first and no
// invocation can hand another's mark back as stale.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Marks {
  int data[];
}
uMarks;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int n_idx; // entries of the code list
  int rows;  // entries the destination offers along the axis
  int fill;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int k = int(gl_GlobalInvocationID.x);
  if (k >= uBlock.n_idx) {
    return;
  }
  int code = uIndices.data[k];
  if (code < 0) {
    code += uBlock.rows;
  }
  if (code < 0 || code >= uBlock.rows) {
    return;
  }
  uMarks.data[code] = 1;
}
