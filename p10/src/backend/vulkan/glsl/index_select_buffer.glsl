#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer index_select: input, index payload and output are plain
// row-major buffer payloads (the shapes that exceed the 3-D image axis
// limit).  The gather walks a flattened payload with one outer block: each
// output element reads row = indices[o / inner], column = o % inner, through
// flat element addressing so arbitrary inner widths stay aligned.  Out-of-
// range lookups contribute zeros instead of reading outside the input.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  vec4 data[];
}
uInput;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int inner;      // width of the inner span
  int row_stride;  // stride between selected rows (== inner)
  int count;       // number of selected rows
  int rows;       // rows available in the input
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  const uint n = uint(uBlock.inner) * uint(uBlock.count);
  const uint nvec = n >> 2u;
  const uint rem = n & 3u;

  if (idx < nvec) {
    const uint base = idx << 2u;
    vec4 res;
    for (int i = 0; i < 4; ++i) {
      const uint o = base + uint(i);
      const uint row = o / uint(uBlock.inner);
      const uint col = o % uint(uBlock.inner);
      const int row_id = uIndices.data[row];
      if (row_id < 0 || row_id >= uBlock.rows) {
        res[i] = 0.0f;
        continue;
      }
      const int src = row_id * uBlock.row_stride + int(col);
      res[i] = uInput.data[uint(src) >> 2u][uint(src) & 3u];
    }
    uOutput.data[idx] = res;
  } else if (idx == nvec) {
    // One invocation owns the whole partial word, so the dispatch stays at
    // div_up(n, 4) groups and tail coverage does not depend on how far the
    // workgroup count is rounded up.
    const uint j = nvec;
    for (uint i = 0u; i < rem; ++i) {
      const uint o = (nvec << 2u) + i;
      const uint row = o / uint(uBlock.inner);
      const uint col = o % uint(uBlock.inner);
      const int row_id = uIndices.data[row];
      if (row_id < 0 || row_id >= uBlock.rows) {
        uOutput.data[j][i] = 0.0f;
        continue;
      }
      const int src = row_id * uBlock.row_stride + int(col);
      uOutput.data[j][i] = uInput.data[uint(src) >> 2u][uint(src) & 3u];
    }
  }
}