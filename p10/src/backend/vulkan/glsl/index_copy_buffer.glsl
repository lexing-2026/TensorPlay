#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer in-place scatter of whole rows along the sequence axis of
// a row-major [N, C, S, W] payload (the shapes that exceed the 3-D image axis
// limit).  One invocation copies four feature lanes from a source row into
// the destination row selected by the Int32 index payload; the same index
// array applies to every N*C plane.  The destination is only written, never
// read, so the write remains the sole access to it besides the source fetch.
layout(set = 0, binding = 0) buffer PRECISION restrict Self {
  vec4 data[];
}
uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Source {
  vec4 data[];
}
uSource;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int planes;  // N * C
  int n_idx; // rows of the source and entries of the payload
  int width; // W: features per row
  int rows;  // S: rows available in the destination
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint nf4 = (uint(uBlock.width) + 3u) >> 2u;
  const uint idx = gl_GlobalInvocationID.x;
  const uint total = uint(uBlock.planes) * uint(uBlock.n_idx) * nf4;
  if (idx >= total) {
    return;
  }

  const uint plane = idx / (uint(uBlock.n_idx) * nf4);
  const uint rem = idx % (uint(uBlock.n_idx) * nf4);
  const uint r = rem / nf4;
  const uint f4 = rem % nf4;

  const int dest_row = uIndices.data[r];
  // Same rule as the texel variant: a code naming no destination row writes
  // nothing, so an out-of-range scatter never touches memory it does not own.
  if (dest_row < 0 || dest_row >= uBlock.rows) {
    return;
  }

  const int src_base =
      int(plane * uint(uBlock.n_idx) + r) * uBlock.width + int(f4) * 4;
  const int dst_base =
      int(plane * uint(uBlock.rows) + uint(dest_row)) * uBlock.width +
      int(f4) * 4;

  for (int i = 0; i < 4; ++i) {
    const int f = int(f4) * 4 + i;
    if (f >= uBlock.width) {
      break;
    }
    const int s = src_base + i;
    const int d = dst_base + i;
    uSelf.data[uint(d) >> 2u][uint(d) & 3u] =
        uSource.data[uint(s) >> 2u][uint(s) & 3u];
  }
}