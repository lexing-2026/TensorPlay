#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer embedding lookup: the weight table, the index payload and
// the output are plain row-major buffer payloads (the shapes that exceed the
// 3-D image axis limit, e.g. a large vocabulary table).  Indices are Int32
// words uploaded to a dense buffer by the caller.  Output writes go through
// the flat element mapping so arbitrary row widths (not multiples of four)
// stay aligned to the logical element order.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Weight {
  vec4 data[];
}
uWeight;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int rows;        // index count: one output row per lookup
  int features;    // width of the weight table
  int weight_rows; // rows in the weight table
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint nf4 = (uint(uBlock.features) + 3u) >> 2u;
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= nf4 * uint(uBlock.rows)) {
    return;
  }
  const uint row = idx / nf4;
  const uint f4 = idx % nf4;

  const int row_id = uIndices.data[row];
  const int wbase = row_id * uBlock.features;
  const int obase = int(row) * uBlock.features;

  // Out-of-range lookups contribute zeros instead of reading outside the
  // weight table.
  const bool in_table = row_id >= 0 && row_id < uBlock.weight_rows;
  for (int i = 0; i < 4; ++i) {
    const int f = int(f4) * 4 + i;
    if (f >= uBlock.features) {
      break;
    }
    const int w = wbase + f;
    const int o = obase + f;
    uOutput.data[uint(o) >> 2u][uint(o) & 3u] =
        in_table ? uWeight.data[uint(w) >> 2u][uint(w) & 3u] : 0.0f;
  }
}