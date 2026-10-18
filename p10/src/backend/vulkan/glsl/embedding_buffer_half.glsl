#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer embedding lookup over half payloads: table, index list and
// output are dense row-major buffers, the table and output packing two lanes
// into each 32-bit word.  Rows are walked through flat element addressing, so
// a feature count that is not a multiple of two shifts every following row by
// half a word and the lane a value sits in still lands where the logical order
// puts it.  An out-of-range lookup contributes a zero lane.
//
// One invocation owns one whole output word: it covers two output elements,
// finds each one's row and column, and moves the 16-bit fields without ever
// widening them.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Weight {
  uint data[];
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

// The 16-bit field holding one element of a dense half payload.
uint half_field(uint element) {
  return (uWeight.data[element >> 1u] >> ((element & 1u) * 16u)) & 0xffffu;
}

void main() {
  const uint features = uint(uBlock.features);
  const uint total = uint(uBlock.rows) * features;
  const uint word = gl_GlobalInvocationID.x;
  if (word >= ((total + 1u) >> 1u)) {
    return;
  }

  uint packed = 0u;
  for (uint lane = 0u; lane < 2u; ++lane) {
    const uint element = (word << 1u) + lane;
    if (element >= total) {
      break;
    }
    const uint row = element / features;
    const uint column = element % features;
    const int row_id = uIndices.data[row];
    if (row_id < 0 || row_id >= uBlock.weight_rows) {
      continue;
    }
    const uint source = uint(row_id) * features + column;
    packed |= half_field(source) << (16u * lane);
  }
  uOutput.data[word] = packed;
}
