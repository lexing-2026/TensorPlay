#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer index_select over half payloads: input and output are dense
// row-major buffers packing two lanes into each 32-bit word, the index list a
// buffer of Int32 words.  The gather walks a flattened payload through flat
// element addressing - output element e belongs to row e / inner and column
// e % inner - so an inner width that is not a multiple of two cannot slide a
// value into the wrong lane of a later row.  An out-of-range lookup contributes
// a zero lane.
//
// One invocation owns one whole output word: it covers two output elements and
// moves their 16-bit fields without widening them.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int inner;       // width of the inner span
  int row_stride;  // stride between selected rows (== inner)
  int count;       // number of selected rows
  int rows;        // rows available in the input
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// The 16-bit field holding one element of a dense half payload.
uint half_field(uint element) {
  return (uInput.data[element >> 1u] >> ((element & 1u) * 16u)) & 0xffffu;
}

void main() {
  const uint inner = uint(uBlock.inner);
  const uint total = inner * uint(uBlock.count);
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
    const uint row = element / inner;
    const uint column = element % inner;
    const int row_id = uIndices.data[row];
    if (row_id < 0 || row_id >= uBlock.rows) {
      continue;
    }
    const uint source = uint(row_id) * uint(uBlock.row_stride) + column;
    packed |= half_field(source) << (16u * lane);
  }
  uOutput.data[word] = packed;
}
