#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer index_select over half payloads: input and output are dense
// row-major buffers packing two lanes into each 32-bit word, the index list a
// buffer of Int32 words.  The gathered payload reads as planes x rows x width,
// the planes being the blocks the axes above the gathered one name, so output
// element e belongs to block e / (count * width), slot (e / width) % count and
// column e % width, and the code that slot names selects the row of e's own
// block.  Flat element addressing keeps an inner width that is not a multiple
// of two from sliding a value into a later row's wrong lane.  A code naming no
// row contributes a zero lane.
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
  int width; // elements a row holds: the axes below the gathered one
  int planes; // blocks above the gathered one
  int count; // rows selected out of each block
  int rows; // rows available in each block
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// The 16-bit field holding one element of a dense half payload.
uint half_field(uint element) {
  return (uInput.data[element >> 1u] >> ((element & 1u) * 16u)) & 0xffffu;
}

uint gathered_field(uint element) {
  const uint width = uint(uBlock.width);
  const uint count = uint(uBlock.count);
  const uint block_row = element / width;
  const uint plane = block_row / count;
  const uint slot = block_row - plane * count;
  const uint column = element - block_row * width;
  const int code = uIndices.data[slot];
  if (code < 0 || code >= uBlock.rows) {
    return 0u;
  }
  const uint source =
      (plane * uint(uBlock.rows) + uint(code)) * width + column;
  return half_field(source);
}

void main() {
  const uint total =
      uint(uBlock.width) * uint(uBlock.count) * uint(uBlock.planes);
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
    packed |= gathered_field(element) << (16u * lane);
  }
  uOutput.data[word] = packed;
}
