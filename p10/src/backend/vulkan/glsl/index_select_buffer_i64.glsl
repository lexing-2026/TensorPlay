#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer index_select over a payload of wide counts: a code wider
// than one word has no texel form and no lane to split across, so one element
// owns a whole pair of words and one invocation moves one element.  The
// gathered payload still reads as planes x rows x width, so output element e
// belongs to block e / (count * width), slot (e / width) % count and column
// e % width, and the Int32 code that slot names selects the row of e's own
// block.  A code naming no row contributes a zero pair, which is the same
// answer the narrower payloads give.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uvec2 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uvec2 data[];
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

void main() {
  const uint element = gl_GlobalInvocationID.x;
  const uint n =
      uint(uBlock.width) * uint(uBlock.count) * uint(uBlock.planes);
  if (element >= n) {
    return;
  }

  const uint width = uint(uBlock.width);
  const uint count = uint(uBlock.count);
  const uint block_row = element / width;
  const uint plane = block_row / count;
  const uint slot = block_row - plane * count;
  const uint column = element - block_row * width;

  const int code = uIndices.data[slot];
  if (code < 0 || code >= uBlock.rows) {
    uOutput.data[element] = uvec2(0u, 0u);
    return;
  }
  const uint source = (plane * uint(uBlock.rows) + uint(code)) * width + column;
  uOutput.data[element] = uInput.data[source];
}
