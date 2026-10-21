#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer index_select over a payload of counts: input, index payload
// and output are plain row-major buffers, one signed word per element.  The
// gathered payload reads as planes x rows x width, the planes being the blocks
// the axes above the gathered one name, so output element e belongs to block
// e / (count * width), slot (e / width) % count and column e % width, and the
// code that slot names selects the row of e's own block.  Flat element
// addressing keeps an inner width that is not a multiple of four from splitting
// a code across two words.  A code naming no row contributes a zero.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  ivec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  ivec4 data[];
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

int gathered(uint element) {
  const uint width = uint(uBlock.width);
  const uint count = uint(uBlock.count);
  const uint block_row = element / width;
  const uint plane = block_row / count;
  const uint slot = block_row - plane * count;
  const uint column = element - block_row * width;
  const int code = uIndices.data[slot];
  if (code < 0 || code >= uBlock.rows) {
    return 0;
  }
  const uint source =
      (plane * uint(uBlock.rows) + uint(code)) * width + column;
  return uInput.data[source >> 2u][source & 3u];
}

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  const uint n =
      uint(uBlock.width) * uint(uBlock.count) * uint(uBlock.planes);
  const uint nvec = n >> 2u;
  const uint rem = n & 3u;

  if (idx < nvec) {
    const uint base = idx << 2u;
    ivec4 res;
    for (int i = 0; i < 4; ++i) {
      res[i] = gathered(base + uint(i));
    }
    uOutput.data[idx] = res;
  } else if (idx == nvec) {
    // One invocation owns the whole partial word, so the dispatch stays at
    // div_up(n, 4) groups and tail coverage does not depend on how far the
    // workgroup count is rounded up.
    for (uint i = 0u; i < rem; ++i) {
      uOutput.data[nvec][i] = gathered((nvec << 2u) + i);
    }
  }
}
