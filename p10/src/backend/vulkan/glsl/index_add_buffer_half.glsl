#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer in-place accumulation of whole rows along any axis of a
// row-major half payload.  Two lanes share one 32-bit word, so one invocation
// owns a whole word and every element of the pair it holds; that is what makes
// a read-modify-write safe here, since a row whose feature count is odd puts
// the last lane of one row beside the first of the next.
//
// The destination flattens to planes x rows x width, the walk runs from the
// destination rather than from the source, and the list is read in order, so
// the rows a repeated code names are added one after another by the single
// invocation that owns the word.  When both lanes fall in one row they share a
// pass over the list.
layout(set = 0, binding = 0) buffer PRECISION restrict Self {
  uint data[];
}
uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Source {
  uint data[];
}
uSource;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int planes;  // elements of the payload above the accumulated axis
  int rows;    // rows the destination offers along the accumulated axis
  int width;   // features per row: the axes below the accumulated one
  int n_idx;   // rows of the source and entries of the index payload
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// The 16-bit field holding one element of a dense half payload.
float half_field(uint word, uint lane) {
  return unpackHalf2x16(word)[lane];
}

float source_value(uint element) {
  return half_field(uSource.data[element >> 1u], element & 1u);
}

void main() {
  const uint width = uint(uBlock.width);
  const uint rows = uint(uBlock.rows);
  const uint block_elements = width * rows;
  const uint total = uint(uBlock.planes) * block_elements;
  const uint word = gl_GlobalInvocationID.x;
  if (word >= ((total + 1u) >> 1u)) {
    return;
  }

  const uint first = 2u * word;
  const bool holds_two = (first + 1u) < total;

  // The row and column each lane of the word belongs to.
  const uint block0 = first / width;
  const uint plane0 = block0 / rows;
  const uint row0 = block0 - plane0 * rows;
  const uint col0 = first - block0 * width;
  const uint block1 = (first + 1u) / width;
  const bool same_row = holds_two && (block1 == block0);

  vec2 held = unpackHalf2x16(uSelf.data[word]);
  for (uint k = 0u; k < uint(uBlock.n_idx); ++k) {
    const int code = uIndices.data[k];
    if (code < 0 || uint(code) != row0) {
      continue;
    }
    const uint line = (plane0 * uint(uBlock.n_idx) + k) * width;
    held[0] += source_value(line + col0);
    if (same_row) {
      held[1] += source_value(line + col0 + 1u);
    }
  }

  if (holds_two && !same_row) {
    const uint plane1 = block1 / rows;
    const uint row1 = block1 - plane1 * rows;
    const uint col1 = (first + 1u) - block1 * width;
    for (uint k = 0u; k < uint(uBlock.n_idx); ++k) {
      const int code = uIndices.data[k];
      if (code < 0 || uint(code) != row1) {
        continue;
      }
      held[1] += source_value((plane1 * uint(uBlock.n_idx) + k) * width + col1);
    }
  }

  uSelf.data[word] = packHalf2x16(held);
}
