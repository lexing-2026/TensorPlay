#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer in-place scatter of whole rows along any axis of a row-major
// half payload.  Two lanes share one 32-bit word, so one invocation owns a whole
// word and both of the elements it holds: a row whose feature count is odd puts
// the last lane of that row beside the first of the next, and a lane written
// inside a word the walk does not own would carry off the value the scatter
// never named.  The walk therefore runs from the destination, one word per
// invocation, and reads the list in order, so a code that names a row twice
// leaves the row its last mention holds.
layout(set = 0, binding = 0) buffer PRECISION Self {
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
  int planes;  // elements of the payload above the scattered axis
  int n_idx;   // rows of the source and entries of the index payload
  int width;   // features per row: the axes below the scattered one
  int rows;    // rows the destination offers along the scattered axis
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

// The value the last code naming row holds at col of the source, if any code
// names it at all.  A code outside the destination rows names nothing, so it
// leaves the lane as the payload holds it.
bool scattered_value(uint plane, uint row, uint col, out float value) {
  const uint width = uint(uBlock.width);
  const uint n_idx = uint(uBlock.n_idx);
  bool named = false;
  for (uint k = 0u; k < n_idx; ++k) {
    const int code = uIndices.data[k];
    if (code < 0 || uint(code) >= uint(uBlock.rows) || uint(code) != row) {
      continue;
    }
    value = source_value((plane * n_idx + k) * width + col);
    named = true;
  }
  return named;
}

void main() {
  const uint width = uint(uBlock.width);
  const uint rows = uint(uBlock.rows);
  const uint total = uint(uBlock.planes) * rows * width;
  const uint word = gl_GlobalInvocationID.x;
  if (word >= ((total + 1u) >> 1u)) {
    return;
  }

  const uint first = 2u * word;

  // The plane, row and column each lane of the word belongs to.  Both lanes
  // usually fall in one row, but an odd feature count splits them across the
  // boundary, and then each follows its own row's codes.
  const uint block0 = first / width;
  const uint plane0 = block0 / rows;
  const uint row0 = block0 - plane0 * rows;
  const uint col0 = first - block0 * width;

  vec2 held = unpackHalf2x16(uSelf.data[word]);
  float value;
  if (scattered_value(plane0, row0, col0, value)) {
    held[0] = value;
  }

  if ((first + 1u) < total) {
    const uint block1 = (first + 1u) / width;
    const uint plane1 = block1 / rows;
    const uint row1 = block1 - plane1 * rows;
    const uint col1 = (first + 1u) - block1 * width;
    if (scattered_value(plane1, row1, col1, value)) {
      held[1] = value;
    }
  }

  uSelf.data[word] = packHalf2x16(held);
}
