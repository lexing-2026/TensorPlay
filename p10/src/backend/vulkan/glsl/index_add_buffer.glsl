#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer in-place accumulation of whole rows along any axis of a
// row-major payload.  The destination flattens to planes x rows x width, where
// planes are the axes above the accumulated one and width the axes below it, so
// one invocation owns four feature lanes of one destination row and adds every
// source row the Int32 index payload names onto them; the same list applies to
// every plane.
//
// The walk runs from the destination rather than from the source, so a code the
// list repeats is added by the single invocation that owns the row it names, in
// the order the list states it.  No two invocations write one element, which is
// what lets the accumulation read and write the same buffer without a lock, and
// lanes that fall in one group read the list together, so its cost is paid once
// per group instead of once per lane.
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
  int planes;  // elements of the payload above the accumulated axis
  int rows;    // rows the destination offers along the accumulated axis
  int width;   // features per row: the axes below the accumulated one
  int n_idx;   // rows of the source and entries of the index payload
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint width = uint(uBlock.width);
  const uint nf4 = (width + 3u) >> 2u;
  const uint row_slots = uint(uBlock.rows) * nf4;
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= uint(uBlock.planes) * row_slots) {
    return;
  }

  const uint plane = idx / row_slots;
  const uint rem = idx - plane * row_slots;
  const uint row = rem / nf4;
  const uint first = (rem - row * nf4) * 4u;
  // A row whose feature count is not a multiple of four ends in a group that
  // holds fewer lanes than it addresses.
  const uint lanes = min(4u, width - first);

  const uint self_base =
      (plane * uint(uBlock.rows) + row) * width + first;
  float held[4];
  for (uint lane = 0u; lane < 4u; ++lane) {
    if (lane >= lanes) {
      break;
    }
    const uint element = self_base + lane;
    held[lane] = uSelf.data[element >> 2u][element & 3u];
  }

  for (int k = 0; k < uBlock.n_idx; ++k) {
    if (uIndices.data[k] != int(row)) {
      continue;
    }
    const uint source_base =
        (plane * uint(uBlock.n_idx) + uint(k)) * width + first;
    for (uint lane = 0u; lane < 4u; ++lane) {
      if (lane >= lanes) {
        break;
      }
      const uint element = source_base + lane;
      held[lane] += uSource.data[element >> 2u][element & 3u];
    }
  }

  for (uint lane = 0u; lane < 4u; ++lane) {
    if (lane >= lanes) {
      break;
    }
    const uint element = self_base + lane;
    uSelf.data[element >> 2u][element & 3u] = held[lane];
  }
}
