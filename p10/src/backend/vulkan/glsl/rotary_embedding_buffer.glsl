#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Rotary position embedding (interleaved pairs) over a linear payload.  A
 * sequence longer than the device's image axis limit has no texture form, and
 * the rotation itself is elementwise in the head dimension, so nothing here
 * needs the texel lanes a packed image would bring: the payload is read as the
 * row-major (..., tokens, head_dim) array it already is.
 *
 * One invocation rotates one pair of dimensions (2i, 2i+1):
 *   out[2i]   = in[2i]   * cos[t][i] - in[2i+1] * sin[t][i]
 *   out[2i+1] = in[2i]   * sin[t][i] + in[2i+1] * cos[t][i]
 * where t is the table row of the token, shifted by position_offset for the
 * rows a cache append adds.  A one-row table broadcasts to every token.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;
// clang-format off
$if TABLE_BUFFER == 1:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly CosBuffer {
    float data[];
  }
  uCos;
  layout(set = 0, binding = 3) buffer PRECISION restrict readonly SinBuffer {
    float data[];
  }
  uSin;
$else:
  layout(set = 0, binding = 2) uniform PRECISION sampler3D uCos;
  layout(set = 0, binding = 3) uniform PRECISION sampler3D uSin;
// clang-format on
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 sizes;        // (half_dim, tokens, batches, 0)
  int head_dim;       // elements per token along the rotated axis
  int table_rows;     // cosine/sine row count (1 = broadcast)
  int table_stride;   // element step to the next table row
  int position_offset;
  int fill0;
  int fill1;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.sizes.x || pos.y >= uBlock.sizes.y ||
      pos.z >= uBlock.sizes.z) {
    return;
  }

  const int pair = pos.x;
  const int token = pos.y;
  const int table_row =
      (uBlock.table_rows == 1) ? 0 : (uBlock.position_offset + token);

  // clang-format off
  $if TABLE_BUFFER == 1:
    const float c = uCos.data[table_row * uBlock.table_stride + pair];
    const float s = uSin.data[table_row * uBlock.table_stride + pair];
  $else:
    const float c = texelFetch(uCos, ivec3(pair, table_row, 0), 0).x;
    const float s = texelFetch(uSin, ivec3(pair, table_row, 0), 0).x;
  // clang-format on

  const int base =
      (pos.z * uBlock.sizes.y + token) * uBlock.head_dim + 2 * pair;
  const float in_even = uInput.data[base];
  const float in_odd = uInput.data[base + 1];
  uOutput.data[base] = in_even * c - in_odd * s;
  uOutput.data[base + 1] = in_even * s + in_odd * c;
}
