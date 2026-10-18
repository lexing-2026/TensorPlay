#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Rotary position embedding (interleaved pairs) over a channel-packed float
// texture.  The input is [..., seq, head_dim] with head_dim = 2 * half_dim;
// the texture stores one head dimension per x position and packs four heads
// per texel along z, so texel (x, row, batch) holds dimension x of four
// heads.  cos/sin tables are [table_rows, half_dim], one value per texel, in
// texture or linear storage.  Each invocation rotates one pair of dimensions
// (2i, 2i+1) for the four head lanes:
//   out[2i]   = in[2i]   * cos[t][i] - in[2i+1] * sin[t][i]
//   out[2i+1] = in[2i]   * sin[t][i] + in[2i+1] * cos[t][i]
// where t = row % seq_len (+ position_offset for KV cache increments).
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
// clang-format off
$if TABLE_BUFFER == 1:
  // A cached table longer than the image axis limit has no texture form; read
  // it as the row-major [positions, half_dim] array it is.
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
  ivec4 extents;     // (half_dim, rows, batches, 0)
  int tokens;        // seq_len (the second-to-last logical dim)
  int half_dim;      // head_dim / 2
  int table_rows;    // cos/sin row count (1 = broadcast)
  int position_offset;
  int table_stride;  // element step to the next linear table row
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.extents.x || pos.y >= uBlock.extents.y ||
      pos.z >= uBlock.extents.z) {
    return;
  }

  const int pair = pos.x;
  const int token = pos.y % uBlock.tokens;
  const int table_row =
      (uBlock.table_rows == 1) ? 0 : (uBlock.position_offset + token);

  // cos/sin for this rotation pair.  A texture table holds one value per texel
  // x position in its .x lane; a linear table is the flat [rows, half_dim]
  // array.
  // clang-format off
  $if TABLE_BUFFER == 1:
    const float c = uCos.data[table_row * uBlock.table_stride + pair];
    const float s = uSin.data[table_row * uBlock.table_stride + pair];
  $else:
    const float c = texelFetch(uCos, ivec3(pair, table_row, 0), 0).x;
    const float s = texelFetch(uSin, ivec3(pair, table_row, 0), 0).x;
  // clang-format on

  // Four head lanes share the same rotation.
  const vec4 in_even = texelFetch(uInput, ivec3(2 * pair, pos.y, pos.z), 0);
  const vec4 in_odd = texelFetch(uInput, ivec3(2 * pair + 1, pos.y, pos.z), 0);
  imageStore(uOutput, ivec3(2 * pair, pos.y, pos.z), in_even * c - in_odd * s);
  imageStore(uOutput, ivec3(2 * pair + 1, pos.y, pos.z), in_even * s + in_odd * c);
}