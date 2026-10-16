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
// heads.  cos/sin tables are [table_rows, half_dim], width-packed (four
// rotation pairs per texel).  Each invocation rotates one pair of dimensions
// (2i, 2i+1) for the four head lanes:
//   out[2i]   = in[2i]   * cos[t][i] - in[2i+1] * sin[t][i]
//   out[2i+1] = in[2i]   * sin[t][i] + in[2i+1] * cos[t][i]
// where t = row % seq_len (+ position_offset for KV cache increments).
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uCos;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uSin;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 extents;     // (half_dim, rows, batches, 0)
  int tokens;        // seq_len (the second-to-last logical dim)
  int half_dim;      // head_dim / 2
  int table_rows;    // cos/sin row count (1 = broadcast)
  int position_offset;
  int fill0;
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

  // cos/sin for this rotation pair.  The tables are ordinary [rows, half]
  // textures (one value per texel x position in the .x lane).
  const float c = texelFetch(uCos, ivec3(pair, table_row, 0), 0).x;
  const float s = texelFetch(uSin, ivec3(pair, table_row, 0), 0).x;

  // Four head lanes share the same rotation.
  const vec4 in_even = texelFetch(uInput, ivec3(2 * pair, pos.y, pos.z), 0);
  const vec4 in_odd = texelFetch(uInput, ivec3(2 * pair + 1, pos.y, pos.z), 0);
  imageStore(uOutput, ivec3(2 * pair, pos.y, pos.z), in_even * c - in_odd * s);
  imageStore(uOutput, ivec3(2 * pair + 1, pos.y, pos.z), in_even * s + in_odd * c);
}