#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Rotary position embedding (interleaved pairs) over a width-packed float
// texture.  The input is [..., seq, head_dim] with head_dim = 2 * half_dim;
// texel (x, row, batch) holds dimensions 4x..4x+3 of that row, i.e. two
// rotation pairs.  cos/sin tables are [table_rows, half_dim] (also
// width-packed, four pairs per texel).  Each invocation handles one cos
// texel (four rotation pairs = two output texels):
//   out[2i]   = in[2i]   * cos[t][i] - in[2i+1] * sin[t][i]
//   out[2i+1] = in[2i]   * sin[t][i] + in[2i+1] * cos[t][i]
// where t = row % seq_len (+ position_offset for KV cache increments).
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uCos;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uSin;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 extents;     // (head_dim/4, rows, batches, 0)
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

  const int token = pos.y % uBlock.tokens;
  const int table_row =
      (uBlock.table_rows == 1) ? 0 : (uBlock.position_offset + token);

  // Four rotation pairs for this invocation.
  const vec4 c = texelFetch(uCos, ivec3(pos.x, table_row, 0), 0);
  const vec4 s = texelFetch(uSin, ivec3(pos.x, table_row, 0), 0);

  // Output texel 0: dimensions 4*(2x) .. 4*(2x)+3 = pairs 4x, 4x+1.
  const vec4 in0 = texelFetch(uInput, ivec3(2 * pos.x, pos.y, pos.z), 0);
  vec4 out0;
  out0.x = in0.x * c.x - in0.y * s.x;
  out0.y = in0.x * s.x + in0.y * c.x;
  out0.z = in0.z * c.y - in0.w * s.y;
  out0.w = in0.z * s.y + in0.w * c.y;

  // Output texel 1: dimensions 4*(2x+1) .. +3 = pairs 4x+2, 4x+3.
  const vec4 in1 = texelFetch(uInput, ivec3(2 * pos.x + 1, pos.y, pos.z), 0);
  vec4 out1;
  out1.x = in1.x * c.z - in1.y * s.z;
  out1.y = in1.x * s.z + in1.y * c.z;
  out1.z = in1.z * c.w - in1.w * s.w;
  out1.w = in1.z * s.w + in1.w * c.w;

  imageStore(uOutput, ivec3(2 * pos.x, pos.y, pos.z), out0);
  imageStore(uOutput, ivec3(2 * pos.x + 1, pos.y, pos.z), out1);
}