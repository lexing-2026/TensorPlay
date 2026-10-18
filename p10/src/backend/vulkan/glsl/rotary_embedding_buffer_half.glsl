#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Rotary position embedding (interleaved pairs) over a half-precision linear
 * payload.  Rotating a pair is exactly what one 32-bit word holds: the even
 * dimension is the low half, the odd one the high half, so one invocation reads
 * the word, rotates both lanes, and writes the word back whole.  No two
 * invocations ever touch the same word, which is what keeps the store free of
 * read-modify-write races.
 *
 * The head dimension has to be even for a pair to sit inside one word; the
 * dispatcher keeps any other layout off this path.
 *
 *   out[2i]   = in[2i]   * cos[t][i] - in[2i+1] * sin[t][i]
 *   out[2i+1] = in[2i]   * sin[t][i] + in[2i+1] * cos[t][i]
 *
 * where t is the table row of the token, shifted by position_offset for the rows
 * a cache append adds.  A one-row table broadcasts to every token.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
  uint data[];
}
uInput;
// clang-format off
$if TABLE_BUFFER == 1:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly CosBuffer {
    uint data[];
  }
  uCos;
  layout(set = 0, binding = 3) buffer PRECISION restrict readonly SinBuffer {
    uint data[];
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

// clang-format off
$if TABLE_BUFFER == 1:
  float table_lane(const in uint word, const in uint lane) {
    const vec2 packed = unpackHalf2x16(word);
    return lane == 0u ? packed.x : packed.y;
  }
// clang-format on

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
    const int table_word =
        (table_row * uBlock.table_stride + pair) >> 1;
    const uint table_lane_index =
        uint(table_row * uBlock.table_stride + pair) & 1u;
    const float c =
        table_lane(uCos.data[table_word], table_lane_index);
    const float s =
        table_lane(uSin.data[table_word], table_lane_index);
  $else:
    const float c = texelFetch(uCos, ivec3(pair, table_row, 0), 0).x;
    const float s = texelFetch(uSin, ivec3(pair, table_row, 0), 0).x;
  // clang-format on

  const int base =
      (pos.z * uBlock.sizes.y + token) * uBlock.head_dim + 2 * pair;
  const int word = base >> 1;
  const vec2 in_pair = unpackHalf2x16(uInput.data[word]);
  uOutput.data[word] = packHalf2x16(vec2(
      in_pair.x * c - in_pair.y * s,
      in_pair.x * s + in_pair.y * c));
}
