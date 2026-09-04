#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int rows; // index count: one output row per lookup
  int features; // width of the weight table
  int weight_rows;
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Embedding lookup over a 2-D [rows, features] weight with a 1-D index
 * payload.  Both tensors share the backend's 2-D texture geometry (width =
 * feature columns, height = row index, one active lane per texel; the
 * channel axis degenerates to one texel deep).  The index payload rides as
 * a flat Int32 texture (one index per texel), uploaded by the caller after
 * narrowing the Int64 codes on the host.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.features || pos.y >= uBlock.rows) {
    return;
  }

  const int row_id = texelFetch(uIndices, ivec3(pos.y, 0, 0), 0).x;

  const float v = texelFetch(uWeight, ivec3(pos.x, row_id, 0), 0).x;
  imageStore(uOutput, pos, vec4(v));
}
