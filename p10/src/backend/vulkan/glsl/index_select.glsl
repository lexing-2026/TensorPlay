#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int inner; // elements gathered per index element
  int row_stride; // flat-layout distance between consecutive gathered rows
  int count; // index count
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * index_select gather over a flattened contiguous payload with a single
 * outer block (1-d selections, and 2-d selections along the row axis).
 * Invocation (e, s): index element e, slot s inside the gathered row.
 * Each position reads its index id out of a flat Int32 texture (one
 * index per texel) and copies the source element at idx * row_stride + s,
 * where row_stride is the distance between consecutive gathered rows in
 * the flat layout.  Both payloads ride flat 1-D textures (x = flat
 * element, one active lane per texel), so the addresses map straight onto
 * the x coordinate.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.inner || pos.y >= uBlock.count) {
    return;
  }

  const int e = pos.y;
  const int idx = texelFetch(uIndices, ivec3(e, 0, 0), 0).x;

  const int src_flat = idx * uBlock.row_stride + pos.x;
  const float v = texelFetch(uInput, ivec3(src_flat, 0, 0), 0).x;
  imageStore(uOutput, ivec3(e * uBlock.inner + pos.x, 0, 0), vec4(v));
}
