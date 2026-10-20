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
  int width; // elements a row holds: the axes below the gathered one
  int planes; // blocks above the gathered one
  int count; // rows selected out of each block
  int rows; // rows available in each block
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * index_select gather over a flattened contiguous payload that reads as
 * planes x rows x width: a 1-d selection, a selection of the rows of a
 * 2-d table, and the same selection repeated inside every block the axes
 * above the gathered one name.
 * Invocation (s, e, p): slot s inside the gathered row, index element e,
 * block p.  Each position reads its index id out of a flat Int32 texture
 * (one index per texel) and copies the source element of block p named by
 * that id, at (p * rows + id) * width + s, while it writes its own element
 * at (p * count + e) * width + s.  Both payloads ride flat 1-D textures
 * (x = flat element, one active lane per texel), so these addresses map
 * straight onto the x coordinate.  A code naming no row of its block
 * contributes a zero.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.width || pos.y >= uBlock.count ||
      pos.z >= uBlock.planes) {
    return;
  }

  const int destination =
      (pos.z * uBlock.count + pos.y) * uBlock.width + pos.x;
  const int idx = texelFetch(uIndices, ivec3(pos.y, 0, 0), 0).x;
  if (idx < 0 || idx >= uBlock.rows) {
    imageStore(uOutput, ivec3(destination, 0, 0), vec4(0.0));
    return;
  }

  const int source =
      (pos.z * uBlock.rows + idx) * uBlock.width + pos.x;
  const float v = texelFetch(uInput, ivec3(source, 0, 0), 0).x;
  imageStore(uOutput, ivec3(destination, 0, 0), vec4(v));
}
