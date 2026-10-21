#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, rgba32i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;
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
 * index_select gather over a flattened contiguous payload that counts rather
 * than measures: a row holds whole codes, so one code per signed texel in its
 * active lane.  The walk is the float one with a word at every payload; a code
 * naming no row of its block is answered by a zero of the same width.
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
    imageStore(uOutput, ivec3(destination, 0, 0), ivec4(0, 0, 0, 0));
    return;
  }

  const int source = (pos.z * uBlock.rows + idx) * uBlock.width + pos.x;
  const int code = texelFetch(uInput, ivec3(source, 0, 0), 0).x;
  imageStore(uOutput, ivec3(destination, 0, 0), ivec4(code, 0, 0, 0));
}
