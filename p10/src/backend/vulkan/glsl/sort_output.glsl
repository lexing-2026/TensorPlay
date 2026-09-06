#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0, rgba32f) uniform highp restrict writeonly image3D values;
layout(set = 0, binding = 1, rgba32i) uniform highp restrict writeonly iimage3D indices;
layout(set = 0, binding = 2) buffer restrict readonly Input { uvec2 data[]; } src;
layout(set = 0, binding = 3) uniform highp restrict Block {
  ivec4 in_sizes;
  ivec4 out_sizes;
  int axis_size;
  int inner;
  int rows;
  int tiles;
  int descending;
  int run;
  int count;
  int start;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  int depth = (uBlock.out_sizes.z + 3) / 4;
  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * depth) return;
  int n = pos.z / depth;
  int c4 = pos.z % depth;
  vec4 v = vec4(0);
  ivec4 ix = ivec4(0);
  for (int lane = 0; lane < 4 && c4 * 4 + lane < uBlock.out_sizes.z; ++lane) {
    int c = c4 * 4 + lane;
    int lin = ((n * uBlock.out_sizes.z + c) * uBlock.out_sizes.y +
        pos.y) * uBlock.out_sizes.x + pos.x;
    int row = lin / (uBlock.count * uBlock.inner) * uBlock.inner +
        lin % uBlock.inner;
    int index = (lin / uBlock.inner) % uBlock.count + uBlock.start;
    uvec2 value = src.data[row * uBlock.axis_size + index];
    v[lane] = uintBitsToFloat(value.x);
    ix[lane] = int(value.y);
  }
  imageStore(values, pos, v);
  imageStore(indices, pos, ix);
}
