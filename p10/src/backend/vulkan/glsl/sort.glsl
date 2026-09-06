#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly Output { uvec2 data[]; } dst;
layout(set = 0, binding = 1) uniform highp sampler3D src;
layout(set = 0, binding = 2) uniform highp restrict Block {
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

bool precedes(uvec2 a, uvec2 b) {
  if (a.y == 0xffffffffu) return false;
  if (b.y == 0xffffffffu) return true;
  float av = uintBitsToFloat(a.x);
  float bv = uintBitsToFloat(b.x);
  bool an = isnan(av);
  bool bn = isnan(bv);
  if (an != bn) return uBlock.descending != 0 ? an : bn;
  if (av != bv && !an) return uBlock.descending != 0 ? av > bv : av < bv;
  return a.y < b.y;
}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
shared uvec2 entries[256];

void main() {
  int row = int(gl_WorkGroupID.y);
  int tile = int(gl_WorkGroupID.x);
  int lane = int(gl_LocalInvocationID.x);
  int base = tile * 256;
  for (int i = lane; i < 256; i += 64) {
    int index = base + i;
    uvec2 value = uvec2(0, 0xffffffffu);
    if (index < uBlock.axis_size) {
      int lin = (row / uBlock.inner * uBlock.axis_size + index) *
          uBlock.inner + row % uBlock.inner;
      int x = lin % uBlock.in_sizes.x;
      lin /= uBlock.in_sizes.x;
      int y = lin % uBlock.in_sizes.y;
      lin /= uBlock.in_sizes.y;
      int c = lin % uBlock.in_sizes.z;
      int n = lin / uBlock.in_sizes.z;
      float v = texelFetch(src, ivec3(x, y,
          n * ((uBlock.in_sizes.z + 3) / 4) + c / 4), 0)[c % 4];
      value = uvec2(floatBitsToUint(v), uint(index));
    }
    entries[i] = value;
  }
  barrier();
  int tile_size = 1;
  while (tile_size < min(256, uBlock.axis_size - base)) tile_size *= 2;
  for (int width = 2; width <= tile_size; width *= 2) {
    for (int stride = width / 2; stride > 0; stride /= 2) {
      for (int i = lane; i < tile_size; i += 64) {
        int partner = i ^ stride;
        if (partner > i) {
          uvec2 a = entries[i];
          uvec2 b = entries[partner];
          bool swap_pair = (i & width) == 0 ? precedes(b, a) : precedes(a, b);
          if (swap_pair) { entries[i] = b; entries[partner] = a; }
        }
      }
      barrier();
    }
  }
  for (int i = lane; i < 256 && base + i < uBlock.axis_size; i += 64) {
    dst.data[row * uBlock.axis_size + base + i] = entries[i];
  }
}
