#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly Output { uvec2 data[]; } dst;
layout(set = 0, binding = 1) buffer restrict readonly Input { float data[]; } src;
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
shared uvec2 entries[2048];

/*
 * Tile sort over buffer (contiguous) storage, for shapes whose extents
 * exceed the 3-D image axis limit.  One workgroup sorts a 2048-element tile
 * of one sort row: strided loops cover every element exactly once with 256
 * lanes, the bitonic network follows the same direction logic as the
 * texture variant, and missing tail elements fill as padding that always
 * compares last.  A (row, axis position) pair maps to the contiguous
 * buffer position (row / inner) * axis_size * inner + index * inner +
 * row % inner, matching the row/outer/inner split of the gather pass.
 */
void main() {
  int row = int(gl_WorkGroupID.y);
  int tile = int(gl_WorkGroupID.x);
  int lane = int(gl_LocalInvocationID.x);
  int base = tile * 2048;
  for (int i = lane; i < 2048; i += 256) {
    int index = base + i;
    uvec2 value = uvec2(0, 0xffffffffu);
    if (index < uBlock.axis_size) {
      int lin = row / uBlock.inner * (uBlock.axis_size * uBlock.inner) +
          index * uBlock.inner + row % uBlock.inner;
      value = uvec2(floatBitsToUint(src.data[lin]), uint(index));
    }
    entries[i] = value;
  }
  barrier();
  int tile_size = 1;
  while (tile_size < min(2048, uBlock.axis_size - base)) tile_size *= 2;
  for (int width = 2; width <= tile_size; width *= 2) {
    for (int stride = width / 2; stride > 0; stride /= 2) {
      for (int i = lane; i < tile_size; i += 256) {
        int partner = i ^ stride;
        if (partner > i) {
          uvec2 a = entries[i];
          uvec2 b = entries[partner];
          bool swap_pair = (i & width) == 0 ? precedes(b, a) : precedes(a, b);
          if (swap_pair) {
            entries[i] = b;
            entries[partner] = a;
          }
        }
      }
      barrier();
    }
  }
  for (int i = lane; i < 2048 && base + i < uBlock.axis_size; i += 256) {
    dst.data[row * uBlock.axis_size + base + i] = entries[i];
  }
}
