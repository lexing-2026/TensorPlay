#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly Output { uvec2 data[]; } dst;
layout(set = 0, binding = 1) buffer restrict readonly Input { uint data[]; } src;
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
 * Tile sort over a half-precision contiguous payload.  Two lanes share one
 * 32-bit word, so an element is read by unpacking the word that holds it.
 * Every lane enters the shared tile as its exact single-precision widening,
 * which lets the comparison, the merge passes and the final gather stay the
 * float-wide walks they already are; the value is narrowed again only when it
 * reaches its destination.
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
      value = uvec2(
          floatBitsToUint(unpackHalf2x16(src.data[lin >> 1])[lin & 1]),
          uint(index));
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
