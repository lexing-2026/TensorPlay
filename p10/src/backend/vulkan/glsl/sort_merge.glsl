#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly Output { uvec2 data[]; } dst;
layout(set = 0, binding = 1) buffer restrict readonly Input { uvec2 data[]; } src;
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
void main() {
  int index = int(gl_GlobalInvocationID.x);
  int row = int(gl_GlobalInvocationID.y);
  if (index >= uBlock.axis_size || row >= uBlock.rows) return;
  int pair_start = index / (2 * uBlock.run) * (2 * uBlock.run);
  int split = min(pair_start + uBlock.run, uBlock.axis_size);
  int end = min(split + uBlock.run, uBlock.axis_size);
  bool left = index < split;
  int other_start = left ? split : pair_start;
  int lo = other_start;
  int hi = left ? end : split;
  int row_offset = row * uBlock.axis_size;
  uvec2 value = src.data[row_offset + index];
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (precedes(src.data[row_offset + mid], value)) lo = mid + 1;
    else hi = mid;
  }
  int rank = index - (left ? pair_start : split) + lo - other_start;
  dst.data[row_offset + pair_start + rank] = value;
}
