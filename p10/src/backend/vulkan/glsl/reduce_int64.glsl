#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly Output { uvec2 data[]; } dst;
$if DTYPE == "float":
  layout(set = 0, binding = 1) uniform highp sampler3D src;
$else:
  layout(set = 0, binding = 1) uniform highp isampler3D src;
layout(set = 0, binding = 2) uniform highp restrict Block {
  ivec4 sizes;
  ivec4 output_sizes;
  ivec4 reduced;
  ivec4 counts;
} uBlock;

uvec2 combine(uvec2 a, uvec2 b) {
  $if PRODUCT:
    uint hi, lo;
    umulExtended(a.x, b.x, hi, lo);
    return uvec2(lo, hi + a.x * b.y + a.y * b.x);
  $else:
    uint lo = a.x + b.x;
    return uvec2(lo, a.y + b.y + uint(lo < a.x));
}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
shared uvec2 partial[64];
void main() {
  int index = int(gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x);
  if (index >= uBlock.counts.x) return;
  int lane = int(gl_LocalInvocationID.x);
  ivec4 coord;
  int linear = index;
  for (int d = 0; d < 4; ++d) {
    coord[d] = linear % uBlock.output_sizes[d];
    linear /= uBlock.output_sizes[d];
  }
  $if PRODUCT:
    uvec2 accum = uvec2(1, 0);
  $else:
    uvec2 accum = uvec2(0);
  for (int r = lane; r < uBlock.counts.y; r += 64) {
    ivec4 p = coord;
    int k = r;
    for (int d = 0; d < 4; ++d) {
      if (uBlock.reduced[d] != 0) {
        p[d] = k % uBlock.sizes[d];
        k /= uBlock.sizes[d];
      }
    }
    ivec3 pos = ivec3(p.x, p.y, p.w * ((uBlock.sizes.z + 3) / 4) + p.z / 4);
    $if PRODUCT:
      int v = texelFetch(src, pos, 0)[p.z % 4];
      uvec2 value = uvec2(uint(v), v < 0 ? 0xffffffffu : 0u);
    $else:
      uvec2 value = uvec2(uint(texelFetch(src, pos, 0)[p.z % 4] != 0), 0);
    accum = combine(accum, value);
  }
  partial[lane] = accum;
  barrier();
  for (int stride = 32; stride > 0; stride /= 2) {
    if (lane < stride) partial[lane] = combine(partial[lane], partial[lane + stride]);
    barrier();
  }
  if (lane == 0) dst.data[index] = partial[0];
}
