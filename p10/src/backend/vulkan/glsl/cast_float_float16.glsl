#version 450 core
/*
 * Single precision to half over texture storage.  The destination lane is
 * narrower than the source, so the value is quantized to the nearest half
 * before the store; see the header for why the rounding cannot be left to the
 * image format conversion.
 */
#include "half_round.h"

layout(set = 0, binding = 0, rgba16f) uniform highp restrict writeonly image3D dst;
layout(set = 0, binding = 1, rgba32f) uniform highp restrict readonly image3D src;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  const vec4 value = imageLoad(src, pos);
  imageStore(
      dst,
      pos,
      vec4(
          round_to_half(value.x),
          round_to_half(value.y),
          round_to_half(value.z),
          round_to_half(value.w)));
}
