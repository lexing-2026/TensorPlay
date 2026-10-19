#version 450 core
/*
 * Signed word to half over texture storage.  The value widens to single
 * precision, which holds every small code exactly, and then quantizes to the
 * nearest half; the quantization is done in the shader because the store's own
 * rounding is the one the device chooses.
 */
#include "half_round.h"

layout(set = 0, binding = 0, rgba16f) uniform highp restrict writeonly image3D dst;
layout(set = 0, binding = 1, rgba32i) uniform highp restrict readonly iimage3D src;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  const vec4 value = vec4(imageLoad(src, pos));
  imageStore(
      dst,
      pos,
      vec4(
          round_to_half(value.x),
          round_to_half(value.y),
          round_to_half(value.z),
          round_to_half(value.w)));
}
