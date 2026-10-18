#version 450 core
/*
 * Single precision to a signed byte (the same texture carries Bool, so a Bool
 * destination is served elsewhere: its lanes must be 0/1, not the truncated
 * value).  The elementwise rule is to truncate toward zero and then wrap into
 * the byte range, matching what a host cast of an out-of-range value yields.
 * Storing an out-of-range integer into a small-format image is undefined, so
 * the wrap happens here instead of in the format conversion.
 */
layout(set = 0, binding = 0, rgba8i) uniform highp restrict writeonly iimage3D dst;
layout(set = 0, binding = 1, rgba32f) uniform highp restrict readonly image3D src;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

int wrap_to_int8(const float v) {
  const float remainder = mod(trunc(v), 256.0);
  return int(remainder >= 128.0 ? remainder - 256.0 : remainder);
}

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  const vec4 value = imageLoad(src, pos);
  imageStore(
      dst,
      pos,
      ivec4(
          wrap_to_int8(value.x),
          wrap_to_int8(value.y),
          wrap_to_int8(value.z),
          wrap_to_int8(value.w)));
}
