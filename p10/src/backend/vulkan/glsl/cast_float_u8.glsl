#version 450 core
/*
 * Single precision to an unsigned byte.  The lanes wrap exactly as a host cast
 * does: truncate toward zero, then take the value modulo 256, which is the
 * byte a signed or an unsigned destination stores.  Storing an out-of-range
 * integer into a small-format image is undefined, so the wrap happens here.
 */
layout(set = 0, binding = 0, rgba8ui) uniform highp restrict writeonly uimage3D dst;
layout(set = 0, binding = 1, rgba32f) uniform highp restrict readonly image3D src;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  imageStore(dst, pos, uvec4(mod(trunc(imageLoad(src, pos)), vec4(256.0))));
}
