#version 450 core
/*
 * Single precision to Bool.  The Bool payload shares the signed-byte texture
 * with Int8, but its lanes are limited to 0 and 1: a value converts to one
 * exactly when it is not zero, which is what the host cast of a float to a
 * bool does as well.
 */
layout(set = 0, binding = 0, rgba8i) uniform highp restrict writeonly iimage3D dst;
layout(set = 0, binding = 1, rgba32f) uniform highp restrict readonly image3D src;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  const bvec4 set = notEqual(imageLoad(src, pos), vec4(0.0));
  imageStore(
      dst,
      pos,
      ivec4(
          set.x ? 1 : 0,
          set.y ? 1 : 0,
          set.z ? 1 : 0,
          set.w ? 1 : 0));
}
