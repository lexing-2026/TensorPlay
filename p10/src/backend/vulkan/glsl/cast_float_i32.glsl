#version 450 core
layout(set = 0, binding = 0, rgba32i) uniform highp restrict writeonly iimage3D dst;
layout(set = 0, binding = 1, rgba32f) uniform highp restrict readonly image3D src;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  imageStore(dst, pos, ivec4(imageLoad(src, pos)));
}
