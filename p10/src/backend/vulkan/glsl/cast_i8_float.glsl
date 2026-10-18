#version 450 core
// A Bool or Int8 payload lives in a signed-byte texture, four lanes per texel.
// Reading it as integers and storing them into a float texture is the whole
// conversion: a Bool lane is already 0 or 1, and GLSL converts each integer
// component exactly.
layout(set = 0, binding = 0, rgba32f) uniform highp restrict writeonly image3D dst;
layout(set = 0, binding = 1, rgba8i) uniform highp restrict readonly iimage3D src;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  imageStore(dst, pos, vec4(imageLoad(src, pos)));
}
