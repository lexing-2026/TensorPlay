#version 450 core
// Unsigned-byte payload (a quantization code, a mask stored as bytes) to
// single precision: the lanes are read as unsigned integers and each converts
// exactly, since every code fits the 24-bit mantissa range of a float.
layout(set = 0, binding = 0, rgba32f) uniform highp restrict writeonly image3D dst;
layout(set = 0, binding = 1, rgba8ui) uniform highp restrict readonly uimage3D src;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  imageStore(dst, pos, vec4(imageLoad(src, pos)));
}
