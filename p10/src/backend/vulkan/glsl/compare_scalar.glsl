#version 450 core
#define OP(X, Y) ${OPERATOR}
layout(set = 0, binding = 0, rgba8i) uniform highp restrict writeonly iimage3D dst;
$if DTYPE == "float":
  layout(set = 0, binding = 1) uniform highp sampler3D src;
$elif DTYPE == "uint8":
  layout(set = 0, binding = 1) uniform highp usampler3D src;
$else:
  layout(set = 0, binding = 1) uniform highp isampler3D src;
layout(set = 0, binding = 2) uniform highp restrict Block {
  float value_float;
  int value_int;
  uint value_uint;
  int fill;
} uBlock;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  $if DTYPE == "float":
    vec4 value = vec4(uBlock.value_float);
  $elif DTYPE == "uint8":
    uvec4 value = uvec4(uBlock.value_uint);
  $else:
    ivec4 value = ivec4(uBlock.value_int);
  imageStore(dst, pos, ivec4(OP(texelFetch(src, pos, 0), value)));
}
