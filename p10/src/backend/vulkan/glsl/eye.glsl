#version 450 core
$if DTYPE == "int":
  layout(set = 0, binding = 0, ${FORMAT}) uniform highp restrict writeonly iimage3D dst;
$else:
  layout(set = 0, binding = 0, ${FORMAT}) uniform highp restrict writeonly image3D dst;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, imageSize(dst)))) return;
  $if DTYPE == "int":
    imageStore(dst, pos, ivec4(int(pos.x == pos.y), 0, 0, 0));
  $else:
    imageStore(dst, pos, vec4(float(pos.x == pos.y), 0, 0, 0));
}
