#version 450 core
layout(set = 0, binding = 0, rgba32f) uniform highp restrict writeonly image3D dst;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec3 sz = imageSize(dst);
  if (any(greaterThanEqual(pos, sz))) return;
  // The tensor is [batch, seq, seq] in row-major element order, with the
  // batch dimension packed four heads per texel along z.  pos.x is the key
  // column and pos.y the query row, so causal attention allows col <= row.
  // Every batch/head lane shares the same mask.
  const float NEG_INF = -1.0e30;
  const float val = (pos.x <= pos.y) ? 0.0f : NEG_INF;
  imageStore(dst, pos, vec4(val));
}