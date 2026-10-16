#version 450 core
layout(set = 0, binding = 0, rgba8i) uniform highp restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, rgba8i) uniform highp restrict readonly iimage3D uInput;
layout(set = 0, binding = 2) uniform highp restrict Block {
  int k;  // reduction width
  int m;  // rows
}
uBlock;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Dense channel-packed Int8 [M, K] -> width-packed: output texel (k/4, m)
// carries A[m][4j .. 4j+3].  The input holds one element per texel x
// position in the .x lane; the tail texel is zero-padded so the GEMM's
// reduction loop can read whole texels.
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int k_tex = (uBlock.k + 3) / 4;
  if (pos.x >= k_tex || pos.y >= uBlock.m) {
    return;
  }
  const int x0 = 4 * pos.x;
  const int v0 = (x0 < uBlock.k) ? imageLoad(uInput, ivec3(x0, pos.y, 0)).x : 0;
  const int v1 = (x0 + 1 < uBlock.k) ? imageLoad(uInput, ivec3(x0 + 1, pos.y, 0)).x : 0;
  const int v2 = (x0 + 2 < uBlock.k) ? imageLoad(uInput, ivec3(x0 + 2, pos.y, 0)).x : 0;
  const int v3 = (x0 + 3 < uBlock.k) ? imageLoad(uInput, ivec3(x0 + 3, pos.y, 0)).x : 0;
  imageStore(uOutput, ivec3(pos.x, pos.y, 0), ivec4(v0, v1, v2, v3));
}