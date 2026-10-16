#version 450 core
layout(set = 0, binding = 0, rgba8i) uniform highp restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, rgba8i) uniform highp restrict readonly iimage3D uInput;
layout(set = 0, binding = 2) uniform highp restrict Block {
  int k;  // reduction width
  int n;  // output channels
}
uBlock;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Dense channel-packed Int8 [N, K] -> height-packed: output texel (n, k/4)
// carries B[4j .. 4j+3][n].  The input holds one element per texel x
// position in the .x lane; the tail texel is zero-padded.
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int k_tex = (uBlock.k + 3) / 4;
  if (pos.x >= uBlock.n || pos.y >= k_tex) {
    return;
  }
  const int y0 = 4 * pos.y;
  const int v0 = (y0 < uBlock.k) ? imageLoad(uInput, ivec3(y0, pos.x, 0)).x : 0;
  const int v1 = (y0 + 1 < uBlock.k) ? imageLoad(uInput, ivec3(y0 + 1, pos.x, 0)).x : 0;
  const int v2 = (y0 + 2 < uBlock.k) ? imageLoad(uInput, ivec3(y0 + 2, pos.x, 0)).x : 0;
  const int v3 = (y0 + 3 < uBlock.k) ? imageLoad(uInput, ivec3(y0 + 3, pos.x, 0)).x : 0;
  imageStore(uOutput, ivec3(pos.x, pos.y, 0), ivec4(v0, v1, v2, v3));
}