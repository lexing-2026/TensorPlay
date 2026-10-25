#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Height packing of an Int8 row-major payload that keeps four codes in every
 * 32-bit word: output texel (n, k/4) carries B[4j .. 4j+3][n].
 *
 * The four codes one texel needs are four consecutive bytes of weight row n,
 * and a row whose length does not divide by four splits them across word
 * boundaries, so each code is picked off its own byte lane rather than read as
 * part of the word it starts in.  A code travels as eight bits and the format
 * names its lanes as signed, so the high bit is the sign on the way out.  This
 * is the build that lets a weight kept one byte per code feed the tiled walk
 * without first laying the whole payload out as an image.
 */
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int k;  // reduction width
  int n;  // output channels
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Eight bits of one word hold the code the linear position names.
int code_at(const uint linear_offset) {
  const uint byte =
      (uInput.data[linear_offset >> 2u] >> (8u * (linear_offset & 3u))) & 0xFFu;
  // The format reads its lanes back as signed, so the high bit is the sign.
  return byte > 127u ? int(byte) - 256 : int(byte);
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int k_tex = (uBlock.k + 3) / 4;
  if (pos.x >= uBlock.n || pos.y >= k_tex) {
    return;
  }
  const uint row = uint(pos.x) * uint(uBlock.k);
  const int y0 = 4 * pos.y;
  const int v0 = (y0 < uBlock.k) ? code_at(row + uint(y0)) : 0;
  const int v1 = (y0 + 1 < uBlock.k) ? code_at(row + uint(y0 + 1)) : 0;
  const int v2 = (y0 + 2 < uBlock.k) ? code_at(row + uint(y0 + 2)) : 0;
  const int v3 = (y0 + 3 < uBlock.k) ? code_at(row + uint(y0 + 3)) : 0;
  imageStore(uOutput, ivec3(pos.x, pos.y, 0), ivec4(v0, v1, v2, v3));
}
