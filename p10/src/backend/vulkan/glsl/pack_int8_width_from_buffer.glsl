#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Width packing of an Int8 row-major payload that keeps four codes in every
 * 32-bit word: output texel (k/4, m) carries A[m][4j .. 4j+3].
 *
 * The four codes one texel needs are four consecutive bytes of the row, and a
 * row whose length does not divide by four splits them across word boundaries,
 * so each code is picked off its own byte lane instead of reading the word it
 * starts in.  A code travels as eight bits and the format names its lanes as
 * signed, so the high bit is the sign on the way out.  The lanes past the
 * reduction width stay zero, which is what the walk's tail needs.
 */
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int k;  // reduction width
  int m;  // rows
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
  if (pos.x >= k_tex || pos.y >= uBlock.m) {
    return;
  }
  const uint row = uint(pos.y) * uint(uBlock.k);
  const int x0 = 4 * pos.x;
  const int v0 = (x0 < uBlock.k) ? code_at(row + uint(x0)) : 0;
  const int v1 = (x0 + 1 < uBlock.k) ? code_at(row + uint(x0 + 1)) : 0;
  const int v2 = (x0 + 2 < uBlock.k) ? code_at(row + uint(x0 + 2)) : 0;
  const int v3 = (x0 + 3 < uBlock.k) ? code_at(row + uint(x0 + 3)) : 0;
  imageStore(uOutput, ivec3(pos.x, pos.y, 0), ivec4(v0, v1, v2, v3));
}
