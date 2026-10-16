#version 450 core
layout(set = 0, binding = 0, rgba8i) uniform highp restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, rgba8i) uniform highp restrict readonly iimage3D uInput;
layout(set = 0, binding = 2) uniform highp restrict Block {
  int k;  // reduction width
  int n;  // output channels
}
uBlock;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Weight-only 4-bit packing.  The input is a dense channel-packed QInt8
// [N, K] weight (one code per texel x position, lane .x) whose values lie
// in the signed 4-bit range [-8, 7].  Each output texel (n, k/8) stores
// eight nibbles (two per byte, little nibble first), so the weight
// payload occupies half its Int8 footprint.  Tail lanes past the K edge
// are packed as zero, which is neutral under the symmetric W4A16 scaling
// (no zero point).
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int k8 = (uBlock.k + 7) / 8;
  if (pos.x >= uBlock.n || pos.y >= k8) {
    return;
  }
  const int y0 = 8 * pos.y;
  int b0 = 0;
  int b1 = 0;
  int b2 = 0;
  int b3 = 0;
  for (int i = 0; i < 8; ++i) {
    const int kk = y0 + i;
    int v = (kk < uBlock.k) ? imageLoad(uInput, ivec3(kk, pos.x, 0)).x : 0;
    v = clamp(v, -8, 7);
    // Map the signed 4-bit range to an unsigned nibble arithmetically so
    // the pack never depends on the signed bitwise-and semantics of the
    // target's int8 load: -8..-1 become 8..15, 0..7 stay 0..7.
    const int nib = (v < 0) ? (v + 16) : v;
    const int byte = i >> 1;
    if ((i & 1) == 0) {
      if (byte == 0) {
        b0 = nib;
      } else if (byte == 1) {
        b1 = nib;
      } else if (byte == 2) {
        b2 = nib;
      } else {
        b3 = nib;
      }
    } else {
      const int shifted = nib << 4;
      if (byte == 0) {
        b0 |= shifted;
      } else if (byte == 1) {
        b1 |= shifted;
      } else if (byte == 2) {
        b2 |= shifted;
      } else {
        b3 |= shifted;
      }
    }
  }
  // rgba8i clamps stored values to [-128, 127], so encode each byte in
  // signed form (0x80..0xFF become -128..-1) to preserve the nibbles.
  imageStore(
      uOutput,
      ivec3(pos.x, pos.y, 0),
      ivec4(
          b0 >= 128 ? b0 - 256 : b0,
          b1 >= 128 ? b1 - 256 : b1,
          b2 >= 128 ? b2 - 256 : b2,
          b3 >= 128 ? b3 - 256 : b3));
}