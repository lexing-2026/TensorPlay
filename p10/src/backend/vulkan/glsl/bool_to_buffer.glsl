#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Bool texture payload to dense byte payload.  A Bool plane is a signed-byte
 * image with one logical channel per texel lane, while a buffer-stored Bool
 * keeps one byte per element in row-major order, so the two disagree about
 * which four elements share a 32-bit unit.  One invocation assembles one
 * destination word: it walks its four logical positions, maps each through the
 * texel geometry, and packs the resulting bytes into the single word it owns,
 * which keeps two invocations from writing different bytes of one word.
 * Bytes past the element count are written as zero so the padded tail of the
 * allocation stays defined.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) logical sizes
  ivec4 sizes;
  int c_depth; // ceil(C / 4)
  int numel;   // logical element count
  int fill0;
  int fill1;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  const int words = (uBlock.numel + 3) / 4;
  if (int(idx) >= words) {
    return;
  }

  const int total = uBlock.sizes.x * uBlock.sizes.y * uBlock.sizes.z *
      uBlock.sizes.w;
  uint word = 0u;
  for (int i = 0; i < 4; ++i) {
    const int linear = int(idx) * 4 + i;
    if (linear >= uBlock.numel || linear >= total) {
      break;
    }
    // Decompose the row-major position over (N, C, H, W).
    const int w = linear % uBlock.sizes.x;
    int rest = linear / uBlock.sizes.x;
    const int h = rest % uBlock.sizes.y;
    rest /= uBlock.sizes.y;
    const int c = rest % uBlock.sizes.z;
    const int n = rest / uBlock.sizes.z;

    const ivec4 texel = imageLoad(
        uInput,
        ivec3(w, h, n * uBlock.c_depth + c / 4));
    const uint byte = texel[c % 4] != 0 ? 1u : 0u;
    word |= byte << (8u * uint(i));
  }
  uOutput.data[idx] = word;
}
