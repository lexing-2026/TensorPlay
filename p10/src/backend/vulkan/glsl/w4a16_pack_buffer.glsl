#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define INPUT_BUFFER ${INPUT_BUFFER}
// clang-format on

layout(std430) buffer;

/*
 * Four-bit weight codes into a nibble stream in linear storage.  The source is
 * the QInt8 payload of an [N, K] weight, one code per element: read row-major
 * from words of four bytes where the weight has no texel form, or from the
 * weight's own texel grid, one code in the first lane of each texel, where it
 * has one but the product around it does not.  The destination packs eight
 * codes into every 32-bit word, two per byte with the lower nibble first, and
 * gives every row its own run of ceil(K / 8) words.
 *
 * Padding a row up to a whole word keeps a weight fetch aligned for the reader,
 * which walks the reduction one word at a time; the cost is that a row whose
 * length does not divide by eight leaves unused nibble slots.  Those slots pack
 * as zero, the neutral code under this scheme's symmetric scaling, so a reader
 * that runs to the end of the word adds nothing.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
// clang-format off
$if INPUT_BUFFER == 1:
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
    uint data[];
  }
  uInput;
$else:
  layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uInput;
// clang-format on
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int k; // codes per weight row
  int n; // weight rows
  int k_words; // ceil(k / 8): words per packed row
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// clang-format off
$if INPUT_BUFFER == 1:
  int code_at(const in int row, const in int kk) {
    const int index = row * uBlock.k + kk;
    const uint byte =
        (uInput.data[index >> 2] >> (uint(index & 3) * 8u)) & 0xFFu;
    // Bit 7 of an int8 lane is its sign, so a byte of 128 or more names a
    // negative code.
    return int(byte & 0x7Fu) - int(byte & 0x80u);
  }
$else:
  int code_at(const in int row, const in int kk) {
    // An rgba8i lane arrives as the signed byte it was stored as, so the code
    // needs no sign repair before the range clamp.
    return imageLoad(uInput, ivec3(kk, row, 0)).x;
  }
// clang-format on

void main() {
  const int word = int(gl_GlobalInvocationID.x);
  if (word >= uBlock.n * uBlock.k_words) {
    return;
  }

  const int row = word / uBlock.k_words;
  const int first = (word - row * uBlock.k_words) * 8;

  uint packed = 0u;
  for (int i = 0; i < 8; ++i) {
    const int kk = first + i;
    const int code = kk < uBlock.k ? code_at(row, kk) : 0;
    // Codes beyond the signed 4-bit range are pulled to its edges.  The range
    // is crossed through the unsigned values, so the pack never depends on the
    // signed shift semantics of the target.
    const int clamped = clamp(code, -8, 7);
    const uint nib = uint(clamped < 0 ? clamped + 16 : clamped);
    packed |= nib << uint(i * 4);
  }

  uOutput.data[word] = packed;
}
