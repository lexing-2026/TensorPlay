#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Fused root-mean-square normalization of the trailing axis over half-precision
 * linear storage.  One workgroup of 256 invocations owns one row; each
 * invocation owns one 32-bit word, which holds the two lanes of two adjacent
 * elements, low half first.  Word ownership is what makes the store safe: no
 * lane is ever written by two invocations, so no read-modify-write dance around
 * a shared word is needed.
 *
 * The moment is accumulated in single precision from the unpacked lanes and only
 * the store narrows, matching how the pointwise half kernels treat arithmetic.
 * A row of up to 2048 elements stays in registers across the four cached words;
 * longer rows re-read their tail.
 *
 * `span` counts elements, so the row's first element is always at an even index:
 * an odd normalized axis would straddle a word between two rows and is kept off
 * this path by the dispatcher.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int span;      // elements of the normalized trailing axis
  int rows;      // rows to normalize
  int rows_axis; // rows named by the second axis before the third takes over
  float inv_span; // 1 / span
  float eps;
  int pad0;
  int pad1;
  int pad2;
}
uBlock;
// clang-format off
$if WEIGHT_MODE == 1:
  layout(set = 0, binding = 3) buffer PRECISION restrict readonly Weight {
    uint data[];
  }
  uWeight;
$elif WEIGHT_MODE == 2:
  layout(set = 0, binding = 3) uniform PRECISION sampler3D uWeight;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const int kLocal = 256;
shared float sSq[kLocal];
shared float sRstd;

vec2 word_lanes(const in uint word) {
  return unpackHalf2x16(word);
}

// The affine parameter is rank-1 along the normalized axis: a short weight
// stays an image and is read per position, a wide one is a flat half array, and
// without a parameter every position scales by one.
float weight_lane(const in int i) {
  // clang-format off
  $if WEIGHT_MODE == 1:
    const vec2 packed = word_lanes(uWeight.data[i >> 1]);
    return (i & 1) == 0 ? packed.x : packed.y;
  $elif WEIGHT_MODE == 2:
    return texelFetch(uWeight, ivec3(i, 0, 0), 0).x;
  $else:
    return 1.0f;
  // clang-format on
}

vec2 weight_at(const in int i0) {
  return vec2(weight_lane(i0), weight_lane(i0 + 1));
}

void main() {
  const int x0 = int(gl_LocalInvocationID.x);
  // The workgroup, not the invocation, names the row: the x axis carries the
  // 256 words that stride along it.
  const int row =
      int(gl_WorkGroupID.y) + int(gl_WorkGroupID.z) * uBlock.rows_axis;
  if (row >= uBlock.rows) {
    return;
  }
  const int words = uBlock.span >> 1;
  const int base = row * words;

  // Constant-indexed locals keep the four strided words out of private memory.
  const int w1 = x0 + kLocal;
  const int w2 = x0 + 2 * kLocal;
  const int w3 = x0 + 3 * kLocal;
  vec2 k0 = vec2(0.0f);
  vec2 k1 = vec2(0.0f);
  vec2 k2 = vec2(0.0f);
  vec2 k3 = vec2(0.0f);
  if (x0 < words) {
    k0 = word_lanes(uInput.data[base + x0]);
  }
  if (w1 < words) {
    k1 = word_lanes(uInput.data[base + w1]);
  }
  if (w2 < words) {
    k2 = word_lanes(uInput.data[base + w2]);
  }
  if (w3 < words) {
    k3 = word_lanes(uInput.data[base + w3]);
  }
  float acc =
      dot(k0, k0) + dot(k1, k1) + dot(k2, k2) + dot(k3, k3);
  for (int w = x0 + 4 * kLocal; w < words; w += kLocal) {
    const vec2 v = word_lanes(uInput.data[base + w]);
    acc += dot(v, v);
  }

  sSq[x0] = acc;
  barrier();
  for (int stride = kLocal / 2; stride > 0; stride >>= 1) {
    if (x0 < stride) {
      sSq[x0] += sSq[x0 + stride];
    }
    barrier();
  }
  if (x0 == 0) {
    sRstd = inversesqrt(sSq[0] * uBlock.inv_span + uBlock.eps);
  }
  barrier();

  const float rstd = sRstd;
  if (x0 < words) {
    uOutput.data[base + x0] = packHalf2x16(
        k0 * rstd * weight_at(2 * x0));
  }
  if (w1 < words) {
    uOutput.data[base + w1] = packHalf2x16(
        k1 * rstd * weight_at(2 * w1));
  }
  if (w2 < words) {
    uOutput.data[base + w2] = packHalf2x16(
        k2 * rstd * weight_at(2 * w2));
  }
  if (w3 < words) {
    uOutput.data[base + w3] = packHalf2x16(
        k3 * rstd * weight_at(2 * w3));
  }
  for (int w = x0 + 4 * kLocal; w < words; w += kLocal) {
    uOutput.data[base + w] = packHalf2x16(
        word_lanes(uInput.data[base + w]) * rstd * weight_at(2 * w));
  }
}
