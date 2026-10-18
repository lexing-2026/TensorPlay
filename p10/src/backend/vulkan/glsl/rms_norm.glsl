#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define WEIGHT_MODE ${WEIGHT_MODE}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 in_sizes; // (W, H, C, N) of the input
  int span;       // elements of the normalized trailing axis
  float inv_span; // 1 / span
  float eps;
  int pad; // uniform block alignment
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Root-mean-square normalization over the trailing width axis.  One workgroup
 * of 256 invocations normalizes one texel row.  The first stride accumulates
 * the square statistics and the write-back stride reuses them, so a row is
 * fetched once from memory.
 *
 * The reuse happens in registers rather than in shared memory: an invocation
 * holds the four texels it strides over in its own locals, which costs nothing
 * against the per-workgroup shared budget, whereas mirroring the row into
 * shared would need 16 KB on top of the reduction scratch and leave a 32 KB
 * compute unit hosting a single row with every barrier serialized behind it.
 * Positions past the fourth stride are re-read from the texture; a hidden axis
 * of up to 1024 texels (4096 channels) is fully covered.  The four channels
 * sharing a texel stay independent, so the pooled moments are tree-reduced as
 * vec4 through shared memory and a rank-1 affine parameter along the width axis
 * is fetched per x position and replicated across the lanes.
 */

const int kLocal = 256;
shared vec4 sSq[kLocal];
shared vec4 sRstd;

// The affine parameter is rank-1 along the width axis, so a position's scale
// is one texel's first lane replicated; without a parameter every position
// scales by one.  Fetched at the store it feeds rather than up front, so the
// kept input values stay the only long-lived registers of the invocation.
vec4 weight_at(const in int i) {
  vec4 w = vec4(1.0f);
  // clang-format off
  $if WEIGHT_MODE == 3:
    w = vec4(texelFetch(uWeight, ivec3(i, 0, 0), 0).x);
  // clang-format on
  return w;
}

void main() {
  const int x0 = int(gl_LocalInvocationID.x);
  const int row_y = int(gl_WorkGroupID.y);
  const int row_z = int(gl_WorkGroupID.z);
  const int span = uBlock.span;

  // The four positions this invocation strides over (x0, x0 + 256, +512, +768)
  // are written as constant-indexed locals so no compiler can demote them to
  // private memory; a hidden axis of up to 1024 texels is therefore fully
  // covered without a second fetch.
  vec4 k0 = vec4(0.0f);
  vec4 k1 = vec4(0.0f);
  vec4 k2 = vec4(0.0f);
  vec4 k3 = vec4(0.0f);
  const int i1 = x0 + kLocal;
  const int i2 = x0 + 2 * kLocal;
  const int i3 = x0 + 3 * kLocal;
  if (x0 < span) {
    k0 = texelFetch(uInput, ivec3(x0, row_y, row_z), 0);
  }
  if (i1 < span) {
    k1 = texelFetch(uInput, ivec3(i1, row_y, row_z), 0);
  }
  if (i2 < span) {
    k2 = texelFetch(uInput, ivec3(i2, row_y, row_z), 0);
  }
  if (i3 < span) {
    k3 = texelFetch(uInput, ivec3(i3, row_y, row_z), 0);
  }
  vec4 acc = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
  for (int i = x0 + 4 * kLocal; i < span; i += kLocal) {
    const vec4 v = texelFetch(uInput, ivec3(i, row_y, row_z), 0);
    acc += v * v;
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
    sRstd = inversesqrt(sSq[0] * uBlock.inv_span + vec4(uBlock.eps));
  }
  barrier();

  if (x0 < span) {
    imageStore(uOutput, ivec3(x0, row_y, row_z), k0 * sRstd * weight_at(x0));
  }
  if (i1 < span) {
    imageStore(uOutput, ivec3(i1, row_y, row_z), k1 * sRstd * weight_at(i1));
  }
  if (i2 < span) {
    imageStore(uOutput, ivec3(i2, row_y, row_z), k2 * sRstd * weight_at(i2));
  }
  if (i3 < span) {
    imageStore(uOutput, ivec3(i3, row_y, row_z), k3 * sRstd * weight_at(i3));
  }
  for (int i = x0 + 4 * kLocal; i < span; i += kLocal) {
    const vec4 v = texelFetch(uInput, ivec3(i, row_y, row_z), 0);
    imageStore(uOutput, ivec3(i, row_y, row_z), v * sRstd * weight_at(i));
  }
}