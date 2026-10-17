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
 * of 256 invocations normalizes one texel row.  The first stride reads the
 * leading cached span into shared memory so its square statistics and normalized
 * write-back are served from there; any remaining tail is walked twice from
 * the texture (cache hits).  The per-lane pooled moments are tree-reduced
 * through shared memory, so the four channels sharing a texel form four
 * independent row statistics.  A rank-1 affine parameter along the width axis
 * is fetched per x position and replicated across the lanes.  Keeping the
 * workgroup small lets several rows occupy each streaming multiprocessor at
 * once, hiding memory latency behind row parallelism.
 */

const int kLocal = 256;
const int kMaxCached = 1024;
shared vec4 sVal[kMaxCached];
shared vec4 sSq[kLocal];
shared vec4 sRstd;

void main() {
  const int x0 = int(gl_LocalInvocationID.x);
  const int row_y = int(gl_WorkGroupID.y);
  const int row_z = int(gl_WorkGroupID.z);

  vec4 acc = vec4(0.0f);
  if (x0 < uBlock.span) {
    const vec4 v0 = texelFetch(uInput, ivec3(x0, row_y, row_z), 0);
    if (x0 < kMaxCached) {
      sVal[x0] = v0;
    }
    acc = v0 * v0;
  }
  for (int i = x0 + kLocal; i < uBlock.span; i += kLocal) {
    const vec4 vi = texelFetch(uInput, ivec3(i, row_y, row_z), 0);
    if (i < kMaxCached) {
      sVal[i] = vi;
    }
    acc += vi * vi;
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

  if (x0 < uBlock.span) {
    vec4 w0 = vec4(1.0f);
    // clang-format off
    $if WEIGHT_MODE == 3:
      w0 = vec4(texelFetch(uWeight, ivec3(x0, 0, 0), 0).x);
    // clang-format on
    const vec4 v0 = (x0 < kMaxCached)
        ? sVal[x0]
        : texelFetch(uInput, ivec3(x0, row_y, row_z), 0);
    imageStore(
        uOutput,
        ivec3(x0, row_y, row_z),
        v0 * sRstd * w0);
  }
  for (int i = x0 + kLocal; i < uBlock.span; i += kLocal) {
    vec4 wi = vec4(1.0f);
    // clang-format off
    $if WEIGHT_MODE == 3:
      wi = vec4(texelFetch(uWeight, ivec3(i, 0, 0), 0).x);
    // clang-format on
    const vec4 vi = (i < kMaxCached)
        ? sVal[i]
        : texelFetch(uInput, ivec3(i, row_y, row_z), 0);
    imageStore(uOutput, ivec3(i, row_y, row_z), vi * sRstd * wi);
  }
}