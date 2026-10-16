#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// LLM token sampler over a pre-sorted probability row.  The caller feeds
// the softmax probabilities sorted in descending order plus their
// original indices.  One thread per row applies the top-k truncation
// (keep the first k entries), the top-p nucleus cutoff (accumulate the
// descending probabilities until the absolute threshold is reached, as
// in the reference sampler), then draws one token by inverse CDF over
// the surviving mass.  The Int64 result is written as two uint32 halves
// (Vulkan textures have no 64-bit integer channels).
layout(set = 0, binding = 0) buffer restrict writeonly Output {
  uvec2 data[];
} uOut;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uValues;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uIndices;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uNoise;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  int k;         // sorted length (vocabulary size)
  int rows;
  int top_k;     // 0 means no top-k truncation
  float top_p;   // 1.0 means no nucleus cutoff
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int r = int(gl_GlobalInvocationID.x);
  if (r >= uBlock.rows) {
    return;
  }

  const int eff_k =
      (uBlock.top_k > 0 && uBlock.top_k < uBlock.k) ? uBlock.top_k : uBlock.k;

  float total_kept = 0.0;
  int keep_n = 0;
  float cum = 0.0;
  for (int i = 0; i < eff_k; ++i) {
    const float v = texelFetch(uValues, ivec3(i, r, 0), 0).x;
    total_kept += v;
    cum += v;
    keep_n = i + 1;
    if (uBlock.top_p < 1.0 && cum >= uBlock.top_p) {
      break;
    }
  }

  if (!(total_kept > 0.0)) {
    uOut.data[r] = uvec2(0u, 0u);
    return;
  }

  const float noise = texelFetch(uNoise, ivec3(r, 0, 0), 0).x;
  const float target = noise * total_kept;
  float acc = 0.0;
  int idx = keep_n - 1;
  for (int i = 0; i < keep_n; ++i) {
    acc += texelFetch(uValues, ivec3(i, r, 0), 0).x;
    if (target < acc) {
      idx = i;
      break;
    }
  }

  const int token = texelFetch(uIndices, ivec3(idx, r, 0), 0).x;
  uOut.data[r] = uvec2(uint(token), 0u);
}