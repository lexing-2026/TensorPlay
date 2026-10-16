#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// LLM token sampler over a pre-sorted logits row.  The caller feeds the
// raw logits sorted in descending order plus their original indices.
// One thread per row computes the softmax weights of the surviving
// candidates (top-k truncation, then the nucleus top-p cutoff), and
// draws one token by inverse CDF over that mass.  Working on logits here
// avoids a separate full-vocabulary softmax pass: sorting logits is
// monotonic in the softmax probabilities.  The Int64 result is written
// as two uint32 halves (Vulkan textures have no 64-bit integer
// channels).
layout(set = 0, binding = 0) buffer restrict writeonly Output {
  uvec2 data[];
} uOut;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uValues;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uIndices;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uNoise;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  int k;            // sorted length (vocabulary size)
  int rows;
  int top_k;        // 0 means no top-k truncation
  float top_p;      // 1.0 means no nucleus cutoff
  float temperature;
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

  // First pass: maximum logit and the total softmax mass of the surviving
  // top-k candidates.
  float mx = -1e30;
  for (int i = 0; i < eff_k; ++i) {
    mx = max(mx, texelFetch(uValues, ivec3(i, r, 0), 0).x);
  }
  float total_all = 0.0;
  for (int i = 0; i < eff_k; ++i) {
    const float v = texelFetch(uValues, ivec3(i, r, 0), 0).x;
    total_all += exp((v - mx) / uBlock.temperature);
  }

  // Second pass: nucleus cutoff on the normalized weights (cum / total_all
  // reaching top_p), tracking the surviving mass for the inverse CDF.
  int keep_n = 0;
  float total_kept = 0.0;
  float cum = 0.0;
  for (int i = 0; i < eff_k; ++i) {
    const float v = texelFetch(uValues, ivec3(i, r, 0), 0).x;
    const float e = exp((v - mx) / uBlock.temperature);
    cum += e;
    keep_n = i + 1;
    if (uBlock.top_p < 1.0 && cum >= uBlock.top_p * total_all) {
      total_kept = cum;
      break;
    }
  }
  if (keep_n == 0) {
    keep_n = eff_k;
  }
  if (total_kept <= 0.0) {
    total_kept = total_all;
  }

  if (!(total_kept > 0.0)) {
    uOut.data[r] = uvec2(0u, 0u);
    return;
  }

  // Third pass: inverse CDF over the surviving mass.
  const float noise = texelFetch(uNoise, ivec3(r, 0, 0), 0).x;
  const float target = noise * total_kept;
  float acc = 0.0;
  int idx = keep_n - 1;
  for (int i = 0; i < keep_n; ++i) {
    const float v = texelFetch(uValues, ivec3(i, r, 0), 0).x;
    acc += exp((v - mx) / uBlock.temperature);
    if (target < acc) {
      idx = i;
      break;
    }
  }

  const int token = texelFetch(uIndices, ivec3(idx, r, 0), 0).x;
  uOut.data[r] = uvec2(uint(token), 0u);
}