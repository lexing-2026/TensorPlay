#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Merges the chunk partials of the split-memory scan into the normalized
// attention output: each chunk contributes its exponential sum re-centered
// on the global row maximum, and the merged sum normalizes the result.  A
// lane whose global sum is zero - every entry masked - emits zeros instead
// of the 0/0 the plain formula would produce.
//
// The partial tensor [B, Hq, Sq, chunks * D] keeps chunk c's accumulator
// texels in head-dim range [c * D, (c + 1) * D); the stats tensor
// [B, Hq, Sq, chunks * 2] keeps the chunk maximum at even head-dim texels
// and the exponential sum at odd ones.  Both are Float32 whatever the
// branch dtype is.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1, rgba32f) uniform PRECISION restrict readonly image3D uPartial;
layout(set = 0, binding = 2, rgba32f) uniform PRECISION restrict readonly image3D uStats;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 extents;      // (D, Sq, B*ceil(Hq/4), unused) of the output image
  int chunks;         // number of scanned chunks
  int q_c_depth;      // ceil(Hq / 4) (unused here, padding-free layout)
  int kv_c_depth;     // unused
  int gqa_group;      // unused
  int hq;             // unused
  int fill1;
  int fill2;
  int fill3;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Substitutes one for a zero sum so the division stays finite; the merged
// numerators of such a row are all zero already.
vec4 safe_denom(const vec4 s) {
  return vec4(
      s.x == 0.0f ? 1.0f : s.x,
      s.y == 0.0f ? 1.0f : s.y,
      s.z == 0.0f ? 1.0f : s.z,
      s.w == 0.0f ? 1.0f : s.w);
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.extents.x || pos.y >= uBlock.extents.y ||
      pos.z >= uBlock.extents.z) {
    return;
  }

  const int d = pos.x;
  const int sq = pos.y;
  const int plane = pos.z;

  vec4 gmax = vec4(-3.0e38f);
  for (int c = 0; c < uBlock.chunks; ++c) {
    gmax = max(gmax, imageLoad(uStats, ivec3(2 * c, sq, plane)));
  }

  vec4 num = vec4(0.0f);
  vec4 gsum = vec4(0.0f);
  for (int c = 0; c < uBlock.chunks; ++c) {
    const vec4 w = exp(imageLoad(uStats, ivec3(2 * c, sq, plane)) - gmax);
    // The chunk partial already carries exp(score - vmax) * value; only the
    // exponential sums weight the merge.
    num += w *
        imageLoad(uPartial, ivec3(c * uBlock.extents.x + d, sq, plane));
    gsum += w * imageLoad(uStats, ivec3(2 * c + 1, sq, plane));
  }
  imageStore(uOutput, pos, num / safe_denom(gsum));
}
