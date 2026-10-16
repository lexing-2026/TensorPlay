#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Second SDPA stage: O = softmax(S) V, fused.
//
// The score tensor [B, Hq, Sq, Skv] is channel-packed: extents
// {Skv, Sq, B * ceil(Hq/4)} (x walks the memory sequence position, y the
// query position, lanes hold four query heads).  The value tensor
// [B, Hkv, Skv, D] is {D, Skv, B * ceil(Hkv/4)} (x walks the head
// dimension, y the memory sequence position).  The output [B, Hq, Sq, D]
// has the same extents as the value image, with the query heads occupying
// the lanes instead.
//
// One invocation computes one output position (query row, head-dim element)
// for all four query heads in its z group.  The row statistics (max and
// exponential sum) are derived by walking the score row first, the second
// walk multiplies each normalized weight by its value texel.  A lane whose
// sum is zero - every entry of the row was masked - emits zeros instead of
// the 0/0 the plain formula would produce.
//
// Group-query attention maps every query head lane onto its kv head's
// value texel exactly like the first stage does; when all four lanes land
// in one texel a single fetch serves the whole group.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uScores;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uValue;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 extents;      // (D, Sq, B*ceil(Hq/4), unused) of the output image
  int seq_len;        // Skv
  int q_c_depth;      // ceil(Hq / 4)
  int kv_c_depth;     // ceil(Hkv / 4)
  int gqa_group;      // query heads per kv head (1 = plain multi-head)
  int hq;             // query head count (lanes beyond it are padding)
  int fill1;
  int fill2;
  int fill3;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float lane_of(const vec4 v, const int idx) {
  return idx == 0 ? v.x : (idx == 1 ? v.y : (idx == 2 ? v.z : v.w));
}

// Substitutes one for a zero sum so the division stays finite; the shifted
// exponentials of such a row are all zero already.
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
  const int batch = plane / uBlock.q_c_depth;
  const int c4 = plane - batch * uBlock.q_c_depth;

  const ivec4 hq_lanes = ivec4(4 * c4 + ivec4(0, 1, 2, 3));
  const ivec4 kv_heads =
      min(hq_lanes, ivec4(uBlock.hq - 1)) / uBlock.gqa_group;
  const ivec4 lane = kv_heads % 4;
  ivec4 kv_planes = ivec4(0);
  for (int i = 0; i < 4; ++i) {
    kv_planes[i] = batch * uBlock.kv_c_depth + kv_heads[i] / 4;
  }

  // Row statistics per head lane over the Skv walk.
  vec4 vmax = vec4(-3.0e38f);
  for (int sk = 0; sk < uBlock.seq_len; ++sk) {
    vmax = max(vmax, texelFetch(uScores, ivec3(sk, sq, plane), 0));
  }
  vec4 vsum = vec4(0.0f);
  for (int sk = 0; sk < uBlock.seq_len; ++sk) {
    vsum += exp(texelFetch(uScores, ivec3(sk, sq, plane), 0) - vmax);
  }
  const vec4 denom = safe_denom(vsum);

  vec4 acc = vec4(0.0f);
  for (int sk = 0; sk < uBlock.seq_len; ++sk) {
    const vec4 p =
        exp(texelFetch(uScores, ivec3(sk, sq, plane), 0) - vmax) / denom;
    if (kv_planes[0] == kv_planes[1] && kv_planes[0] == kv_planes[2] &&
        kv_planes[0] == kv_planes[3]) {
      const vec4 v = texelFetch(uValue, ivec3(d, sk, kv_planes[0]), 0);
      acc.x += p.x * lane_of(v, lane[0]);
      acc.y += p.y * lane_of(v, lane[1]);
      acc.z += p.z * lane_of(v, lane[2]);
      acc.w += p.w * lane_of(v, lane[3]);
    } else {
      for (int i = 0; i < 4; ++i) {
        acc[i] += p[i] * lane_of(
            texelFetch(uValue, ivec3(d, sk, kv_planes[i]), 0), lane[i]);
      }
    }
  }
  imageStore(uOutput, pos, acc);
}
