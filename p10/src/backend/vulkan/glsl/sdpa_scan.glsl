#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Split-memory scan for short queries over long key ranges.  One
// invocation reduces one chunk of the key axis for one output head-dim
// element and a head group: it walks that chunk of the score row, and
// folds the value contribution of each position into unnormalized partial
// accumulators.  A combine pass rescales the chunks by their maxima and
// merges them into the normalized output, which keeps every invocation's
// serial walk short and the chunk reduction parallel.
//
// The score tensor [B, Hq, Sq, Skv] is channel-packed: x walks the memory
// position, y the query position, z a group of four query heads.  The
// value tensor [B, Hkv, Skv, D] is {D, Skv, B * ceil(Hkv/4)}.  The partial
// tensor [B, Hq, Sq, chunks * D] has extents {chunks * D, Sq,
// B * ceil(Hq/4)} and lives in Float32 whatever the branch dtype is;
// chunk c occupies head-dim texels [c * D, (c + 1) * D).  The stats tensor
// [B, Hq, Sq, chunks * 2] keeps the chunk maximum at even head-dim texels
// and the exponential sum at odd ones.
layout(set = 0, binding = 0) uniform PRECISION sampler3D uScores;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uValue;
layout(set = 0, binding = 2, rgba32f) uniform PRECISION restrict writeonly image3D uPartial;
layout(set = 0, binding = 3, rgba32f) uniform PRECISION restrict writeonly image3D uStats;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 extents;      // (D, Sq, B*ceil(Hq/4), unused)
  int skv;            // full key axis length
  int chunk_len;      // keys per chunk (the last one may be shorter)
  int q_c_depth;      // ceil(Hq / 4)
  int kv_c_depth;     // ceil(Hkv / 4)
  int gqa_group;      // query heads per kv head (1 = plain multi-head)
  int hq;             // query head count (lanes beyond it are padding)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float lane_of(const vec4 v, const int idx) {
  return idx == 0 ? v.x : (idx == 1 ? v.y : (idx == 2 ? v.z : v.w));
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  // Grid is (D * chunks, Sq, planes): x decides both the chunk and the
  // head-dim element the partial accumulates.
  if (pos.x >= uBlock.extents.x * uBlock.extents.w ||
      pos.y >= uBlock.extents.y || pos.z >= uBlock.extents.z) {
    return;
  }

  const int chunk = pos.x / uBlock.extents.x;
  const int d = pos.x - chunk * uBlock.extents.x;
  const int sq = pos.y;
  const int plane = pos.z;
  const int batch = plane / uBlock.q_c_depth;
  const int c4 = plane - batch * uBlock.q_c_depth;

  // Group-query attention: same lane mapping as the other stages.
  const ivec4 hq_lanes = ivec4(4 * c4 + ivec4(0, 1, 2, 3));
  const ivec4 kv_heads =
      min(hq_lanes, ivec4(uBlock.hq - 1)) / uBlock.gqa_group;
  const ivec4 lane = kv_heads % 4;
  ivec4 kv_planes = ivec4(0);
  for (int i = 0; i < 4; ++i) {
    kv_planes[i] = batch * uBlock.kv_c_depth + kv_heads[i] / 4;
  }
  const bool same_texel = kv_planes[0] == kv_planes[1] &&
      kv_planes[0] == kv_planes[2] && kv_planes[0] == kv_planes[3];

  const int begin = chunk * uBlock.chunk_len;
  const int end = min(begin + uBlock.chunk_len, uBlock.skv);

  vec4 vmax = vec4(-3.0e38f);
  for (int sk = begin; sk < end; ++sk) {
    vmax = max(vmax, texelFetch(uScores, ivec3(sk, sq, plane), 0));
  }
  vec4 vsum = vec4(0.0f);
  for (int sk = begin; sk < end; ++sk) {
    vsum += exp(texelFetch(uScores, ivec3(sk, sq, plane), 0) - vmax);
  }

  vec4 acc = vec4(0.0f);
  for (int sk = begin; sk < end; ++sk) {
    const vec4 p = exp(texelFetch(uScores, ivec3(sk, sq, plane), 0) - vmax);
    if (same_texel) {
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

  imageStore(uPartial, ivec3(chunk * uBlock.extents.x + d, sq, plane), acc);
  imageStore(uStats, ivec3(2 * chunk, sq, plane), vmax);
  imageStore(uStats, ivec3(2 * chunk + 1, sq, plane), vsum);
}
