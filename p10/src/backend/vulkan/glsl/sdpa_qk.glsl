#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// First SDPA stage: S = Q K^T (with causal clamping and additive mask).
//
// The 4-D tensors are channel-packed, so an image of a [B, Hq, Sq, D] tensor
// has extents {D, S, B * ceil(Hq/4)}: the x axis walks the head dimension D
// one element per texel, y is the query position, z selects a group of
// four query heads and the texel lanes hold those four heads.  The scores
// image (a [B, Hq, Sq, Skv] tensor) is {Skv, Sq, B * ceil(Hq/4)} with x
// walking the key/memory sequence position, y the query position and lanes
// holding the four query heads of a z group.
//
// One invocation computes the scores of one (query row, memory position)
// pair for all four heads in its z group.  The dot product over D reads one
// element per texel; the four head values ride in the lanes, so each lane
// accumulates its own head's dot product.
//
// Group-query attention: a query head reads the key/value head
// h / (Hq / Hkv) instead of its own index.  The four kv texel coordinates
// of a lane group are precomputed, and when every lane lands in the same
// kv texel (always true when Hq == Hkv, and for aligned groups) one fetch
// serves all four lanes; only misaligned groups pay four fetches.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uScores;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uQuery;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uKey;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uMask;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 extents;      // (Skv, Sq, B*ceil(Hq/4), unused) of the scores image
  int dim;            // head_dim D
  float scale;
  int is_causal;
  int has_mask;
  int mask_mode;      // 0: shared, broadcast lane 0 of plane zero
                      // 1: per-head, mask plane equals the score plane
                      // 2: per-batch, mask plane = scores plane / q_c_depth
                      // 3: per-head over batch axis 1, plane = plane % q_c_depth
  int q_c_depth;      // ceil(Hq / 4)
  int kv_c_depth;     // ceil(Hkv / 4)
  int gqa_group;      // query heads per kv head (1 = plain multi-head)
  int hq;             // query head count (lanes beyond it are padding)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Selects one lane of a texel without relying on dynamic component
// subscripting, which stays outside the guaranteed subset.
float lane_of(const vec4 v, const int idx) {
  return idx == 0 ? v.x : (idx == 1 ? v.y : (idx == 2 ? v.z : v.w));
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.extents.x || pos.y >= uBlock.extents.y ||
      pos.z >= uBlock.extents.z) {
    return;
  }

  const int sk = pos.x;
  const int sq = pos.y;
  const int plane = pos.z;
  if (uBlock.is_causal != 0 && sk > sq) {
    // Top-left aligned causal: query position t attends keys <= t.  The
    // masked half stores the clamp sentinel and skips the walk over D.
    imageStore(uScores, pos, vec4(-1.0e30f));
    return;
  }
  const int batch = plane / uBlock.q_c_depth;
  const int c4 = plane - batch * uBlock.q_c_depth;

  // Query head of each lane; padding lanes clamp so their mapped kv head
  // stays in range (their results are never read back).
  const ivec4 hq_lanes = ivec4(4 * c4 + ivec4(0, 1, 2, 3));
  const ivec4 kv_heads =
      min(hq_lanes, ivec4(uBlock.hq - 1)) / uBlock.gqa_group;
  const ivec4 lane = kv_heads % 4;
  ivec4 kv_planes = ivec4(0);
  for (int i = 0; i < 4; ++i) {
    kv_planes[i] = batch * uBlock.kv_c_depth + kv_heads[i] / 4;
  }

  vec4 acc = vec4(0.0f);
  for (int d = 0; d < uBlock.dim; ++d) {
    const vec4 q = texelFetch(uQuery, ivec3(d, sq, plane), 0);
    if (kv_planes[0] == kv_planes[1] && kv_planes[0] == kv_planes[2] &&
        kv_planes[0] == kv_planes[3]) {
      const vec4 k = texelFetch(uKey, ivec3(d, sk, kv_planes[0]), 0);
      acc.x += q.x * lane_of(k, lane[0]);
      acc.y += q.y * lane_of(k, lane[1]);
      acc.z += q.z * lane_of(k, lane[2]);
      acc.w += q.w * lane_of(k, lane[3]);
    } else {
      for (int i = 0; i < 4; ++i) {
        acc[i] += q[i] * lane_of(
            texelFetch(uKey, ivec3(d, sk, kv_planes[i]), 0), lane[i]);
      }
    }
  }
  acc *= uBlock.scale;

  if (uBlock.is_causal != 0 && sk > sq) {
    acc = vec4(-1.0e30f);
  }

  if (uBlock.has_mask != 0) {
    // Mask planes are always XSq x Skv; their channels are head lanes.  A
    // mask with a size-1 head axis (or a 2-D shared mask) stores data only
    // in lane 0, so that lane is broadcast over the head lanes; head-axis
    // masks keep one lane per head.
    vec4 mask;
    if (uBlock.mask_mode == 1) {
      mask = texelFetch(uMask, ivec3(sk, sq, plane), 0);
    } else if (uBlock.mask_mode == 3) {
      mask = texelFetch(uMask, ivec3(sk, sq, plane % uBlock.q_c_depth), 0);
    } else if (uBlock.mask_mode == 2) {
      mask = vec4(texelFetch(uMask, ivec3(sk, sq, batch), 0).x);
    } else {
      mask = vec4(texelFetch(uMask, ivec3(sk, sq, 0), 0).x);
    }
    acc += mask;
  }

  imageStore(uScores, pos, acc);
}
