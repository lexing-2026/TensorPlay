#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define OP(ACC, VAL) ${OPERATOR}
#define IDENTITY ${IDENTITY}
// clang-format on

layout(std430) buffer;

// Fold pass of a scatter that moves a half payload, over linear storage: one
// invocation owns one whole 32-bit word, which carries two lanes, and folds into
// each lane the source value of every entry of the list that names it.  The pair
// sharing a word is why the ownership is the word and not the lane: an odd row
// length puts the last half of one element beside the first of the next, so a
// lane-only store would hand a stale half to whoever owns the other.
//
// The running fold is kept at full width and rounded on the way out, which is how
// the row accumulation already treats a half payload.  As there, a lane the list
// never names is left unstored, and an average divides by the entries that
// reached the lane plus the destination itself when it is counted.
layout(set = 0, binding = 0) buffer PRECISION restrict InOut {
  uint data[];
}
uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Names {
  int data[];
}
uNames;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Src {
  uint data[];
}
uSrc;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int count; // elements of the destination
  int n;     // entries of the index list
  int keep;  // whether the destination counts as a contribution
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// The 16-bit field holding one element of a dense half payload.
float half_field(uint word, uint lane) {
  return unpackHalf2x16(word)[lane];
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uint(uBlock.count);
  if (word >= ((n + 1u) >> 1u)) {
    return;
  }

  const bool keep = uBlock.keep != 0;
  vec2 held = keep ? unpackHalf2x16(uSelf.data[word]) : vec2(IDENTITY);
  ivec2 hits = ivec2(0);
  for (int k = 0; k < uBlock.n; ++k) {
    const int named = uNames.data[k];
    if (named < 0 || (uint(named) >> 1u) != word) {
      continue;
    }
    const float value = half_field(uSrc.data[k >> 1], k & 1);
    const int lane = named & 1;
    held[lane] = OP(held[lane], value);
    hits[lane] += 1;
  }
  bvec2 reached = greaterThan(hits, ivec2(0));
  if (any(reached)) {
    // clang-format off
    $if MEAN:
      held = mix(held, held / (vec2(hits) + vec2(keep ? 1.0 : 0.0)), reached);
    // clang-format on
    uSelf.data[word] = packHalf2x16(
        mix(unpackHalf2x16(uSelf.data[word]), held, reached));
  }
}
