#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define OP(ACC, VAL) ${OPERATOR}
#define IDENTITY ${IDENTITY}
// clang-format on

layout(std430) buffer;

// Fold pass of a scatter over linear storage: one invocation owns one whole word
// of the destination - four single-precision lanes - and folds into each lane the
// source value of every entry of the list that names it, in the order the list
// states.  Owning the word is what makes the store safe without a read-modify-
// write of a narrower field, and a fixed order is what makes the total of a
// repeated element the same number for every caller.
//
// A lane the list never names keeps what the destination holds and is left
// unstored, so the padding of a destination whose count is not a multiple of four
// keeps whatever the allocation put there.  A lane it does names starts either
// from what the destination holds or from the identity of the fold, which is the
// whole difference between counting the old value and discarding it.
//
// An average divides by the entries that reached the lane, plus the one the
// destination contributes when it is counted.  A lane no entry names divides by
// one, so it keeps its value exactly.
layout(set = 0, binding = 0) buffer PRECISION restrict InOut {
  vec4 data[];
}
uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Names {
  int data[];
}
uNames;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Src {
  vec4 data[];
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

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uint(uBlock.count);
  if (word >= ((n + 3u) >> 2u)) {
    return;
  }

  const bool keep = uBlock.keep != 0;
  vec4 held = keep ? uSelf.data[word] : vec4(IDENTITY);
  ivec4 hits = ivec4(0);
  for (int k = 0; k < uBlock.n; ++k) {
    const int named = uNames.data[k];
    if (named < 0 || (uint(named) >> 2u) != word) {
      continue;
    }
    const float value = uSrc.data[k >> 2][k & 3];
    const int lane = named & 3;
    held[lane] = OP(held[lane], value);
    hits[lane] += 1;
  }
  bvec4 reached = greaterThan(hits, ivec4(0));
  if (any(reached)) {
    // clang-format off
    $if MEAN:
      held = mix(held, held / (vec4(hits) + vec4(keep ? 1.0 : 0.0)), reached);
    // clang-format on
    uSelf.data[word] = mix(uSelf.data[word], held, reached);
  }
}
