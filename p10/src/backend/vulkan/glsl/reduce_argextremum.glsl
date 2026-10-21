#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, rgba32i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 in_sizes; // (W, H, C, N) of the input
  ivec4 out_sizes; // (W, H, C, N) of the output
  int axis; // 0: W, 1: H, 2: C, 3: N
  int in_c_depth;
  int out_c_depth;
  int greater; // 1: track the maximum, 0: track the minimum
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Single-axis extremum-index reduction.  Width, height, and batch
 * reductions track (value, position) per texel lane — the lanes reduce
 * independently, so the output carries one position per channel lane.
 * Strict comparisons implement the first-occurrence tie-break.  A lane that
 * is not a number owns the row it sits in: it takes the row from a numeric
 * lane and holds it against every later lane, so the earliest such lane is
 * the one named.  A channel reduction walks the texel depth of
 * one batch over all real lanes and collapses the lanes into one value with
 * the winning global position replicated across the output lanes; padded
 * lanes are skipped.
 */

// Whether a candidate takes a row from the value currently held.  A held lane
// that is not a number is never displaced, so its position stays the earliest
// one; a numeric lane falls back to the strict comparison.
vec4 extremum_take(const in vec4 candidate, const in vec4 held,
                   const in int greater) {
  const vec4 strict = greater != 0 ? vec4(greaterThan(candidate, held))
                                   : vec4(lessThan(candidate, held));
  return clamp(strict + vec4(isnan(candidate)) - vec4(isnan(held)), 0.0, 1.0);
}

bool extremum_take1(const in float candidate, const in float held,
                    const in int greater) {
  if (isnan(held)) {
    return false;
  }
  if (isnan(candidate)) {
    return true;
  }
  return greater != 0 ? candidate > held : candidate < held;
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  vec4 acc;
  vec4 acc_pos;

  if (uBlock.axis == 0) {
    acc = texelFetch(uInput, ivec3(0, pos.y, pos.z), 0);
    acc_pos = vec4(0.0f);
    for (int x = 1; x < uBlock.in_sizes.x; ++x) {
      const vec4 v = texelFetch(uInput, ivec3(x, pos.y, pos.z), 0);
      const vec4 take = extremum_take(v, acc, uBlock.greater);
      acc = mix(acc, v, take);
      acc_pos = mix(acc_pos, vec4(float(x)), take);
    }
    imageStore(uOutput, pos, ivec4(acc_pos));
  } else if (uBlock.axis == 1) {
    acc = texelFetch(uInput, ivec3(pos.x, 0, pos.z), 0);
    acc_pos = vec4(0.0f);
    for (int y = 1; y < uBlock.in_sizes.y; ++y) {
      const vec4 v = texelFetch(uInput, ivec3(pos.x, y, pos.z), 0);
      const vec4 take = extremum_take(v, acc, uBlock.greater);
      acc = mix(acc, v, take);
      acc_pos = mix(acc_pos, vec4(float(y)), take);
    }
    imageStore(uOutput, pos, ivec4(acc_pos));
  } else if (uBlock.axis == 2) {
    // Channel: positions are global channel indices within the batch;
    // padded lanes are skipped.
    const int n = pos.z / uBlock.out_c_depth;
    const int channels = uBlock.in_sizes.z;
    bool first = true;
    float best = 0.0f;
    float best_pos = 0.0f;
    for (int z = n * uBlock.in_c_depth; z < (n + 1) * uBlock.in_c_depth; ++z) {
      const vec4 v = texelFetch(uInput, ivec3(pos.x, pos.y, z), 0);
      const int base = (z - n * uBlock.in_c_depth) * 4;
      for (int lane = 0; lane < 4; ++lane) {
        if (base + lane >= channels) {
          continue;
        }
        const float value = v[lane];
        const bool take = first || extremum_take1(value, best, uBlock.greater);
        if (take) {
          best = value;
          best_pos = float(base + lane);
          first = false;
        }
      }
    }
    imageStore(uOutput, ivec3(pos.x, pos.y, n), ivec4(best_pos));
  } else {
    // Batch: the position is the batch index, tracked per lane.
    const int c4 = pos.z % uBlock.in_c_depth;
    acc = texelFetch(uInput, ivec3(pos.x, pos.y, c4), 0);
    acc_pos = vec4(0.0f);
    for (int n = 1; n < uBlock.in_sizes.w; ++n) {
      const vec4 v = texelFetch(
          uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + c4), 0);
      const vec4 take = extremum_take(v, acc, uBlock.greater);
      acc = mix(acc, v, take);
      acc_pos = mix(acc_pos, vec4(float(n)), take);
    }
    imageStore(uOutput, pos, ivec4(acc_pos));
  }
}
