/*
 * A fold along the channel axis reads whole texels: the last texel of a channel
 * count that is not a multiple of four carries lanes naming no channel, and what
 * those lanes hold is whatever the packer or a reused allocation left there.
 *
 * Hiding those lanes behind a multiplication by a zero mask still evaluates them,
 * and zero times an infinity is a not-a-number, as is zero times a not-a-number,
 * so one padded lane that reads as either folds into every lane that was asked
 * for.  The lanes at or beyond the channel count are instead replaced by the
 * identity the fold started from, which leaves the kept lanes to the arithmetic
 * alone.
 */

// `first_lane` is the lane index the fetched texel begins at, `count` the lanes
// the shape actually names.
vec4 guarded_lanes(
    const vec4 value, const int first_lane, const int count,
    const float identity) {
  return vec4(
      first_lane + 0 < count ? value.x : identity,
      first_lane + 1 < count ? value.y : identity,
      first_lane + 2 < count ? value.z : identity,
      first_lane + 3 < count ? value.w : identity);
}
