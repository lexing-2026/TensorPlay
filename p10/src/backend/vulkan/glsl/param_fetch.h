/*
 * Per-channel parameter fetch.  A length-C parameter tensor is stored as a
 * 1D texture whose texel x axis holds the values, so lane i of channel
 * group c4 lives at texel x = c4 * 4 + i; out-of-range lanes clamp to the
 * last valid element and yield values that are never read back.
 */
vec4 param_vec(sampler3D t, const int c4, const int channels) {
  const ivec4 idx =
      min(ivec4(c4 * 4) + ivec4(0, 1, 2, 3), ivec4(channels - 1));
  return vec4(
      texelFetch(t, ivec3(idx.x, 0, 0), 0).x,
      texelFetch(t, ivec3(idx.y, 0, 0), 0).x,
      texelFetch(t, ivec3(idx.z, 0, 0), 0).x,
      texelFetch(t, ivec3(idx.w, 0, 0), 0).x);
}
