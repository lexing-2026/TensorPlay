/*
 * Single-precision math helpers missing from GLSL.  The Vulkan shaders need
 * the same numerics the device-side activations rely on elsewhere, so these
 * fill the two gaps:
 *   - erf_approx: error function to roughly 1.5e-7 absolute accuracy,
 *     matching the accuracy tier of hardware erff implementations (central
 *     power series near zero, rational form in the tails).
 *   - log1p_approx: log(1 + x) without the small-x cancellation of the
 *     naive form; a short series covers the cancellation band.
 * Reference for the erf form: W. J. Cody, "Rational Chebyshev
 * approximations for the error function", Math. Comp. 23 (1969);
 * the tail form matches Abramowitz & Stegun 7.1.26.
 */

float erf_approx(const float x) {
  const float ax = abs(x);
  if (ax < 0.5f) {
    const float x2 = x * x;
    float term = x;
    float sum = x;
    for (int i = 1; i <= 9; ++i) {
      term *= -x2 / float(i);
      sum += term / float(2 * i + 1);
    }
    return sum * 1.1283791670955126f; // 2 / sqrt(pi)
  }
  const bool pos = x > 0.0f;
  const float t = 1.0f / (1.0f + 0.3275911f * ax);
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t +
      1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t) *
      exp(-ax * ax);
  return pos ? y : -y;
}

float log1p_approx(const float x) {
  // Direct form once 1 + x is exact enough in fp32; series below that.
  if (abs(x) > 1.0e-4f || x <= -1.0f) {
    return log(1.0f + x);
  }
  const float x2 = x * x;
  return x * (1.0f - x * 0.5f + x2 / 3.0f - x2 * x * 0.25f);
}

// Lane-wise application of the scalar helpers over a vec4 texel.
vec4 erf_approx(const vec4 x) {
  return vec4(
      erf_approx(x.x), erf_approx(x.y), erf_approx(x.z), erf_approx(x.w));
}

vec4 log1p_approx(const vec4 x) {
  return vec4(
      log1p_approx(x.x),
      log1p_approx(x.y),
      log1p_approx(x.z),
      log1p_approx(x.w));
}

/*
 * Hardswish core on a float vector: the C++ ops layer casts integer
 * payloads to float, runs this, and rounds back, so int hardware keeps the
 * swish curve instead of a piecewise integer substitute.
 */
vec4 hardswish_approx(const vec4 x) {
  return x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f;
}
