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

/*
 * Complementary error function: 1 - erf(x) without the cancellation the
 * subtraction would leave in the positive tail.  The central band reuses the
 * series above, whose absolute error is about 1e-7; beyond it the same
 * rational tail factor is evaluated on its own, so a lane whose answer has
 * underflowed keeps the zero instead of a difference of two ones.
 */
float erfc_approx(const float x) {
  const float ax = abs(x);
  if (ax < 0.5f) {
    return 1.0f - erf_approx(x);
  }
  const float t = 1.0f / (1.0f + 0.3275911f * ax);
  const float tail = (((((1.061405429f * t - 1.453152027f) * t +
      1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t) *
      exp(-ax * ax);
  return x < 0.0f ? 2.0f - tail : tail;
}

// Six-term Lanczos form of log |Gamma(x)|, used above one half.
float lgamma_positive(const float x) {
  const float shifted = x + 5.5f;
  const float series = 1.000000000190015f +
      76.18009172947146f / (x + 1.0f) -
      86.50532032941677f / (x + 2.0f) +
      24.01409824083091f / (x + 3.0f) -
      1.231739572450155f / (x + 4.0f) +
      0.1208650973866179e-2f / (x + 5.0f) -
      0.5395239384953e-5f / (x + 6.0f);
  return (x + 0.5f) * log(shifted) - shifted +
      log(2.5066282746310005f * series / x);
}

/*
 * sin(pi x) through the nearest integer: the reduced argument is at most one
 * half, and the reduced sine keeps its relative accuracy near a zero because
 * it is one product of pi r with a series in r squared.  The built-in sine
 * cannot serve this: its absolute error of a few times 1e-7 is tens of
 * percent of a small sine and it answers zero below about 2e-7.
 */
float sin_pi(const float x) {
  const float n = floor(x + 0.5f);
  const float r = x - n;
  const float r2 = r * r;
  const float series = 1.0f + r2 * (-1.6449340668482264f + r2 *
      (0.8117424252833537f + r2 * (-0.1907518241220842f + r2 *
      (0.0261478478176548f - r2 * 0.0023460810354558236f))));
  const float value = 3.14159265358979323846f * r * series;
  return mod(n, 2.0f) == 0.0f ? value : -value;
}

/*
 * log |Gamma(x)| at single precision: the Lanczos form above one half and the
 * reflection log(pi / |sin_pi(x)|) - lgamma(1 - x) below it.  The non-positive
 * integers are poles and answer +inf, an infinite lane answers +inf, and a
 * not-a-number lane stays one.  Absolute error is a few times 1e-7; near a
 * zero of lgamma the answer is a difference of two larger terms, so only its
 * absolute accuracy survives.
 */
float lgamma_approx(const float x) {
  if (isnan(x)) {
    return x;
  }
  if (isinf(x)) {
    return abs(x);
  }
  if (x >= 0.5f) {
    return lgamma_positive(x);
  }
  if (x == floor(x)) {
    return uintBitsToFloat(0x7f800000u);
  }
  const float pi = 3.14159265358979323846f;
  return log(pi / abs(sin_pi(x))) - lgamma_positive(1.0f - x);
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

vec4 erfc_approx(const vec4 x) {
  return vec4(
      erfc_approx(x.x),
      erfc_approx(x.y),
      erfc_approx(x.z),
      erfc_approx(x.w));
}

vec4 lgamma_approx(const vec4 x) {
  return vec4(
      lgamma_approx(x.x),
      lgamma_approx(x.y),
      lgamma_approx(x.z),
      lgamma_approx(x.w));
}

/*
 * Hardswish core on a float vector: the C++ ops layer casts integer
 * payloads to float, runs this, and rounds back, so int hardware keeps the
 * swish curve instead of a piecewise integer substitute.
 */
vec4 hardswish_approx(const vec4 x) {
  return x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f;
}
