#pragma once

// Coefficients for the vectorized float32 error function.
//
// Two forms cover the line and the kernels select between them per lane.
// Near zero the Maclaurin series is evaluated directly: the tail form ends
// in ``1 - r``, and for a small argument that subtraction cancels away the
// result's low bits, so a tail-only kernel is accurate in absolute terms and
// thousands of ulp out in relative terms.  Away from zero the series would
// need many more terms than the tail form, which reaches float precision
// with five coefficients and one exponential.
//
// Worst case over the whole line is under three float ulp, and the two forms
// meet without a step at the split point.  These constants live here rather
// than beside either kernel so the vector class and the unary-op fast paths
// evaluate the same function.

namespace tensorplay::vecmath {

// erf(x) = x * P(x^2) for |x| < kErfSplit, P in ascending powers of x^2.
inline constexpr float kErfSeries[7] = {
    1.1283791670955126e+00f,
    -3.7612638903183752e-01f,
    1.1283791670955126e-01f,
    -2.6866170645131252e-02f,
    5.2239776254421878e-03f,
    -8.5440360144887751e-04f,
    1.2055332981789664e-04f,
};

inline constexpr float kErfSplit = 0.7f;

// erf(|x|) = 1 - Q(t) * t * exp(-x^2) elsewhere, with
// t = 1 / (1 + kErfTailScale * |x|) and Q in ascending powers of t.
inline constexpr float kErfTailScale = 0.3275911f;

inline constexpr float kErfTail[5] = {
    0.254829592f,
    -0.284496736f,
    1.421413741f,
    -1.453152027f,
    1.061405429f,
};

}  // namespace tensorplay::vecmath
