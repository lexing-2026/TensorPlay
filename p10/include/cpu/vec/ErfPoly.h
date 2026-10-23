#pragma once

// Coefficients for the vectorized float32 error function.
//
// One form covers the whole line:
//
//   erf(|x|) = 1 - Q(t) * t * exp(-x^2),   t = 1 / (1 + kErfTailScale * |x|)
//
// with Q in ascending powers of t.  The reciprocal argument maps [0, inf)
// onto (0, 1], so a single five-coefficient polynomial serves both the
// region around zero and the saturated tail, and no lane needs a branch.
//
// Worst case over the line is 1.4e-7 in exact arithmetic and about 5e-7 once
// the coefficients and the products are rounded to float.  That bound is
// absolute: near zero the result is one minus a value close to one, so the
// cancellation caps the relative accuracy of the surviving low bits.  Every
// caller here works against an absolute tolerance and the exponential term
// carries its own few-ulp error, so the form is used as it stands.
//
// These constants live here rather than beside any single kernel so the
// vector classes and the unary-op fast paths evaluate the same polynomial.

namespace tensorplay::vecmath {

// erf(|x|) = 1 - Q(t) * t * exp(-x^2), with
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
