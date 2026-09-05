#pragma once

#include "BFloat16.h"
#include "DType.h"
#include "Half.h"

#include <complex>

namespace tensorplay {

// Compute-domain promotion: reduced-precision element types carry their
// arithmetic through float so intermediate rounding matches the wider
// accumulator the kernels assume, and complex values over non-floating
// element types run through a floating-point complex domain (the standard
// complex templates require a real floating-point value_type).
template <typename scalar_t>
struct OpMathType {
    using type = scalar_t;
};
template <>
struct OpMathType<Half> {
    using type = float;
};
template <>
struct OpMathType<BFloat16> {
    using type = float;
};
template <>
struct OpMathType<Float8_e4m3fn> {
    using type = float;
};
template <>
struct OpMathType<Float8_e5m2> {
    using type = float;
};
template <>
struct OpMathType<Float8_e4m3fnuz> {
    using type = float;
};
template <>
struct OpMathType<Float8_e5m2fnuz> {
    using type = float;
};
template <>
struct OpMathType<Float8_e8m0fnu> {
    using type = float;
};
template <>
struct OpMathType<std::complex<Half>> {
    using type = std::complex<float>;
};
template <>
struct OpMathType<std::complex<BFloat16>> {
    using type = std::complex<float>;
};

template <typename T>
using opmath_type = typename OpMathType<T>::type;

}  // namespace tensorplay
