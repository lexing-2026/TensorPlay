#pragma once

#include "BFloat16.h"
#include "DType.h"
#include "Half.h"

namespace tensorplay {

// Compute-domain promotion: reduced-precision element types carry their
// arithmetic through float so intermediate rounding matches the wider
// accumulator the kernels assume, and complex values over non-floating
// element types run through a floating-point complex domain.
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
struct OpMathType<complex<Half>> {
    using type = complex<float>;
};
template <>
struct OpMathType<complex<BFloat16>> {
    using type = complex<float>;
};

template <typename T>
using opmath_type = typename OpMathType<T>::type;

}  // namespace tensorplay
