// Complex.h -- a complex type that works on every device the CUDA backend
// targets: a generic template, reduced-width (Half/BFloat16) specializations,
// free operators, and a math-function layer that forwards to thrust/std
// depending on the compilation pass.  Storage layout is interleaved (re, im)
// and interops with std::complex and thrust::complex at the same width.
#pragma once

#include <complex>

// The device runtime and the thrust complex type are only reachable when a
// device compiler is driving the translation unit; every use of them below
// carries the same guard, so a host-only pass needs neither header.
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_runtime.h>
#include <thrust/complex.h>
#endif

#include <cmath>
#include <limits>
#include <type_traits>

#include "Half.h"
#include "BFloat16.h"

// Half.h and BFloat16.h drop their own spelling at the end of the header, so
// this one has to stand on its own -- and stay empty unless a device compiler
// is driving the translation unit.
#ifndef TP_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define TP_HOST_DEVICE __host__ __device__
#else
#define TP_HOST_DEVICE
#endif
#endif

namespace tensorplay {

// tensorplay::complex is an implementation of complex numbers that aims
// to work on all devices supported by this framework.
//
// Most of the APIs duplicate std::complex.
//
// [Note on Constructors]
//
// The constructor forms follow the corresponding standard-library forms.
//
// There are three types of constructors:
// - initializing from real and imag:
//     `constexpr complex( const T& re = T(), const T& im = T() );`
// - implicitly-declared copy constructor
// - converting constructors
//
// Converting constructors:
// - std::complex defines converting constructor between float/double/long
//   double, while we define converting constructor between float/double.
// - For these converting constructors, upcasting is implicit, downcasting
//   is explicit.
// - We also define explicit casting from std::complex/thrust::complex
//   - Note that the conversion from thrust is not constexpr.
//
// [Casting operators]
//
// std::complex does not have casting operators. We define casting operators
// casting to std::complex and thrust::complex.

template <typename T>
struct alignas(sizeof(T) * 2) complex {
  using value_type = T;

  T real_ = T(0);
  T imag_ = T(0);

  constexpr complex() = default;
  TP_HOST_DEVICE constexpr complex(const T& re, const T& im = T())
      : real_(re), imag_(im) {}
  template <typename U>
  explicit constexpr complex(const std::complex<U>& other)
      : complex(other.real(), other.imag()) {}
#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  explicit TP_HOST_DEVICE complex(const thrust::complex<U>& other)
      : real_(other.real()), imag_(other.imag()) {}
#endif

  // Use SFINAE to specialize casting constructor for complex<float> and
  // complex<double>
  template <typename U = T>
  TP_HOST_DEVICE explicit constexpr complex(
      const std::enable_if_t<std::is_same_v<U, float>, complex<double>>& other)
      : real_(other.real_), imag_(other.imag_) {}
  template <typename U = T>
  TP_HOST_DEVICE constexpr complex(
      const std::enable_if_t<std::is_same_v<U, double>, complex<float>>& other)
      : real_(other.real_), imag_(other.imag_) {}

  constexpr complex<T>& operator=(T re) {
    real_ = re;
    imag_ = 0;
    return *this;
  }

  constexpr complex<T>& operator+=(T re) {
    real_ += re;
    return *this;
  }

  constexpr complex<T>& operator-=(T re) {
    real_ -= re;
    return *this;
  }

  constexpr complex<T>& operator*=(T re) {
    real_ *= re;
    imag_ *= re;
    return *this;
  }

  constexpr complex<T>& operator/=(T re) {
    real_ /= re;
    imag_ /= re;
    return *this;
  }

  template <typename U>
  constexpr complex<T>& operator=(const complex<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }

  template <typename U>
  constexpr complex<T>& operator+=(const complex<U>& rhs) {
    real_ += rhs.real();
    imag_ += rhs.imag();
    return *this;
  }

  template <typename U>
  constexpr complex<T>& operator-=(const complex<U>& rhs) {
    real_ -= rhs.real();
    imag_ -= rhs.imag();
    return *this;
  }

  template <typename U>
  constexpr complex<T>& operator*=(const complex<U>& rhs) {
    // (a + bi) * (c + di) = (a*c - b*d) + (a * d + b * c) i
    T a = real_;
    T b = imag_;
    U c = rhs.real();
    U d = rhs.imag();
    real_ = a * c - b * d;
    imag_ = a * d + b * c;
    return *this;
  }

  template <typename U>
  TP_HOST_DEVICE complex<T>& operator/=(const complex<U>& rhs) {
    // (a + bi) / (c + di) = (ac + bd)/(c^2 + d^2) + (bc - ad)/(c^2 + d^2) i
    // the calculation below follows numpy's complex division
    T a = real_;
    T b = imag_;
    U c = rhs.real();
    U d = rhs.imag();

    T abs_c = c < 0 ? -c : c;
    T abs_d = d < 0 ? -d : d;

    if (abs_c >= abs_d) {
      if (abs_c == U(0) && abs_d == U(0)) {
        /* divide by zeros should yield a complex inf or nan */
        real_ = a / abs_c;
        imag_ = b / abs_d;
      } else {
        auto rat = d / c;
        auto scl = U(1.0) / (c + d * rat);
        real_ = (a + b * rat) * scl;
        imag_ = (b - a * rat) * scl;
      }
    } else {
      auto rat = c / d;
      auto scl = U(1.0) / (d + c * rat);
      real_ = (a * rat + b) * scl;
      imag_ = (b * rat - a) * scl;
    }
    return *this;
  }

  template <typename U>
  constexpr complex<T>& operator=(const std::complex<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  TP_HOST_DEVICE complex<T>& operator=(const thrust::complex<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }
#endif

  template <typename U>
  explicit constexpr operator std::complex<U>() const {
    return std::complex<U>(std::complex<T>(real(), imag()));
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  TP_HOST_DEVICE explicit operator thrust::complex<U>() const {
    return static_cast<thrust::complex<U>>(thrust::complex<T>(real(), imag()));
  }
#endif

  // consistent with NumPy behavior
  explicit constexpr operator bool() const {
    return real() || imag();
  }

  TP_HOST_DEVICE constexpr T real() const {
    return real_;
  }
  constexpr void real(T value) {
    real_ = value;
  }
  TP_HOST_DEVICE constexpr T imag() const {
    return imag_;
  }
  constexpr void imag(T value) {
    imag_ = value;
  }
};

namespace complex_literals {

constexpr complex<float> operator""_if(long double imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

constexpr complex<double> operator""_id(long double imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

constexpr complex<float> operator""_if(unsigned long long imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

constexpr complex<double> operator""_id(unsigned long long imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

} // namespace complex_literals

template <typename T>
constexpr complex<T> operator+(const complex<T>& val) {
  return val;
}

template <typename T>
constexpr complex<T> operator-(const complex<T>& val) {
  return complex<T>(-val.real(), -val.imag());
}

template <typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result += rhs;
}

template <typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result += rhs;
}

template <typename T>
constexpr complex<T> operator+(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs + rhs.real(), rhs.imag());
}

template <typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result -= rhs;
}

template <typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result -= rhs;
}

template <typename T>
constexpr complex<T> operator-(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs - rhs.real(), -rhs.imag());
}

template <typename T>
constexpr complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result *= rhs;
}

template <typename T>
constexpr complex<T> operator*(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result *= rhs;
}

template <typename T>
constexpr complex<T> operator*(const T& lhs, const complex<T>& rhs) {
  complex<T> result = rhs;
  return result *= lhs;
}

template <typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result /= rhs;
}

template <typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result /= rhs;
}

template <typename T>
constexpr complex<T> operator/(const T& lhs, const complex<T>& rhs) {
  complex<T> result(lhs, T());
  return result /= rhs;
}

// Define operators between integral scalars and complex. std::complex does
// not support this when T is a floating-point number. This is useful because
// it saves a lot of "static_cast" when operating a complex and an integer.
#define TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION               \
  typename std::enable_if_t<                                   \
      std::is_floating_point_v<fT> && std::is_integral_v<iT>,  \
      int> = 0

template <typename fT, typename iT, TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr complex<fT> operator+(const complex<fT>& a, const iT& b) {
  return a + static_cast<fT>(b);
}

template <typename fT, typename iT, TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr complex<fT> operator+(const iT& a, const complex<fT>& b) {
  return static_cast<fT>(a) + b;
}

template <typename fT, typename iT, TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr complex<fT> operator-(const complex<fT>& a, const iT& b) {
  return a - static_cast<fT>(b);
}

template <typename fT, typename iT, TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr complex<fT> operator-(const iT& a, const complex<fT>& b) {
  return static_cast<fT>(a) - b;
}

template <typename fT, typename iT, TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr complex<fT> operator*(const complex<fT>& a, const iT& b) {
  return a * static_cast<fT>(b);
}

template <typename fT, typename iT, TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr complex<fT> operator*(const iT& a, const complex<fT>& b) {
  return static_cast<fT>(a) * b;
}

template <typename fT, typename iT, TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr complex<fT> operator/(const complex<fT>& a, const iT& b) {
  return a / static_cast<fT>(b);
}

template <typename fT, typename iT, TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr complex<fT> operator/(const iT& a, const complex<fT>& b) {
  return static_cast<fT>(a) / b;
}

#undef TP_COMPLEX_INTEGER_OP_TEMPLATE_CONDITION

template <typename T>
constexpr bool operator==(const complex<T>& lhs, const complex<T>& rhs) {
  return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

template <typename T>
constexpr bool operator==(const complex<T>& lhs, const T& rhs) {
  return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <typename T>
constexpr bool operator==(const T& lhs, const complex<T>& rhs) {
  return (lhs == rhs.real()) && (T() == rhs.imag());
}

template <typename T>
constexpr bool operator!=(const complex<T>& lhs, const complex<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
constexpr bool operator!=(const complex<T>& lhs, const T& rhs) {
  return !(lhs == rhs);
}

template <typename T>
constexpr bool operator!=(const T& lhs, const complex<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
std::basic_ostream<char>& operator<<(std::basic_ostream<char>& os,
                                     const complex<T>& x) {
  return (os << static_cast<std::complex<T>>(x));
}

template <typename T>
std::basic_istream<char>& operator>>(std::basic_istream<char>& is,
                                     complex<T>& x) {
  std::complex<T> tmp;
  is >> tmp;
  x = tmp;
  return is;
}

template <typename T>
TP_HOST_DEVICE complex<T> polar(const T& r, const T& theta = T()) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<complex<T>>(thrust::polar(r, theta));
#else
  // std::polar() requires r >= 0, so spell out the explicit implementation to
  // avoid a branch.
  return complex<T>(r * std::cos(theta), r * std::sin(theta));
#endif
}

// Reduced-width specializations: the scalar wrappers are not constexpr and
// carry no arithmetic operators, so arithmetic widens to float and rounds
// the result back to the storage width.  Layout stays interleaved (re, im)
// at 4-byte total alignment so device storage interops with the full-width
// kernels and with serialized buffers.
template <>
struct alignas(4) complex<Half> {
  using value_type = Half;

  Half real_;
  Half imag_;

  // Constructors
  complex() = default;
  // Half constructor is not constexpr so the following constructor can't
  // be constexpr
  TP_HOST_DEVICE explicit inline complex(const Half& real,
                                                 const Half& imag)
      : real_(real), imag_(imag) {}
  TP_HOST_DEVICE inline complex(const complex<float>& value)
      : real_(value.real()), imag_(value.imag()) {}

  // Conversion operator
  TP_HOST_DEVICE inline operator complex<float>() const {
    return {real_, imag_};
  }

  TP_HOST_DEVICE constexpr Half real() const {
    return real_;
  }
  TP_HOST_DEVICE constexpr Half imag() const {
    return imag_;
  }

  TP_HOST_DEVICE complex<Half>& operator+=(const complex<Half>& other) {
    real_ = static_cast<float>(real_) + static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) + static_cast<float>(other.imag_);
    return *this;
  }

  TP_HOST_DEVICE complex<Half>& operator-=(const complex<Half>& other) {
    real_ = static_cast<float>(real_) - static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) - static_cast<float>(other.imag_);
    return *this;
  }

  TP_HOST_DEVICE complex<Half>& operator*=(const complex<Half>& other) {
    auto a = static_cast<float>(real_);
    auto b = static_cast<float>(imag_);
    auto c = static_cast<float>(other.real());
    auto d = static_cast<float>(other.imag());
    real_ = a * c - b * d;
    imag_ = a * d + b * c;
    return *this;
  }
};

template <>
struct alignas(4) complex<BFloat16> {
  using value_type = BFloat16;

  BFloat16 real_;
  BFloat16 imag_;

  // Constructors
  complex() = default;
  // BFloat16 constructor is not constexpr so the following constructor can't
  // be constexpr
  TP_HOST_DEVICE explicit inline complex(
      const BFloat16& real,
      const BFloat16& imag)
      : real_(real), imag_(imag) {}
  TP_HOST_DEVICE inline complex(const complex<float>& value)
      : real_(value.real()), imag_(value.imag()) {}

  // Conversion operator
  TP_HOST_DEVICE inline operator complex<float>() const {
    return {real_, imag_};
  }

  TP_HOST_DEVICE constexpr BFloat16 real() const {
    return real_;
  }
  TP_HOST_DEVICE constexpr BFloat16 imag() const {
    return imag_;
  }

  TP_HOST_DEVICE complex<BFloat16>& operator+=(
      const complex<BFloat16>& other) {
    real_ = static_cast<float>(real_) + static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) + static_cast<float>(other.imag_);
    return *this;
  }

  TP_HOST_DEVICE complex<BFloat16>& operator-=(
      const complex<BFloat16>& other) {
    real_ = static_cast<float>(real_) - static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) - static_cast<float>(other.imag_);
    return *this;
  }

  TP_HOST_DEVICE complex<BFloat16>& operator*=(
      const complex<BFloat16>& other) {
    auto a = static_cast<float>(real_);
    auto b = static_cast<float>(imag_);
    auto c = static_cast<float>(other.real());
    auto d = static_cast<float>(other.imag());
    real_ = a * c - b * d;
    imag_ = a * d + b * c;
    return *this;
  }
};

// Extract double from complex<double>; is identity otherwise.
template <typename T>
struct scalar_value_type {
  using type = T;
};
template <typename T>
struct scalar_value_type<complex<T>> {
  using type = T;
};

} // namespace tensorplay

namespace std {

template <typename T>
class numeric_limits<tensorplay::complex<T>> : public numeric_limits<T> {};

template <typename T>
bool isnan(const tensorplay::complex<T>& v) {
  return std::isnan(v.real()) || std::isnan(v.imag());
}

template <typename T>
TP_HOST_DEVICE T real(const tensorplay::complex<T>& z) {
  return z.real();
}

template <typename T>
TP_HOST_DEVICE T imag(const tensorplay::complex<T>& z) {
  return z.imag();
}

template <typename T>
TP_HOST_DEVICE T abs(const tensorplay::complex<T>& z) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return thrust::abs(static_cast<thrust::complex<T>>(z));
#else
  return std::abs(static_cast<std::complex<T>>(z));
#endif
}

template <typename T>
TP_HOST_DEVICE T arg(const tensorplay::complex<T>& z) {
  return std::atan2(std::imag(z), std::real(z));
}

template <typename T>
constexpr T norm(const tensorplay::complex<T>& z) {
  return z.real() * z.real() + z.imag() * z.imag();
}

template <typename T>
constexpr tensorplay::complex<T> conj(const tensorplay::complex<T>& z) {
  return tensorplay::complex<T>(z.real(), -z.imag());
}

} // namespace std

// ---------------------------------------------------------------------------
// math functions
// ---------------------------------------------------------------------------

namespace tensorplay_complex_math {

// The math layer forwards to the complex library of whichever toolchain is
// driving the pass: a device compiler supplies thrust, a host-only pass the
// standard library.  Both provide the same function set over the same
// interleaved layout, and the cast operators on tensorplay::complex convert
// to either one.
#if defined(__CUDACC__) || defined(__HIPCC__)
namespace tp_cplx = thrust;
#else
namespace tp_cplx = std;
#endif

// Exponential functions

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> exp(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::exp(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> log(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::log(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> log10(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::log10(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> log2(
    const tensorplay::complex<T>& x) {
  const tensorplay::complex<T> log2 = tensorplay::complex<T>(::log(2.0), 0.0);
  return tensorplay_complex_math::log(x) / log2;
}

// Power functions

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> sqrt(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::sqrt(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> pow(
    const tensorplay::complex<T>& x,
    const tensorplay::complex<T>& y) {
  return static_cast<tensorplay::complex<T>>(tp_cplx::pow(
      static_cast<tp_cplx::complex<T>>(x), static_cast<tp_cplx::complex<T>>(y)));
}

// Regression in ROCm 7.2: the thrust pow path loses single-rounding on the
// float specialization, so pre-scale with an FMA-formed multiplication.
#if defined(__HIPCC__)
namespace detail {
// FMA-aware complex multiplication for float precision on AMD GPUs.
// This prevents SLP vectorizer from breaking FMA formation, which causes
// numerical precision loss in complex arithmetic.
TP_HOST_DEVICE inline tp_cplx::complex<float> complex_mul_fma(
    tp_cplx::complex<float> a,
    tp_cplx::complex<float> b) {
  // Complex multiplication: (a.r + a.i*i) * (b.r + b.i*i)
  // = (a.r*b.r - a.i*b.i) + (a.r*b.i + a.i*b.r)*i
  // Using __builtin_fmaf ensures FMA at source level:
  // real: a.r*b.r + (-(a.i*b.i)) = FMA(a.r, b.r, -(a.i*b.i))
  // imag: a.i*b.r + a.r*b.i = FMA(a.r, b.i, a.i*b.r)
  float real_part = __builtin_fmaf(a.real(), b.real(), -(a.imag() * b.imag()));
  float imag_part = __builtin_fmaf(a.real(), b.imag(), a.imag() * b.real());
  return tp_cplx::complex<float>(real_part, imag_part);
}
} // namespace detail

template <>
TP_HOST_DEVICE inline tensorplay::complex<float> pow(
    const tensorplay::complex<float>& x,
    const tensorplay::complex<float>& y) {
  auto log_x = tp_cplx::log(static_cast<tp_cplx::complex<float>>(x));
  auto y_log_x =
      detail::complex_mul_fma(static_cast<tp_cplx::complex<float>>(y), log_x);
  return static_cast<tensorplay::complex<float>>(tp_cplx::exp(y_log_x));
}
#endif

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> pow(
    const tensorplay::complex<T>& x,
    const T& y) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::pow(static_cast<tp_cplx::complex<T>>(x), y));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> pow(
    const T& x,
    const tensorplay::complex<T>& y) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::pow(x, static_cast<tp_cplx::complex<T>>(y)));
}

template <typename T, typename U>
TP_HOST_DEVICE inline tensorplay::complex<decltype(T() * U())> pow(
    const tensorplay::complex<T>& x,
    const tensorplay::complex<U>& y) {
  return static_cast<tensorplay::complex<T>>(tp_cplx::pow(
      static_cast<tp_cplx::complex<T>>(x), static_cast<tp_cplx::complex<T>>(y)));
}

template <typename T, typename U>
TP_HOST_DEVICE inline tensorplay::complex<decltype(T() * U())> pow(
    const tensorplay::complex<T>& x,
    const U& y) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::pow(static_cast<tp_cplx::complex<T>>(x), y));
}

template <typename T, typename U>
TP_HOST_DEVICE inline tensorplay::complex<decltype(T() * U())> pow(
    const T& x,
    const tensorplay::complex<U>& y) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::pow(x, static_cast<tp_cplx::complex<T>>(y)));
}

// Trigonometric functions

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> sin(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::sin(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> cos(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::cos(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> tan(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::tan(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> asin(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::asin(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> acos(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::acos(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> atan(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::atan(static_cast<tp_cplx::complex<T>>(x)));
}

// Hyperbolic functions

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> sinh(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::sinh(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> cosh(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::cosh(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> tanh(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::tanh(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> asinh(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::asinh(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> acosh(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::acosh(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> atanh(
    const tensorplay::complex<T>& x) {
  return static_cast<tensorplay::complex<T>>(
      tp_cplx::atanh(static_cast<tp_cplx::complex<T>>(x)));
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> log1p(
    const tensorplay::complex<T>& z) {
#if defined(__APPLE__) || defined(__MACOSX) || defined(__CUDACC__) || \
    defined(__HIPCC__) || defined(__SYCL_DEVICE_ONLY__)
  // log1p(z) = log(1 + z)
  // Let's define 1 + z = r * e ^ (i * a), then we have
  // log(r * e ^ (i * a)) = log(r) + i * a
  // With z = x + iy, the term r can be written as
  // r = ((1 + x) ^ 2 + y ^ 2) ^ 0.5
  //   = (1 + x ^ 2 + 2 * x + y ^ 2) ^ 0.5
  // So, log(r) is
  // log(r) = 0.5 * log(1 + x ^ 2 + 2 * x + y ^ 2)
  //        = 0.5 * log1p(x * (x + 2) + y ^ 2)
  // we need to use the expression only on certain condition to avoid overflow
  // and underflow from `(x * (x + 2) + y ^ 2)`
  T x = z.real();
  T y = z.imag();
  T zabs = std::abs(z);
  T theta = std::atan2(y, x + T(1));
  if (zabs < 0.5) {
    T r = x * (T(2) + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {T(0.5) * std::log1p(r), theta};
  } else {
    T z0 = std::hypot(x + 1, y);
    return {std::log(z0), theta};
  }
#else
  // CPU path
  tensorplay::complex<T> u = z + T(1);
  if (u == T(1)) {
    return z;
  } else {
    auto log_u = log(u);
    if (u - T(1) == z) {
      return log_u;
    }
    return log_u * (z / (u - T(1)));
  }
#endif
}

template <typename T>
TP_HOST_DEVICE inline tensorplay::complex<T> expm1(
    const tensorplay::complex<T>& z) {
  // expm1(z) = exp(z) - 1
  // Define z = x + i * y
  // f = e ^ (x + i * y) - 1
  //   = e ^ x * e ^ (i * y) - 1
  //   = (e ^ x * cos(y) - 1) + i * (e ^ x * sin(y))
  //   = (e ^ x - 1) * cos(y) - (1 - cos(y)) + i * e ^ x * sin(y)
  //   = expm1(x) * cos(y) - 2 * sin(y / 2) ^ 2 + i * e ^ x * sin(y)
  T x = z.real();
  T y = z.imag();
  T a = std::sin(y / 2);
  T er = std::expm1(x) * std::cos(y) - T(2) * a * a;
  T ei = std::exp(x) * std::sin(y);
  return {er, ei};
}

} // namespace tensorplay_complex_math

namespace tensorplay {

using tensorplay_complex_math::acos;
using tensorplay_complex_math::acosh;
using tensorplay_complex_math::asin;
using tensorplay_complex_math::asinh;
using tensorplay_complex_math::atan;
using tensorplay_complex_math::atanh;
using tensorplay_complex_math::cos;
using tensorplay_complex_math::cosh;
using tensorplay_complex_math::exp;
using tensorplay_complex_math::expm1;
using tensorplay_complex_math::log;
using tensorplay_complex_math::log10;
using tensorplay_complex_math::log1p;
using tensorplay_complex_math::log2;
using tensorplay_complex_math::pow;
using tensorplay_complex_math::sin;
using tensorplay_complex_math::sinh;
using tensorplay_complex_math::sqrt;
using tensorplay_complex_math::tan;
using tensorplay_complex_math::tanh;

// Host kernels carry their complex payloads as std::complex; the power answers
// for that spelling too, so those call sites need no conversion of their own.
// It stays inside this namespace rather than the math layer, which is
// re-exported into std where it would collide with std::pow.
template <typename T>
TP_HOST_DEVICE inline std::complex<T> pow(
    const std::complex<T>& x,
    const std::complex<T>& y) {
  return static_cast<std::complex<T>>(
      tensorplay_complex_math::pow(static_cast<tensorplay::complex<T>>(x),
                                   static_cast<tensorplay::complex<T>>(y)));
}

template <typename T>
TP_HOST_DEVICE inline std::complex<T> pow(
    const std::complex<T>& x,
    const T& y) {
  return static_cast<std::complex<T>>(
      tensorplay_complex_math::pow(static_cast<tensorplay::complex<T>>(x), y));
}

} // namespace tensorplay

namespace std {

using tensorplay_complex_math::acos;
using tensorplay_complex_math::acosh;
using tensorplay_complex_math::asin;
using tensorplay_complex_math::asinh;
using tensorplay_complex_math::atan;
using tensorplay_complex_math::atanh;
using tensorplay_complex_math::cos;
using tensorplay_complex_math::cosh;
using tensorplay_complex_math::exp;
using tensorplay_complex_math::expm1;
using tensorplay_complex_math::log;
using tensorplay_complex_math::log10;
using tensorplay_complex_math::log1p;
using tensorplay_complex_math::log2;
using tensorplay_complex_math::pow;
using tensorplay_complex_math::sin;
using tensorplay_complex_math::sinh;
using tensorplay_complex_math::sqrt;
using tensorplay_complex_math::tan;
using tensorplay_complex_math::tanh;

} // namespace std
