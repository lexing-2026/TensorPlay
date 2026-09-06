#pragma once

/// Defines the Half type (IEEE half-precision floating point, fp16).
/// Arithmetic is performed by converting to float and back, which is the

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#if defined(__CUDACC__)
#define TP_HOST_DEVICE __host__ __device__
#else
#define TP_HOST_DEVICE
#endif

// std::bit_cast needs C++20; nvcc is often invoked under an older -std via
// CMake dialect machinery.  __builtin_bit_cast (GCC 11 / Clang 13 / nvcc 12+)
// carries identical semantics with no dialect requirement.
#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
#define TP_BIT_CAST(dst_t, src) __builtin_bit_cast(dst_t, (src))
#else
#define TP_BIT_CAST(dst_t, src) std::bit_cast<dst_t>(src)
#endif

namespace tensorplay {

struct alignas(2) Half {
  uint16_t x;

  TP_HOST_DEVICE Half() = default;

  struct from_bits_t {};
  static constexpr TP_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr TP_HOST_DEVICE Half(unsigned short bits, from_bits_t)
      : x(bits) {}

  /* implicit */ inline TP_HOST_DEVICE Half(float value);
  inline TP_HOST_DEVICE operator float() const;

  /* implicit */ inline TP_HOST_DEVICE Half(double value)
      : Half(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE Half(int value)
      : Half(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE Half(long value)
      : Half(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE Half(long long value)
      : Half(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE Half(unsigned int value)
      : Half(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE Half(unsigned long value)
      : Half(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE Half(unsigned long long value)
      : Half(static_cast<float>(value)) {}

  inline TP_HOST_DEVICE Half& operator+=(const Half& b) {
    *this = static_cast<float>(*this) + static_cast<float>(b);
    return *this;
  }
  inline TP_HOST_DEVICE Half& operator-=(const Half& b) {
    *this = static_cast<float>(*this) - static_cast<float>(b);
    return *this;
  }
  inline TP_HOST_DEVICE Half& operator*=(const Half& b) {
    *this = static_cast<float>(*this) * static_cast<float>(b);
    return *this;
  }
  inline TP_HOST_DEVICE Half& operator/=(const Half& b) {
    *this = static_cast<float>(*this) / static_cast<float>(b);
    return *this;
  }
};

// --- bit-level conversion helpers (host fallback) ---

namespace detail {

inline uint16_t float_to_half_bits(float f) {
  uint32_t bits = TP_BIT_CAST(uint32_t, f);
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = ((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFFu;

  if (((bits >> 23) & 0xFFu) == 0xFFu) {
    // Inf or NaN: preserve exponent, keep at least one mantissa bit
    uint16_t half = static_cast<uint16_t>(sign | 0x7C00u);
    if (mantissa != 0) {
      half |= (mantissa >> 13) | 0x200u;
    }
    return half;
  }
  if (exponent <= 0) {
    // Subnormal or zero
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t rem = mantissa & ((1u << shift) - 1u);
    mantissa >>= shift;
    if (rem > (1u << (shift - 1)) ||
        (rem == (1u << (shift - 1)) && (mantissa & 1u))) {
      mantissa++;
    }
    if (mantissa == 0x400u) {
      return static_cast<uint16_t>(sign | 0x0400u); // rounds up to min normal
    }
    return static_cast<uint16_t>(sign | mantissa);
  }
  if (exponent >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00u); // overflow -> Inf
  }
  uint32_t half_mantissa = mantissa >> 13;
  uint32_t rem = mantissa & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half_mantissa & 1u))) {
    half_mantissa++;
    if (half_mantissa == 0x400u) {
      half_mantissa = 0;
      exponent++;
      if (exponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);
      }
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | half_mantissa);
}

inline float half_to_float_bits(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;

  if (exponent == 0) {
    if (mantissa == 0) {
      return TP_BIT_CAST(float, sign);
    }
    // Subnormal: normalize
    uint32_t e = 127 - 15 + 1;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      e--;
    }
    mantissa &= 0x3FFu;
    return TP_BIT_CAST(float, sign | (e << 23) | (mantissa << 13));
  }
  if (exponent == 0x1F) {
    return TP_BIT_CAST(float, sign | 0x7F800000u | (mantissa << 13));
  }
  return TP_BIT_CAST(float, sign | ((exponent - 15 + 127) << 23) | (mantissa << 13));
}

} // namespace detail

// The out-of-class definitions repeat TP_HOST_DEVICE so the annotation set
// matches the in-class declarations on every device compiler (clang's HIP
// frontend rejects a bare __host__ definition overloading a __host__
// __device__ declaration; nvcc merely tolerates the mismatch).
inline TP_HOST_DEVICE Half::Half(float value) {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
  x = __half_as_short(__float2half(value));
#else
  x = detail::float_to_half_bits(value);
#endif
}

inline TP_HOST_DEVICE Half::operator float() const {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
  return __half2float(*reinterpret_cast<const __half*>(&x));
#else
  return detail::half_to_float_bits(x);
#endif
}

inline TP_HOST_DEVICE Half operator-(const Half& a) { return Half(-static_cast<float>(a)); }
inline TP_HOST_DEVICE Half operator+(const Half& a, const Half& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}
inline TP_HOST_DEVICE Half operator-(const Half& a, const Half& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}
inline TP_HOST_DEVICE Half operator*(const Half& a, const Half& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}
inline TP_HOST_DEVICE Half operator/(const Half& a, const Half& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline TP_HOST_DEVICE bool operator==(const Half& a, const Half& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator!=(const Half& a, const Half& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator<(const Half& a, const Half& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator<=(const Half& a, const Half& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator>(const Half& a, const Half& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator>=(const Half& a, const Half& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

// Mixed-type overloads used by generic kernels that mix Half with
// double/int64_t scalars. Without these, the implicit conversions in both
// directions make the built-in operator ambiguous.
inline TP_HOST_DEVICE double operator+(Half a, double b) { return static_cast<double>(a) + b; }
inline TP_HOST_DEVICE double operator+(double a, Half b) { return a + static_cast<double>(b); }
inline TP_HOST_DEVICE double operator-(Half a, double b) { return static_cast<double>(a) - b; }
inline TP_HOST_DEVICE double operator-(double a, Half b) { return a - static_cast<double>(b); }
inline TP_HOST_DEVICE double operator*(Half a, double b) { return static_cast<double>(a) * b; }
inline TP_HOST_DEVICE double operator*(double a, Half b) { return a * static_cast<double>(b); }
inline TP_HOST_DEVICE double operator/(Half a, double b) { return static_cast<double>(a) / b; }
inline TP_HOST_DEVICE double operator/(double a, Half b) { return a / static_cast<double>(b); }
inline TP_HOST_DEVICE float operator+(Half a, float b) { return static_cast<float>(a) + b; }
inline TP_HOST_DEVICE float operator+(float a, Half b) { return a + static_cast<float>(b); }
inline TP_HOST_DEVICE float operator-(Half a, float b) { return static_cast<float>(a) - b; }
inline TP_HOST_DEVICE float operator-(float a, Half b) { return a - static_cast<float>(b); }
inline TP_HOST_DEVICE float operator*(Half a, float b) { return static_cast<float>(a) * b; }
inline TP_HOST_DEVICE float operator*(float a, Half b) { return a * static_cast<float>(b); }
inline TP_HOST_DEVICE float operator/(Half a, float b) { return static_cast<float>(a) / b; }
inline TP_HOST_DEVICE float operator/(float a, Half b) { return a / static_cast<float>(b); }
inline TP_HOST_DEVICE Half operator+(const Half& a, int64_t b) { return a + Half(b); }
inline TP_HOST_DEVICE Half operator+(int64_t a, const Half& b) { return Half(a) + b; }
inline TP_HOST_DEVICE Half operator-(const Half& a, int64_t b) { return a - Half(b); }
inline TP_HOST_DEVICE Half operator-(int64_t a, const Half& b) { return Half(a) - b; }
inline TP_HOST_DEVICE Half operator*(const Half& a, int64_t b) { return a * Half(b); }
inline TP_HOST_DEVICE Half operator*(int64_t a, const Half& b) { return Half(a) * b; }
inline TP_HOST_DEVICE Half operator/(const Half& a, int64_t b) { return a / Half(b); }
inline TP_HOST_DEVICE Half operator/(int64_t a, const Half& b) { return Half(a) / b; }
inline TP_HOST_DEVICE bool operator>(const Half& a, int64_t b) { return static_cast<float>(a) > static_cast<float>(b); }
inline TP_HOST_DEVICE bool operator<(const Half& a, int64_t b) { return static_cast<float>(a) < static_cast<float>(b); }

inline std::ostream& operator<<(std::ostream& out, const Half& value) {
  out << static_cast<float>(value);
  return out;
}

} // namespace tensorplay

namespace std {

template <>
class numeric_limits<tensorplay::Half> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr tensorplay::Half min() {
    return tensorplay::Half(0x0400, tensorplay::Half::from_bits());
  }
  static constexpr tensorplay::Half lowest() {
    return tensorplay::Half(0xFBFF, tensorplay::Half::from_bits());
  }
  static constexpr tensorplay::Half max() {
    return tensorplay::Half(0x7BFF, tensorplay::Half::from_bits());
  }
  static constexpr tensorplay::Half epsilon() {
    return tensorplay::Half(0x1400, tensorplay::Half::from_bits());
  }
  static constexpr tensorplay::Half round_error() {
    return tensorplay::Half(0x3800, tensorplay::Half::from_bits());
  }
  static constexpr tensorplay::Half infinity() {
    return tensorplay::Half(0x7C00, tensorplay::Half::from_bits());
  }
  static constexpr tensorplay::Half quiet_NaN() {
    return tensorplay::Half(0x7E00, tensorplay::Half::from_bits());
  }
  static constexpr tensorplay::Half signaling_NaN() {
    return tensorplay::Half(0x7D00, tensorplay::Half::from_bits());
  }
  static constexpr tensorplay::Half denorm_min() {
    return tensorplay::Half(0x0001, tensorplay::Half::from_bits());
  }
};

} // namespace std

#undef TP_HOST_DEVICE
