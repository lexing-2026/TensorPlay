#pragma once

/// Defines the BFloat16 type (bfloat16: 1 sign, 8 exponent, 7 mantissa bits).
/// Arithmetic is performed by converting to float and back.

#include <bit>
#include <cstdint>
#include <cstring>
#include <ostream>

#ifdef __CUDACC__
#include <cuda_bf16.h>
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

struct alignas(2) BFloat16 {
  uint16_t x;

  TP_HOST_DEVICE BFloat16() = default;

  struct from_bits_t {};
  static constexpr TP_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr TP_HOST_DEVICE BFloat16(unsigned short bits, from_bits_t)
      : x(bits) {}

  /* implicit */ inline TP_HOST_DEVICE BFloat16(float value);
  inline TP_HOST_DEVICE operator float() const;

  /* implicit */ inline TP_HOST_DEVICE BFloat16(double value)
      : BFloat16(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE BFloat16(int value)
      : BFloat16(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE BFloat16(long value)
      : BFloat16(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE BFloat16(long long value)
      : BFloat16(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE BFloat16(unsigned int value)
      : BFloat16(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE BFloat16(unsigned long value)
      : BFloat16(static_cast<float>(value)) {}
  /* implicit */ inline TP_HOST_DEVICE BFloat16(unsigned long long value)
      : BFloat16(static_cast<float>(value)) {}

  inline TP_HOST_DEVICE BFloat16& operator+=(const BFloat16& b) {
    *this = static_cast<float>(*this) + static_cast<float>(b);
    return *this;
  }
  inline TP_HOST_DEVICE BFloat16& operator-=(const BFloat16& b) {
    *this = static_cast<float>(*this) - static_cast<float>(b);
    return *this;
  }
  inline TP_HOST_DEVICE BFloat16& operator*=(const BFloat16& b) {
    *this = static_cast<float>(*this) * static_cast<float>(b);
    return *this;
  }
  inline TP_HOST_DEVICE BFloat16& operator/=(const BFloat16& b) {
    *this = static_cast<float>(*this) / static_cast<float>(b);
    return *this;
  }
};

namespace detail {

// Round-to-nearest-even conversion from f32 to bf16 (host fallback)
inline uint16_t float_to_bfloat16_bits(float f) {
  uint32_t bits = TP_BIT_CAST(uint32_t, f);
  uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}

inline float bfloat16_to_float_bits(uint16_t b) {
  return TP_BIT_CAST(float, static_cast<uint32_t>(b) << 16);
}

} // namespace detail

inline BFloat16::BFloat16(float value) {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
  x = __bfloat16_as_ushort(__float2bfloat16(value));
#else
  x = detail::float_to_bfloat16_bits(value);
#endif
}

inline BFloat16::operator float() const {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
#else
  return detail::bfloat16_to_float_bits(x);
#endif
}

inline TP_HOST_DEVICE BFloat16 operator-(const BFloat16& a) { return BFloat16(-static_cast<float>(a)); }
inline TP_HOST_DEVICE BFloat16 operator+(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}
inline TP_HOST_DEVICE BFloat16 operator-(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}
inline TP_HOST_DEVICE BFloat16 operator*(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}
inline TP_HOST_DEVICE BFloat16 operator/(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline TP_HOST_DEVICE bool operator==(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator!=(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) != static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator<(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator<=(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator>(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
inline TP_HOST_DEVICE bool operator>=(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

// Mixed-type overloads used by generic kernels that mix BFloat16 with
// double/int64_t scalars.
inline TP_HOST_DEVICE double operator+(BFloat16 a, double b) { return static_cast<double>(a) + b; }
inline TP_HOST_DEVICE double operator+(double a, BFloat16 b) { return a + static_cast<double>(b); }
inline TP_HOST_DEVICE double operator-(BFloat16 a, double b) { return static_cast<double>(a) - b; }
inline TP_HOST_DEVICE double operator-(double a, BFloat16 b) { return a - static_cast<double>(b); }
inline TP_HOST_DEVICE double operator*(BFloat16 a, double b) { return static_cast<double>(a) * b; }
inline TP_HOST_DEVICE double operator*(double a, BFloat16 b) { return a * static_cast<double>(b); }
inline TP_HOST_DEVICE double operator/(BFloat16 a, double b) { return static_cast<double>(a) / b; }
inline TP_HOST_DEVICE double operator/(double a, BFloat16 b) { return a / static_cast<double>(b); }
inline TP_HOST_DEVICE float operator+(BFloat16 a, float b) { return static_cast<float>(a) + b; }
inline TP_HOST_DEVICE float operator+(float a, BFloat16 b) { return a + static_cast<float>(b); }
inline TP_HOST_DEVICE float operator-(BFloat16 a, float b) { return static_cast<float>(a) - b; }
inline TP_HOST_DEVICE float operator-(float a, BFloat16 b) { return a - static_cast<float>(b); }
inline TP_HOST_DEVICE float operator*(BFloat16 a, float b) { return static_cast<float>(a) * b; }
inline TP_HOST_DEVICE float operator*(float a, BFloat16 b) { return a * static_cast<float>(b); }
inline TP_HOST_DEVICE float operator/(BFloat16 a, float b) { return static_cast<float>(a) / b; }
inline TP_HOST_DEVICE float operator/(float a, BFloat16 b) { return a / static_cast<float>(b); }
inline TP_HOST_DEVICE BFloat16 operator+(const BFloat16& a, int64_t b) { return a + BFloat16(b); }
inline TP_HOST_DEVICE BFloat16 operator+(int64_t a, const BFloat16& b) { return BFloat16(a) + b; }
inline TP_HOST_DEVICE BFloat16 operator-(const BFloat16& a, int64_t b) { return a - BFloat16(b); }
inline TP_HOST_DEVICE BFloat16 operator-(int64_t a, const BFloat16& b) { return BFloat16(a) - b; }
inline TP_HOST_DEVICE BFloat16 operator*(const BFloat16& a, int64_t b) { return a * BFloat16(b); }
inline TP_HOST_DEVICE BFloat16 operator*(int64_t a, const BFloat16& b) { return BFloat16(a) * b; }
inline TP_HOST_DEVICE BFloat16 operator/(const BFloat16& a, int64_t b) { return a / BFloat16(b); }
inline TP_HOST_DEVICE BFloat16 operator/(int64_t a, const BFloat16& b) { return BFloat16(a) / b; }
inline TP_HOST_DEVICE bool operator>(const BFloat16& a, int64_t b) { return static_cast<float>(a) > static_cast<float>(b); }
inline TP_HOST_DEVICE bool operator<(const BFloat16& a, int64_t b) { return static_cast<float>(a) < static_cast<float>(b); }

inline std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  out << static_cast<float>(value);
  return out;
}

} // namespace tensorplay

#undef TP_HOST_DEVICE
