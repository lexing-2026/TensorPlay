#pragma once

// Quantized scalar types.  Each carries the raw storage value plus the
// affine quantize / dequantize semantics: q = clamp(zp + nearbyint(x /
// scale)) with round-half-even from nearbyint, and x = (q - zp) * scale.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tensorplay {

struct alignas(1) qint8 {
  using underlying = int8_t;
  int8_t val_;
  qint8() = default;
  explicit qint8(int8_t val) : val_(val) {}
};

struct alignas(1) quint8 {
  using underlying = uint8_t;
  uint8_t val_;
  quint8() = default;
  explicit quint8(uint8_t val) : val_(val) {}
};

struct alignas(4) qint32 {
  using underlying = int32_t;
  int32_t val_;
  qint32() = default;
  explicit qint32(int32_t val) : val_(val) {}
};

// Quantize one float value for an affine grid of the underlying type's
// full range; the caller passes the already-reciprocal multiplier.
template <typename T>
inline T quantize_val(double scale, int64_t zero_point, float value) {
  static_assert(
      std::is_same_v<T, qint8> || std::is_same_v<T, quint8> ||
          std::is_same_v<T, qint32>,
      "quantize_val expects a quantized tensor type");
  constexpr int32_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int32_t qmax = std::numeric_limits<typename T::underlying>::max();
  float inv_scale = static_cast<float>(1.0 / scale);
  int32_t r = static_cast<int32_t>(zero_point) +
      static_cast<int32_t>(std::nearbyint(value * inv_scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return T(static_cast<typename T::underlying>(r));
}

template <typename T>
inline float dequantize_val(double scale, int64_t zero_point, T value) {
  return static_cast<float>(
      (static_cast<double>(value.val_) - static_cast<double>(zero_point)) *
      scale);
}

// Requantize int32 accumulator values back to the storage type with the
// multiplier/zp of the destination grid.
template <typename T>
inline T requantize_from_int(double multiplier, int64_t zero_point, int64_t src) {
  constexpr int32_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int32_t qmax = std::numeric_limits<typename T::underlying>::max();
  int32_t r = static_cast<int32_t>(zero_point) +
      static_cast<int32_t>(std::nearbyint(
          static_cast<double>(src) * multiplier));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return T(static_cast<typename T::underlying>(r));
}

// Lane-wise quantize of a float buffer into the storage type; used by the
// vector quantized layers' quantize() entry points.
template <typename T, int precision = 8>
inline void quantize_vec(
    double scale,
    int64_t zero_point,
    const float* src,
    T* dst,
    size_t count) {
  (void)precision;
  for (size_t i = 0; i < count; i++) {
    dst[i] = quantize_val<T>(scale, zero_point, src[i]);
  }
}

} // namespace tensorplay
