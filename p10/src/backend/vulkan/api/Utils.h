#pragma once

#ifdef USE_VULKAN

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "Exception.h"

namespace tensorplay {
namespace vulkan {
namespace api {

#define VK_UNUSED __attribute__((__unused__))

// Shared small utilities (fixed-size vectors, extent helpers, hash
// combining) live in a nested ``utils`` namespace; call sites across the
// API and impl layers spell them ``api::utils::``.
namespace utils {

//
// Small fixed-size integer vector types used across shader argument blocks
// and texture extents.
//
template <typename T, size_t N>
class vec final {
 public:
  vec() {
    data_.fill(T(0));
  }

  template <typename... Args>
  vec(const T s, const Args... args) : data_{s, static_cast<T>(args)...} {}

  explicit vec(const std::vector<int64_t>& sizes) {
    data_.fill(T(1));
    const size_t count = std::min(N, sizes.size());
    for (size_t i = 0; i < count; ++i) {
      data_[i] = static_cast<T>(sizes[i]);
    }
  }

  T& operator[](const size_t index) {
    return data_[index];
  }

  const T& operator[](const size_t index) const {
    return data_[index];
  }

  bool operator==(const vec& other) const {
    return data_ == other.data_;
  }

  bool operator!=(const vec& other) const {
    return data_ != other.data_;
  }

  T* data() {
    return data_.data();
  }

  const T* data() const {
    return data_.data();
  }

  static constexpr size_t size() {
    return N;
  }

 private:
  std::array<T, N> data_;
};

using uvec2 = vec<uint32_t, 2u>;
using uvec3 = vec<uint32_t, 3u>;
using uvec4 = vec<uint32_t, 4u>;
using ivec2 = vec<int32_t, 2u>;
using ivec3 = vec<int32_t, 3u>;
using ivec4 = vec<int32_t, 4u>;
using vec2 = vec<float, 2u>;
using vec3 = vec<float, 3u>;
using vec4 = vec<float, 4u>;

inline uint32_t div_up(const uint32_t numerator, const uint32_t denominator) {
  return (numerator + denominator - 1u) / denominator;
}

inline uint32_t align_up(const uint32_t value, const uint32_t bound) {
  return div_up(value, bound) * bound;
}

inline uint32_t safe_downcast_to_u32(const int64_t v) {
  return static_cast<uint32_t>(v);
}

// Element at the N-th innermost index of `sizes`; 1 when out of range.
inline int64_t val_at(const int index, const int64_t* const sizes, const size_t size) {
  const int64_t ndim = static_cast<int64_t>(size);
  const int64_t offset = ndim + index;
  return (index < 0 && offset >= 0 && offset < ndim) ? sizes[offset] : 1;
}

inline int64_t val_at(const int index, const std::vector<int64_t>& sizes) {
  return val_at(index, sizes.data(), sizes.size());
}

template <typename T, size_t N>
inline vec<T, N> make_vec_prepadded1(const std::vector<int64_t>& sizes) {
  vec<T, N> result;
  result[0] = T(1);
  for (size_t i = 0; i < std::min(N - 1, sizes.size()); ++i) {
    result[i + 1] = static_cast<T>(sizes[sizes.size() - 1 - i]);
  }
  return result;
}

// {N, C, H, W} with leading ones for missing dims: the {W, H, C, N} layout
// helpers above cover Whcn-style blocks, while front-padded Nchw blocks are
// what the relayout shaders index with.
inline ivec4 make_ivec4_prepadded1(const std::vector<int64_t>& sizes) {
  VK_CHECK_COND(sizes.size() <= 4u);

  ivec4 result{1, 1, 1, 1};
  const size_t base = 4u - sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    result[i + base] = static_cast<int32_t>(sizes[i]);
  }

  return result;
}

// {W, H, C, N} ordering helpers, matching the shader-side convention.
inline uvec4 make_whcn_uvec4(const std::vector<int64_t>& sizes) {
  return uvec4(
      static_cast<uint32_t>(val_at(-1, sizes)),
      static_cast<uint32_t>(val_at(-2, sizes)),
      static_cast<uint32_t>(val_at(-3, sizes)),
      static_cast<uint32_t>(val_at(-4, sizes)));
}

inline ivec4 make_whcn_ivec4(const std::vector<int64_t>& sizes) {
  return ivec4(
      static_cast<int32_t>(val_at(-1, sizes)),
      static_cast<int32_t>(val_at(-2, sizes)),
      static_cast<int32_t>(val_at(-3, sizes)),
      static_cast<int32_t>(val_at(-4, sizes)));
}

inline size_t multiply_integers(const std::vector<int64_t>& sizes) {
  size_t result = 1u;
  for (const int64_t s : sizes) {
    result *= static_cast<size_t>(s);
  }
  return result;
}

inline size_t hash_combine(const size_t seed, const size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

} // namespace utils
} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
