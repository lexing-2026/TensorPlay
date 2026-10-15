#pragma once

// Cross-dtype lane conversion between `VectorizedN` views: `convert`
// reinterprets values element-wise, `round_convert` rounds float sources
// and clamps to the destination integer range so out-of-range lanes stay
// defined.  Both dispatch through the `VecConvert` / `VecRoundConvert`
// customization points, which the per-ISA layers specialize with
// intrinsics where direct lane-shuffling conversions exist.

#include "cpu/vec/vec_base.h"
#include "cpu/vec/vec_n.h"

#include <cmath>
#include <limits>

namespace tensorplay::vec {
inline namespace CPU_CAPABILITY {

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    typename Enabled = void>
struct VecConvert {
  static inline VectorizedN<dst_t, dst_n> apply(
      const VectorizedN<src_t, src_n>& src) {
    constexpr int count = std::min(
        VectorizedN<src_t, src_n>::size(), VectorizedN<dst_t, dst_n>::size());
    __at_align__ src_t src_buf[VectorizedN<src_t, src_n>::size()];
    src.store(src_buf);
    __at_align__ dst_t dst_buf[VectorizedN<dst_t, dst_n>::size()];
    for (int i = 0; i < count; i++) {
      dst_buf[i] = static_cast<dst_t>(src_buf[i]);
    }
    return VectorizedN<dst_t, dst_n>::loadu(dst_buf, count);
  }
};

// Specialized for float-to-int vector conversion.  Keeps the result inside
// the destination type's limits; NaN and out-of-range lanes become defined
// values instead of wrapping.
template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    typename =
        std::enable_if_t<is_integer_v<dst_t> && is_floating_point_v<src_t>>>
struct VecRoundConvert {
  static inline VectorizedN<dst_t, dst_n> apply(
      const VectorizedN<src_t, src_n>& src) {
    constexpr int count = std::min(
        VectorizedN<src_t, src_n>::size(), VectorizedN<dst_t, dst_n>::size());
    __at_align__ src_t src_buf[VectorizedN<src_t, src_n>::size()];
    src.store(src_buf);
    __at_align__ dst_t dst_buf[VectorizedN<dst_t, dst_n>::size()];
    constexpr auto min_val =
        static_cast<src_t>(std::numeric_limits<dst_t>::min());
    constexpr auto max_val =
        static_cast<src_t>(std::numeric_limits<dst_t>::max());
    for (int i = 0; i < count; i++) {
      dst_buf[i] = static_cast<dst_t>(
          std::clamp(std::round(src_buf[i]), min_val, max_val));
    }
    return VectorizedN<dst_t, dst_n>::loadu(dst_buf, count);
  }
};

template <typename dst_t, int dst_n, typename src_t, int src_n>
struct VecConvert<
    dst_t,
    dst_n,
    src_t,
    src_n,
    std::enable_if_t<std::is_same_v<dst_t, src_t> && dst_n == src_n>> {
  static inline VectorizedN<dst_t, dst_n> apply(
      const VectorizedN<src_t, src_n>& src) {
    return src;
  }
};

template <typename dst_t, typename src_t>
inline std::enable_if_t<std::is_same_v<dst_t, src_t>, Vectorized<src_t>>
convert(const Vectorized<src_t>& src) {
  return src;
}

template <typename dst_t, typename src_t>
inline std::enable_if_t<std::is_same_v<dst_t, src_t>, Vectorized<src_t>>
round_convert(const Vectorized<src_t>& src) {
  return src;
}

template <typename dst_t, typename src_t>
inline std::enable_if_t<!std::is_same_v<dst_t, src_t>, Vectorized<dst_t>>
convert(const Vectorized<src_t>& src) {
  return VecConvert<dst_t, 1, src_t, 1>::apply(src);
}

template <typename dst_t, typename src_t>
inline std::enable_if_t<!std::is_same_v<dst_t, src_t>, Vectorized<dst_t>>
round_convert(const Vectorized<src_t>& src) {
  return VecRoundConvert<dst_t, 1, src_t, 1>::apply(src);
}

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    std::enable_if_t<dst_n != 1, int> = 0>
inline VectorizedN<dst_t, dst_n> convert(const VectorizedN<src_t, src_n>& src) {
  return VecConvert<dst_t, dst_n, src_t, src_n>::apply(src);
}

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    std::enable_if_t<dst_n != 1, int> = 0>
inline VectorizedN<dst_t, dst_n> round_convert(
    const VectorizedN<src_t, src_n>& src) {
  return VecRoundConvert<dst_t, dst_n, src_t, src_n>::apply(src);
}

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    bool keep = false,
    std::enable_if_t<dst_n == 1, int> = 0>
inline std::conditional_t<keep, VectorizedN<dst_t, 1>, Vectorized<dst_t>>
convert(const VectorizedN<src_t, src_n>& src) {
  return VecConvert<dst_t, dst_n, src_t, src_n>::apply(src);
}

template <
    typename dst_t,
    int dst_n,
    typename src_t,
    int src_n,
    bool keep = false,
    std::enable_if_t<dst_n == 1, int> = 0>
inline std::conditional_t<keep, VectorizedN<dst_t, 1>, Vectorized<dst_t>>
round_convert(const VectorizedN<src_t, src_n>& src) {
  return VecRoundConvert<dst_t, dst_n, src_t, src_n>::apply(src);
}

} // namespace CPU_CAPABILITY

// Generic widening/narrowing conversions for the reduced-precision floating
// point types.  The x86 vector layers override these with intrinsic
// register conversions (see functional.h); every other backend falls back to
// this buffer path, which is portable across all Vectorized specializations.
template <
    typename scalar_t,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_to_float(
    const Vectorized<scalar_t>& a) {
  constexpr int narrow_size = Vectorized<scalar_t>::size();
  constexpr int float_size = Vectorized<float>::size();
  static_assert(narrow_size == 2 * float_size);
  __at_align__ scalar_t narrow[narrow_size];
  a.store(narrow);
  __at_align__ float wide[narrow_size];
  for (const auto i : tensorplay::irange(narrow_size)) {
    wide[i] = static_cast<float>(narrow[i]);
  }
  return std::make_tuple(
      Vectorized<float>::loadu(wide),
      Vectorized<float>::loadu(wide + float_size));
}

template <
    typename scalar_t,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline Vectorized<scalar_t> convert_from_float(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  constexpr int narrow_size = Vectorized<scalar_t>::size();
  constexpr int float_size = Vectorized<float>::size();
  static_assert(narrow_size == 2 * float_size);
  __at_align__ float wide[narrow_size];
  a.store(wide);
  b.store(wide + float_size);
  __at_align__ scalar_t narrow[narrow_size];
  for (const auto i : tensorplay::irange(narrow_size)) {
    narrow[i] = static_cast<scalar_t>(wide[i]);
  }
  return Vectorized<scalar_t>::loadu(narrow);
}

} // namespace tensorplay::vec
