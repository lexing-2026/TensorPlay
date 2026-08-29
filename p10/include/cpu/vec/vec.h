#pragma once

// Dispatch header: selects the vec layer for the CPU capability this TU is

#if defined(CPU_CAPABILITY_AVX512)
// Full-width 512-bit layer (float/double; int32/int64 pending).
#include "cpu/vec/vec512/vec512.h"
#else
#include "cpu/vec/vec128/vec128.h"
#include "cpu/vec/vec256/vec256.h"
#endif

#include <cstring>

namespace tensorplay::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

inline Vectorized<bool> convert_to_bool(Vectorized<int8_t> x) {
  __at_align__ bool buffer[x.size()];
  x.ne(Vectorized<int8_t>(0)).store(buffer);

  Vectorized<bool> ret;
  static_assert(x.size() == ret.size());
  std::memcpy(&ret, buffer, ret.size() * sizeof(bool));
  return ret;
}

template <typename T>
inline Vectorized<T> convert_to_int(Vectorized<T> x) {
  return x;
}

} // namespace tensorplay::vec::inline CPU_CAPABILITY
} // namespace tensorplay::vec
