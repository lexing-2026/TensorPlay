#pragma once

// Dispatch header: selects the vec layer for the CPU capability this TU is
// compiled for. Mirrors ATen/cpu/vec/vec.h.

#if defined(CPU_CAPABILITY_AVX512)
// AVX512 layer not ported yet; kernels compiled with CPU_CAPABILITY_AVX512
// fall back to the generic (vec_base) layer.
#include "cpu/vec/vec_base.h"
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
