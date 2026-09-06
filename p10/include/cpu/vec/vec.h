#pragma once

// Dispatch header: selects the vec layer for the CPU capability this TU is
// compiled with. The x86 tiers use the AVX2/AVX-512 backends; the VSX,
// ZVECTOR and SVE tiers use their own intrinsic backends; every other
// configuration (x86 DEFAULT, plain aarch64, unknown hosts) falls back to
// the generic template in vec_base.h.

#if defined(CPU_CAPABILITY_AVX512)
// Full-width 512-bit layer (float/double; int32/int64 pending).
#include "cpu/vec/vec512/vec512.h"
#elif defined(CPU_CAPABILITY_VSX)
// PowerPC VSX tier: 256-bit emulation over two 128-bit registers.
#include "cpu/vec/vec256/vsx/vsx.h"
#elif defined(CPU_CAPABILITY_ZVECTOR)
// s390x vector-facility tier: 256-bit emulation over two 128-bit registers.
#include "cpu/vec/vec256/zarch/zarch.h"
#elif defined(CPU_CAPABILITY_SVE256) || defined(CPU_CAPABILITY_SVE128)
// aarch64 SVE tiers: single fixed-length vector per value.
#include "cpu/vec/sve/vec_sve.h"
#elif defined(__aarch64__) && !defined(__arm__)
// aarch64 desktop default (plain Linux distributions, macOS arm64): 128-bit
// NEON backends. aarch32 keeps the generic fallback.
#include "cpu/vec/vec128/vec128_neon.h"
#else
// x86 DEFAULT / bare -mavx2 builds and the generic fallback path.
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
