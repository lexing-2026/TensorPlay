#pragma once

// NEON helper layer for the aarch64 128-bit vec tier. Intrinsics come from
// <arm_neon.h> (pulled in by cpu/vec/intrinsics.h on __aarch64__). The
// transcendental policy mirrors the desktop defaults: SLEEF's ADVSIMD
// runtime dispatchers on Linux and macOS, scalar <cmath> on Android.

#include "cpu/vec/vec_base.h"

#include <cstdint>

#if defined(__aarch64__) && !defined(__ANDROID__)
#define TP_NEON_SLEEF 1
#endif

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

// Bitwise NOT of a 128-bit unsigned vector (vmvnq only exists for 32-bit
// element spellings on some compilers).
inline uint32x4_t _tp_all_ones_u32() {
  return vdupq_n_u32(0xffffffffu);
}
inline uint64x2_t _tp_all_ones_u64() {
  return vdupq_n_u64(0xffffffffffffffffull);
}

// Member-op macros shared by the float/double NEON specializations.
#define TP_NEON_DEFINE_MEMBER_OP(op, op_type, func)            \
  Vectorized<op_type> op(const Vectorized<op_type>& other)     \
      const {                                                  \
    return Vectorized<op_type>(func(values, other.values));    \
  }

// Comparison producing an all-ones/all-zeros mask reinterpreted as the
// value type; utype is the matching unsigned vector base (32/64).
#define TP_NEON_DEFINE_MEMBER_CMP(op, op_type, ftype, vop, utype) \
  Vectorized<op_type> op(const Vectorized<op_type>& other)         \
      const {                                                      \
    return Vectorized<op_type>(                                    \
        vreinterpretq_##ftype##_##utype(vop(values, other.values))); \
  }

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
