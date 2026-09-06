#pragma once

// VSX helper layer for the 256-bit emulation over two 128-bit vector
// registers. <altivec.h> is pulled in by cpu/vec/intrinsics.h (via
// vec_base.h below) whenever the VSX ISA is enabled, and the bool/vector/
// pixel keyword macros it introduces are undefined immediately after that
// include; this header must stay below it in the include order. Everything
// here is a thin, always-inlined wrapper over the intrinsics plus the
// compile-time lane masks used by blend<mask>.

#include "cpu/vec/vec_base.h"

#include <cstdint>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

using vfloat32 = __vector float;
using vfloat64 = __vector double;
using vint32 = __vector int;
using vint64 = __vector long long;
using vuint32 = __vector unsigned int;
using vuint64 = __vector unsigned long long;
using vbool32 = __vector __bool int;
using vbool64 = __vector __bool long long;

// Reinterpret signed integer vectors as unsigned for the bit-shift
// intrinsics, which only exist over unsigned element types.
inline vuint32 make_vuint(vint32 v) {
  return reinterpret_cast<vuint32>(v);
}
inline vuint64 make_vuint(vint64 v) {
  return reinterpret_cast<vuint64>(v);
}

// Endian-neutral unaligned 128-bit load/store for VSX.
template <typename T>
inline vfloat32 vsx_ld(int offset, const T* ptr) {
  return vec_vsx_ld(offset, reinterpret_cast<const float*>(ptr));
}
template <typename T>
inline void vsx_st(vfloat32 v, int offset, T* ptr) {
  vec_vsx_st(v, offset, reinterpret_cast<float*>(ptr));
}
template <typename T>
inline vfloat64 vsx_ld_d(int offset, const T* ptr) {
  return vec_vsx_ld(offset, reinterpret_cast<const double*>(ptr));
}
template <typename T>
inline void vsx_st_d(vfloat64 v, int offset, T* ptr) {
  vec_vsx_st(v, offset, reinterpret_cast<double*>(ptr));
}

// Classifies which blend<mask> fast path applies for a 4-bit-per-128-bit-half
// lane mask. Returns a value 0..8; see float_vsx.h for the mapping.
constexpr int blend_choice(
    uint32_t mask,
    uint32_t half1 = 0xF,
    uint32_t half2 = 0xF0) {
  uint32_t none = 0;
  uint32_t both = half1 | half2;
  mask = mask & both;
  if (mask == none) {
    return 0;
  }
  if (mask == both) {
    return 1;
  }
  if (mask == half1) {
    return 2;
  }
  if (mask == half2) {
    return 3;
  }
  if (mask > 0 && mask < half1) {
    return 4;
  }
  if ((mask & half2) == half2) {
    return 5;
  }
  if ((mask & half1) == 0 && mask > half1) {
    return 6;
  }
  if ((mask & half1) == half1 && mask > half1) {
    return 7;
  }
  return 8;
}

constexpr int blend_choice_dbl(uint32_t mask) {
  return blend_choice(mask, 0x3, 0xC);
}

// Expands the low four bits of mask into a per-lane all-ones/all-zeros
// vector for the first 128-bit half.
constexpr vbool32 vsx_mask1(uint32_t mask) {
  uint32_t g0 = (mask & 1) * 0xffffffffu;
  uint32_t g1 = ((mask & 2) >> 1) * 0xffffffffu;
  uint32_t g2 = ((mask & 4) >> 2) * 0xffffffffu;
  uint32_t g3 = ((mask & 8) >> 3) * 0xffffffffu;
  vuint32 raw = {g0, g1, g2, g3};
  return (vbool32)raw;
}

// Same for the second 128-bit half (bits 4..7 of mask).
constexpr vbool32 vsx_mask2(uint32_t mask) {
  return vsx_mask1((mask & 0xFF) >> 4);
}

constexpr vbool64 vsx_dbl_mask1(uint32_t mask) {
  uint64_t g0 = (mask & 1) * 0xffffffffffffffffull;
  uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffffull;
  vuint64 raw = {g0, g1};
  return (vbool64)raw;
}

constexpr vbool64 vsx_dbl_mask2(uint32_t mask) {
  return vsx_dbl_mask1((mask & 0xF) >> 2);
}

// Member definitions shared by the float/double specializations. Each macro
// applies the intrinsic to both 128-bit halves.
#define TP_VSX_DEFINE_MEMBER_UNARY_OP(op, op_type, func)       \
  Vectorized<op_type> op() const {                             \
    return Vectorized<op_type>{func(_vec0), func(_vec1)};      \
  }

#define TP_VSX_DEFINE_MEMBER_OP(op, op_type, func)             \
  Vectorized<op_type> op(const Vectorized<op_type>& other)     \
      const {                                                  \
    return Vectorized<op_type>{                                \
        func(_vec0, other._vec0), func(_vec1, other._vec1)};   \
  }

#define TP_VSX_DEFINE_MEMBER_TERNARY_OP(op, op_type, func)      \
  Vectorized<op_type> op(                                       \
      const Vectorized<op_type>& b, const Vectorized<op_type>& c) \
      const {                                                   \
    return Vectorized<op_type>{                                 \
        func(_vec0, b._vec0, c._vec0),                          \
        func(_vec1, b._vec1, c._vec1)};                         \
  }

#define TP_VSX_DEFINE_MEMBER_OP_AND_ONE(op, op_type, func)                    \
  Vectorized<op_type> op(const Vectorized<op_type>& other) const {            \
    const auto v_one = vec_splats(static_cast<op_type>(1.0));                 \
    auto ret0 = (typename Vectorized<op_type>::vec_internal_type)func(        \
        _vec0, other._vec0);                                                  \
    auto ret1 = (typename Vectorized<op_type>::vec_internal_type)func(        \
        _vec1, other._vec1);                                                  \
    return Vectorized<op_type>{                                               \
        (typename Vectorized<op_type>::vec_internal_type)vec_and(ret0, v_one), \
        (typename Vectorized<op_type>::vec_internal_type)vec_and(ret1, v_one)}; \
  }

// Comparison producing an all-ones/all-zeros mask vector.
#define TP_VSX_DEFINE_MEMBER_CMP(op, op_type, func)            \
  Vectorized<op_type> op(const Vectorized<op_type>& other)     \
      const {                                                  \
    return Vectorized<op_type>{                                \
        (typename Vectorized<op_type>::vec_internal_type)      \
            func(_vec0, other._vec0),                          \
        (typename Vectorized<op_type>::vec_internal_type)      \
            func(_vec1, other._vec1)};                         \
  }

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
