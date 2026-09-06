#pragma once

// ZVECTOR helper layer: two 128-bit vector registers emulate the 256-bit
// lane width. Intrinsics come from <vecintrin.h> (pulled in by
// cpu/vec/intrinsics.h when the vector facility is enabled); the include
// order requirement matches the VSX helper header.

#include "cpu/vec/vec_base.h"

#include <cstdint>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

using zf32 = __vector float;
using zf64 = __vector double;
using zi32 = __vector int;
using zi64 = __vector long long;
using zu32 = __vector unsigned int;
using zu64 = __vector unsigned long long;
using zb32 = __vector __bool int;
using zb64 = __vector __bool long long;

// Type-agnostic 128-bit save/restore used by the macro layer (avoids the
// strict element-type matching of the load/store intrinsics).
template <typename V, typename T>
inline void z_store_internal(const V& v, T* dst) {
  __builtin_memcpy(reinterpret_cast<void*>(dst), &v, sizeof(V));
}
template <typename V, typename T>
inline V z_load_internal(const T* src) {
  V v;
  __builtin_memcpy(&v, reinterpret_cast<const void*>(src), sizeof(V));
  return v;
}

// Unaligned 128-bit access. The extended-load intrinsics want the element
// pointer type exactly, so callers cast per element type; offset is in
// bytes from the base address.
template <typename T>
inline zf32 z_ld(int64_t offset, const T* ptr) {
  return vec_xl(offset, reinterpret_cast<float*>(const_cast<T*>(ptr)));
}
template <typename T>
inline void z_st(zf32 v, int64_t offset, T* ptr) {
  vec_xst(v, offset, reinterpret_cast<float*>(ptr));
}
template <typename T>
inline zf64 z_ld_d(int64_t offset, const T* ptr) {
  return vec_xl(offset, reinterpret_cast<double*>(const_cast<T*>(ptr)));
}
template <typename T>
inline void z_st_d(zf64 v, int64_t offset, T* ptr) {
  vec_xst(v, offset, reinterpret_cast<double*>(ptr));
}
// The integer overloads of the extended-load intrinsics are missing in
// some compilers, so the integer paths load through the float/double
// spellings and bitcast the register.
template <typename T>
inline zi32 z_ld_i(int64_t offset, const T* ptr) {
  return (zi32)z_ld(offset, ptr);
}
template <typename T>
inline void z_st_i(zi32 v, int64_t offset, T* ptr) {
  z_st((zf32)v, offset, ptr);
}
template <typename T>
inline zi64 z_ld_l(int64_t offset, const T* ptr) {
  return (zi64)z_ld_d(offset, ptr);
}
template <typename T>
inline void z_st_l(zi64 v, int64_t offset, T* ptr) {
  z_st_d((zf64)v, offset, ptr);
}

// Lane-mask fast-path classifier shared with the blend<mask> templates.
constexpr int z_blend_choice(
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

constexpr int z_blend_choice_dbl(uint32_t mask) {
  return z_blend_choice(mask, 0x3, 0xC);
}

// Per-lane all-ones/all-zeros masks from the low bits of mask.
constexpr zb32 z_mask1(uint32_t mask) {
  uint32_t g0 = (mask & 1) * 0xffffffffu;
  uint32_t g1 = ((mask & 2) >> 1) * 0xffffffffu;
  uint32_t g2 = ((mask & 4) >> 2) * 0xffffffffu;
  uint32_t g3 = ((mask & 8) >> 3) * 0xffffffffu;
  zu32 raw = {g0, g1, g2, g3};
  return (zb32)raw;
}

constexpr zb32 z_mask2(uint32_t mask) {
  return z_mask1((mask & 0xFF) >> 4);
}

constexpr zb64 z_dbl_mask1(uint32_t mask) {
  uint64_t g0 = (mask & 1) * 0xffffffffffffffffull;
  uint64_t g1 = ((mask & 2) >> 1) * 0xffffffffffffffffull;
  zu64 raw = {g0, g1};
  return (zb64)raw;
}

constexpr zb64 z_dbl_mask2(uint32_t mask) {
  return z_dbl_mask1((mask & 0xF) >> 2);
}

#define TP_ZV_DEFINE_MEMBER_UNARY_OP(op, op_type, func)        \
  Vectorized<op_type> op() const {                             \
    return Vectorized<op_type>{func(_vec0), func(_vec1)};      \
  }

#define TP_ZV_DEFINE_MEMBER_OP(op, op_type, func)              \
  Vectorized<op_type> op(const Vectorized<op_type>& other)     \
      const {                                                  \
    return Vectorized<op_type>{                                \
        func(_vec0, other._vec0), func(_vec1, other._vec1)};   \
  }

#define TP_ZV_DEFINE_MEMBER_OP_AND_ONE(op, op_type, func)                      \
  Vectorized<op_type> op(const Vectorized<op_type>& other) const {             \
    typename Vectorized<op_type>::vec_internal_type ret0 =                     \
        (typename Vectorized<op_type>::vec_internal_type)func(_vec0, other._vec0); \
    typename Vectorized<op_type>::vec_internal_type ret1 =                     \
        (typename Vectorized<op_type>::vec_internal_type)func(_vec1, other._vec1); \
    __at_align__ op_type tmp0[Vectorized<op_type>::size() / 2];                \
    __at_align__ op_type tmp1[Vectorized<op_type>::size() / 2];                \
    z_store_internal(ret0, tmp0);                                              \
    z_store_internal(ret1, tmp1);                                              \
    for (int i = 0; i < Vectorized<op_type>::size() / 2; ++i) {                \
      tmp0[i] = (tmp0[i] != static_cast<op_type>(0)) ? static_cast<op_type>(1) \
                                                     : static_cast<op_type>(0);\
      tmp1[i] = (tmp1[i] != static_cast<op_type>(0)) ? static_cast<op_type>(1) \
                                                     : static_cast<op_type>(0);\
    }                                                                          \
    return Vectorized<op_type>{                                                \
        z_load_internal<typename Vectorized<op_type>::vec_internal_type>(tmp0),\
        z_load_internal<typename Vectorized<op_type>::vec_internal_type>(tmp1)};\
  }

#define TP_ZV_DEFINE_MEMBER_CMP(op, op_type, func)              \
  Vectorized<op_type> op(const Vectorized<op_type>& other)      \
      const {                                                   \
    return Vectorized<op_type>{                                 \
        (typename Vectorized<op_type>::vec_internal_type)       \
            func(_vec0, other._vec0),                           \
        (typename Vectorized<op_type>::vec_internal_type)       \
            func(_vec1, other._vec1)};                          \
  }

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
