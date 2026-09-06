#pragma once

// SVE helper layer for fixed-length vector builds. The ACLE types
// (svfloat32_t, ...) are sized by -msve-vector-bits at compile time, so the
// same headers serve the SVE128 and SVE256 tiers; the lane count matches
// VECTOR_WIDTH from vec_base.h in both cases.

#include "cpu/vec/vec_base.h"

#include <cstdint>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

// Vector-length-specific spellings: plain sv types are sizeless and cannot
// be class members; the arm_sve_vector_bits attribute pins the size to
// VECTOR_WIDTH (set by the tier's -msve-vector-bits flag) while remaining
// ABI-compatible with the sizeless forms in intrinsic calls.
typedef svbool_t vls_bool_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint64_t vls_int64_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint32_t vls_int32_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint64_t vls_uint64_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint32_t vls_uint32_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat64_t vls_float64_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat32_t vls_float32_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));

// All-true predicate over the fixed vector length.
inline svbool_t sve_ptrue_f32() {
  return svptrue_b32();
}
inline svbool_t sve_ptrue_f64() {
  return svptrue_b64();
}
inline svbool_t sve_ptrue_s64() {
  return svptrue_b64();
}
inline svbool_t sve_ptrue_s32() {
  return svptrue_b32();
}

// whilelt predicate for the first `count` lanes.
inline svbool_t sve_first_f32(int64_t count) {
  return svwhilelt_b32_s32(0, static_cast<int32_t>(count));
}
inline svbool_t sve_first_f64(int64_t count) {
  return svwhilelt_b64_s64(0, count);
}

// Build an all-ones/all-zeros float lane pattern.
inline vls_float32_t sve_ones_f32() {
  return svreinterpret_f32_u32(svdup_n_u32(0xffffffffu));
}
inline vls_float32_t sve_zeros_f32() {
  return svdup_n_f32(0.0f);
}
inline vls_float64_t sve_ones_f64() {
  return svreinterpret_f64_u64(svdup_n_u64(0xffffffffffffffffull));
}
inline vls_float64_t sve_zeros_f64() {
  return svdup_n_f64(0.0);
}

// Convert a comparison-result float vector (all-ones/all-zeros lanes) into
// a lane predicate.
inline svbool_t sve_bits_to_pred_f32(vls_float32_t mask) {
  return svcmpne_n_u32(
      svptrue_b32(), svreinterpret_u32_f32(mask), 0u);
}
inline svbool_t sve_bits_to_pred_f64(vls_float64_t mask) {
  return svcmpne_n_u64(
      svptrue_b64(), svreinterpret_u64_f64(mask), 0ull);
}

// Compare-style intrinsics produce a float vector whose lanes are
// all-ones where the predicate holds.
inline svfloat32_t sve_cmp_f32(svbool_t m) {
  return svsel_f32(m, sve_ones_f32(), sve_zeros_f32());
}
inline svfloat64_t sve_cmp_f64(svbool_t m) {
  return svsel_f64(m, sve_ones_f64(), sve_zeros_f64());
}

// 0/1-valued comparison results.
inline svfloat32_t sve_cmp01_f32(svbool_t m) {
  return svsel_f32(m, svdup_n_f32(1.0f), svdup_n_f32(0.0f));
}
inline svfloat64_t sve_cmp01_f64(svbool_t m) {
  return svsel_f64(m, svdup_n_f64(1.0), svdup_n_f64(0.0));
}

// Mask-vector to predicate for blend<mask>: expand the compile-time bits
// into a per-lane 0/0xffffffff pattern and compare against zero.
template <int64_t mask, int N>
inline svbool_t sve_lane_pred_f32() {
  uint32_t bits[N];
  for (int i = 0; i < N; ++i) {
    bits[i] = ((mask >> i) & 1) ? 0xffffffffu : 0u;
  }
  svuint32_t mv = svld1_u32(svptrue_b32(), bits);
  return svcmpne_n_u32(svptrue_b32(), mv, 0u);
}
template <int64_t mask, int N>
inline svbool_t sve_lane_pred_f64() {
  uint64_t bits[N];
  for (int i = 0; i < N; ++i) {
    bits[i] = ((mask >> i) & 1) ? 0xffffffffffffffffull : 0ull;
  }
  svuint64_t mv = svld1_u64(svptrue_b64(), bits);
  return svcmpne_n_u64(svptrue_b64(), mv, 0ull);
}

// max/min with NaN propagation (a NaN on either side wins).
inline svfloat32_t sve_max_nan_f32(svfloat32_t a, svfloat32_t b) {
  svbool_t pg = svptrue_b32();
  svbool_t nan_a = svcmpne_f32(pg, a, a);
  svbool_t nan_b = svcmpne_f32(pg, b, b);
  svfloat32_t m = svmax_f32_x(pg, a, b);
  m = svsel_f32(nan_a, a, m);
  return svsel_f32(nan_b, b, m);
}
inline svfloat32_t sve_min_nan_f32(svfloat32_t a, svfloat32_t b) {
  svbool_t pg = svptrue_b32();
  svbool_t nan_a = svcmpne_f32(pg, a, a);
  svbool_t nan_b = svcmpne_f32(pg, b, b);
  svfloat32_t m = svmin_f32_x(pg, a, b);
  m = svsel_f32(nan_a, a, m);
  return svsel_f32(nan_b, b, m);
}
inline svfloat64_t sve_max_nan_f64(svfloat64_t a, svfloat64_t b) {
  svbool_t pg = svptrue_b64();
  svbool_t nan_a = svcmpne_f64(pg, a, a);
  svbool_t nan_b = svcmpne_f64(pg, b, b);
  svfloat64_t m = svmax_f64_x(pg, a, b);
  m = svsel_f64(nan_a, a, m);
  return svsel_f64(nan_b, b, m);
}
inline svfloat64_t sve_min_nan_f64(svfloat64_t a, svfloat64_t b) {
  svbool_t pg = svptrue_b64();
  svbool_t nan_a = svcmpne_f64(pg, a, a);
  svbool_t nan_b = svcmpne_f64(pg, b, b);
  svfloat64_t m = svmin_f64_x(pg, a, b);
  m = svsel_f64(nan_a, a, m);
  return svsel_f64(nan_b, b, m);
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
