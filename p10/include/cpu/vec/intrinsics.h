#pragma once

//
// CPU_CAPABILITY_* macros are normally supplied by the build system when a
// kernel translation unit is compiled for a specific instruction set
// (e.g. -DCPU_CAPABILITY_AVX2 -DCPU_CAPABILITY=AVX2 -mavx2 -mfma).  When those
// flags are absent, the macros are derived from the compiler's own ISA
// macros so that e.g. -mavx2 builds automatically get the AVX2 vec layer.

#if !defined(CPU_CAPABILITY_DEFAULT) && \
    !defined(CPU_CAPABILITY_AVX512) && !defined(CPU_CAPABILITY_AVX2) && \
    !defined(CPU_CAPABILITY_VSX) && !defined(CPU_CAPABILITY_ZVECTOR) && \
    !defined(CPU_CAPABILITY_SVE256) && !defined(CPU_CAPABILITY_SVE128) && \
    !defined(CPU_CAPABILITY_RVVM1) && !defined(CPU_CAPABILITY_RVVM2)
#if defined(__AVX2__)
#define CPU_CAPABILITY_AVX2
#elif defined(__VSX__)
#define CPU_CAPABILITY_VSX
#elif defined(__s390x__) && defined(__VX__)
#define CPU_CAPABILITY_ZVECTOR
#elif defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && \
    defined(__ARM_FEATURE_SVE_BITS) && \
    (__ARM_FEATURE_SVE_BITS == 256 || __ARM_FEATURE_SVE_BITS == 128)
#if __ARM_FEATURE_SVE_BITS == 256
#define CPU_CAPABILITY_SVE256
#else
#define CPU_CAPABILITY_SVE128
#endif
#elif defined(__riscv) && defined(__riscv_v_intrinsic)
// RISC-V vector tier picked by the build system's VLEN probe; the intrinsic
// suffix version macro guards against pre-1.0 vector toolchains.
#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000 && \
    defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen >= 256
#define CPU_CAPABILITY_RVVM2
#elif defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000 && \
    defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen >= 128
#define CPU_CAPABILITY_RVVM1
#else
#define CPU_CAPABILITY_DEFAULT
#endif
#else
#define CPU_CAPABILITY_DEFAULT
#endif
#endif

// included so that vector type declarations (__m256, __vector float,
// svfloat32_t, ...) are visible in every TU. Intrinsic *functions* are only
// usable where the ISA is enabled; uninstantiated class members in the
// vec256 headers therefore compile fine under DEFAULT.
//
// Arch-dispatched intrinsic headers, one include for every platform the CPU
// vector layer compiles on. This is the only such header: the tree also
// once carried an "Intrinsics.h" twin, which broke on case-insensitive
// filesystems (macOS/Windows) because #pragma once collapsed the two into
// one and the NEON includes below never happened.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC or clang-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__clang__) && (defined(__ARM_NEON__) || defined(__aarch64__))
/* Clang-compatible compiler, targeting arm neon */
#include <arm_neon.h>
#if defined(__ARM_FEATURE_SVE)
/* CLANG-compatible compiler, targeting ARM with SVE */
#include <arm_sve.h>
#endif
#elif defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#if _MSC_VER <= 1900
#define _mm256_extract_epi64(X, Y) \
  (_mm_extract_epi64(_mm256_extractf128_si256(X, Y >> 1), Y % 2))
#define _mm256_extract_epi32(X, Y) \
  (_mm_extract_epi32(_mm256_extractf128_si256(X, Y >> 2), Y % 4))
#define _mm256_extract_epi16(X, Y) \
  (_mm_extract_epi16(_mm256_extractf128_si256(X, Y >> 3), Y % 8))
#define _mm256_extract_epi8(X, Y) \
  (_mm_extract_epi8(_mm256_extractf128_si256(X, Y >> 4), Y % 16))
#endif
#elif defined(__GNUC__) && (defined(__ARM_NEON__) || defined(__aarch64__))
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#if defined(__ARM_FEATURE_SVE)
/* GCC-compatible compiler, targeting ARM with SVE */
#include <arm_sve.h>
#endif
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif defined(__s390x__)
// targets Z/architecture
#if defined(__VX__)
#include <vecintrin.h>
#endif
#elif (defined(__GNUC__) || defined(__xlC__)) && \
    (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
/* We need to undef those tokens defined by <altivec.h> to avoid conflicts
   with the C++ types. => Can still use __bool/__vector */
#undef bool
#undef vector
#undef pixel
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting ARM with SPE */
#include <spe.h>
#endif

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#define _M_X86_ 1
#endif

// Runtime capability detection: used by the dispatch layer to select the
// kernel compiled for the CPU's instruction set. Uses compiler builtins to
// avoid a dependency on cpuinfo. The cpuid builtin exists on x86 targets
// only; everywhere else (and on MSVC, which has no such builtin) the
// dispatcher's non-AVX paths are the answer.
inline bool tp_cpu_supports_avx2() {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

inline bool tp_cpu_supports_avx512() {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  return __builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512dq");
#else
  return false;
#endif
}

// ZVECTOR tier: the s390x vector extension operates on 128-bit vectors, and
// the fp32 SIMD instructions are only usable when the kernel implements the
// VXE facility; probe the hardware capability word provided by the kernel.
#if defined(__s390x__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef TP_HWCAP_S390_VXE
#define TP_HWCAP_S390_VXE 8192
#endif
inline bool tp_cpu_supports_zvector() {
  return (getauxval(AT_HWCAP) & TP_HWCAP_S390_VXE) != 0;
}
#else
inline bool tp_cpu_supports_zvector() { return false; }
#endif

// SVE tiers: an SVE-capable aarch64 kernel reports the feature bit through
// AT_HWCAP and the per-thread vector length through prctl(PR_SVE_GET_VL),
// whose low 16 bits are the vector length in bytes. The bf16 feature
// (AT_HWCAP2 bits 14 and 13 cover the extensions in two kernel eras) gates
// the tier because the SVE kernels rely on bf16 arithmetic.
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <sys/prctl.h>
#ifndef TP_HWCAP_SVE
#define TP_HWCAP_SVE (1 << 22)
#endif
#ifndef TP_HWCAP2_BF16
#define TP_HWCAP2_BF16 (1 << 14)
#endif
#ifndef TP_HWCAP2_SVEBF16
#define TP_HWCAP2_SVEBF16 (1 << 13)
#endif
#ifndef TP_PR_SVE_GET_VL
#define TP_PR_SVE_GET_VL 51
#endif
#ifndef TP_PR_SVE_VL_LEN_MASK
#define TP_PR_SVE_VL_LEN_MASK 0xffff
#endif
inline bool tp_cpu_has_arm_sve_bf16() {
  return (getauxval(AT_HWCAP) & TP_HWCAP_SVE) != 0 &&
      (getauxval(AT_HWCAP2) & (TP_HWCAP2_BF16 | TP_HWCAP2_SVEBF16)) != 0;
}
// Returns the max SVE vector length in bits, or 0 when SVE is unavailable.
inline int tp_cpu_sve_vector_length_bits() {
  if (!tp_cpu_has_arm_sve_bf16()) {
    return 0;
  }
  long vl = prctl(TP_PR_SVE_GET_VL);
  if (vl < 0) {
    return 0;
  }
  return static_cast<int>(vl & TP_PR_SVE_VL_LEN_MASK) * 8;
}
#else
inline bool tp_cpu_has_arm_sve_bf16() { return false; }
inline int tp_cpu_sve_vector_length_bits() { return 0; }
#endif

// RISC-V vector tiers: the "V" extension is reported through AT_HWCAP
// (bit 'V' = 18 on riscv64 kernels); the vector register length in bits
// comes from the CSR vlenb (bytes per vector) read via inline asm,
// matching the RVVM1 (VLEN=128) / RVVM2 (VLEN=256) tier split.
#if defined(__riscv) && __riscv_xlen == 64 && defined(__linux__)
#include <sys/auxv.h>
#ifndef TP_HWCAP_RISCV_VECTOR
#define TP_HWCAP_RISCV_VECTOR (1 << 18)
#endif
inline bool tp_cpu_has_rvv() {
  return (getauxval(AT_HWCAP) & TP_HWCAP_RISCV_VECTOR) != 0;
}
inline int tp_cpu_rvv_vector_length_bits() {
  if (!tp_cpu_has_rvv()) {
    return 0;
  }
  uintptr_t vlenb = 0;
  __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
  return static_cast<int>(vlenb * 8);
}
#else
inline bool tp_cpu_has_rvv() { return false; }
inline int tp_cpu_rvv_vector_length_bits() { return 0; }
#endif

// float division by zero); kept for interface compatibility.
#define TP_UBSAN_IGNORE_FLOAT_DIV
