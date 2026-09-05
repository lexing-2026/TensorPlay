#pragma once

//
// CPU_CAPABILITY_* macros are normally supplied by the build system when a
// kernel translation unit is compiled for a specific instruction set
// (e.g. -DCPU_CAPABILITY_AVX2 -DCPU_CAPABILITY=AVX2 -mavx2 -mfma).  When those
// flags are absent, the macros are derived from the compiler's own ISA
// macros so that e.g. -mavx2 builds automatically get the AVX2 vec layer.

#if !defined(CPU_CAPABILITY_AVX512) && !defined(CPU_CAPABILITY_AVX2) && \
    !defined(CPU_CAPABILITY_VSX) && !defined(CPU_CAPABILITY_ZVECTOR) && \
    !defined(CPU_CAPABILITY_SVE256) && !defined(CPU_CAPABILITY_SVE128)
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
#else
#define CPU_CAPABILITY_DEFAULT
#endif
#endif

// included so that vector type declarations (__m256 etc.) are visible in
// every TU. Intrinsic *functions* are only usable where the ISA is enabled
// uninstantiated class members in the vec256 headers therefore compile fine
// under DEFAULT.
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
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

// float division by zero); kept for interface compatibility.
#define TP_UBSAN_IGNORE_FLOAT_DIV
