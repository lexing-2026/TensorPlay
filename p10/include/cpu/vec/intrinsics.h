#pragma once

//
// CPU_CAPABILITY_* macros are normally supplied by the build system when a
// kernel translation unit is compiled for a specific instruction set
// (e.g. -DCPU_CAPABILITY_AVX2 -DCPU_CAPABILITY=AVX2 -mavx2 -mfma).  When those
// flags are absent, the macros are derived from the compiler's own ISA
// macros so that e.g. -mavx2 builds automatically get the AVX2 vec layer.

#if !defined(CPU_CAPABILITY_AVX512) && !defined(CPU_CAPABILITY_AVX2)
#if defined(__AVX2__)
#define CPU_CAPABILITY_AVX2
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

// float division by zero); kept for interface compatibility.
#define TP_UBSAN_IGNORE_FLOAT_DIV
