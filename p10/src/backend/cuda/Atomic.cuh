// dropped (p10 uses std::complex and has no atomic complex users). The
// algorithm bodies are kept verbatim.
#pragma once

#include "BFloat16.h"
#include "Half.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace tensorplay {
namespace cuda {

template <typename T>
__host__ __device__ inline bool tp_at_isnan(T v) {
    return std::isnan(v);
}

template <typename T>
struct AtomicFPOp;

template <>
struct AtomicFPOp<tensorplay::Half> {
  template <typename func_t>
  inline __device__ tensorplay::Half operator()(
      tensorplay::Half* address,
      tensorplay::Half val,
      const func_t& func) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    tensorplay::Half hsum;
    do {
      assumed = old;
      hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      hsum = func(hsum, val);
      old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                                : (old & 0xffff0000) | hsum.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    return hsum;
  }
};

template <>
struct AtomicFPOp<tensorplay::BFloat16> {
  template <typename func_t>
  inline __device__ tensorplay::BFloat16 operator()(
      tensorplay::BFloat16* address,
      tensorplay::BFloat16 val,
      const func_t& func) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    tensorplay::BFloat16 bsum;
    do {
      assumed = old;
      bsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      bsum = func(bsum, val);
      old = (size_t)address & 2 ? (old & 0xffff) | (bsum.x << 16)
                                : (old & 0xffff0000) | bsum.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
    bsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    return bsum;
  }
};

template <>
struct AtomicFPOp<double> {
  template <typename func_t>
  inline __device__ double operator()(
      double* address,
      double val,
      const func_t& func) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, func(val, assumed));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN
      // != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
  }
};

#define ATOMIC_INTEGER_IMPL(NAME)                                              \
  template <typename T, size_t n>                                              \
  struct Atomic##NAME##IntegerImpl;                                            \
                                                                               \
  template <typename T>                                                        \
  struct Atomic##NAME##IntegerImpl<T, 1> {                                     \
    template <typename func_t>                                                 \
    inline __device__ void operator()(T* address, T val, const func_t& func) { \
      size_t offset = (size_t)address & 3;                                     \
      uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);          \
      uint32_t old = *address_as_ui;                                           \
      uint32_t shift = offset * 8;                                             \
      uint32_t old_byte;                                                       \
      uint32_t newval;                                                         \
      uint32_t assumed;                                                        \
                                                                               \
      do {                                                                     \
        assumed = old;                                                         \
        old_byte = (old >> shift) & 0xff;                                      \
        newval = static_cast<uint8_t>(func(val, static_cast<T>(old_byte)));    \
        newval = (old & ~(0x000000ff << shift)) | (newval << shift);           \
        old = atomicCAS(address_as_ui, assumed, newval);                       \
      } while (assumed != old);                                                \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename T>                                                        \
  struct Atomic##NAME##IntegerImpl<T, 2> {                                     \
    template <typename func_t>                                                 \
    inline __device__ void operator()(T* address, T val, const func_t& func) { \
      size_t offset = (size_t)address & 2;                                     \
      uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);          \
      bool is_32_align = offset;                                               \
      uint32_t old = *address_as_ui;                                           \
      uint32_t old_bytes;                                                      \
      uint32_t newval;                                                         \
      uint32_t assumed;                                                        \
                                                                               \
      do {                                                                     \
        assumed = old;                                                         \
        old_bytes = is_32_align ? old >> 16 : old & 0xffff;                    \
        newval = static_cast<uint16_t>(func(val, static_cast<T>(old_bytes)));  \
        newval = is_32_align ? (old & 0xffff) | (newval << 16)                 \
                             : (old & 0xffff0000) | newval;                    \
        old = atomicCAS(address_as_ui, assumed, newval);                       \
      } while (assumed != old);                                                \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename T>                                                        \
  struct Atomic##NAME##IntegerImpl<T, 4> {                                     \
    template <typename func_t>                                                 \
    inline __device__ void operator()(T* address, T val, const func_t& func) { \
      uint32_t* address_as_ui = (uint32_t*)(address);                          \
      uint32_t old = *address_as_ui;                                           \
      uint32_t newval;                                                         \
      uint32_t assumed;                                                        \
                                                                               \
      do {                                                                     \
        assumed = old;                                                         \
        newval = static_cast<uint32_t>(func(val, static_cast<T>(old)));        \
        old = atomicCAS(address_as_ui, assumed, newval);                       \
      } while (assumed != old);                                                \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename T>                                                        \
  struct Atomic##NAME##IntegerImpl<T, 8> {                                     \
    template <typename func_t>                                                 \
    inline __device__ void operator()(T* address, T val, const func_t& func) { \
      unsigned long long* address_as_ui = (unsigned long long*)(address);      \
      unsigned long long old = *address_as_ui;                                 \
      unsigned long long newval;                                               \
      unsigned long long assumed;                                              \
                                                                               \
      do {                                                                     \
        assumed = old;                                                         \
        newval = static_cast<uint64_t>(func(val, static_cast<T>(old)));        \
        old = atomicCAS(address_as_ui, assumed, newval);                       \
      } while (assumed != old);                                                \
    }                                                                          \
  };

#define GPU_ATOMIC_INTEGER(NAME, OP, DTYPE)                           \
  inline __device__ void gpuAtomic##NAME(DTYPE* address, DTYPE val) { \
    Atomic##NAME##IntegerImpl<DTYPE, sizeof(DTYPE)>()(                \
        address, val, [](DTYPE a, DTYPE b) { return OP; });           \
  }

ATOMIC_INTEGER_IMPL(Add)
GPU_ATOMIC_INTEGER(Add, a || b, bool)

// Don't instantiate gpuAtomicAdd with the macro as it seems non-standard (see
// int32, int64)
inline __device__ void gpuAtomicAdd(uint8_t* address, uint8_t val) {
  AtomicAddIntegerImpl<uint8_t, sizeof(uint8_t)>()(
      address, val, [](uint8_t a, uint8_t b) { return a + b; });
}

inline __device__ void gpuAtomicAdd(int8_t* address, int8_t val) {
  AtomicAddIntegerImpl<int8_t, sizeof(int8_t)>()(
      address, val, [](int8_t a, int8_t b) { return a + b; });
}

inline __device__ void gpuAtomicAdd(int16_t* address, int16_t val) {
  AtomicAddIntegerImpl<int16_t, sizeof(int16_t)>()(
      address, val, [](int16_t a, int16_t b) { return a + b; });
}

inline __device__ void gpuAtomicAdd(uint16_t* address, uint16_t val) {
  AtomicAddIntegerImpl<uint16_t, sizeof(uint16_t)>()(
      address, val, [](uint16_t a, uint16_t b) { return a + b; });
}

inline __device__ void gpuAtomicAdd(uint32_t* address, uint32_t val) {
  AtomicAddIntegerImpl<uint32_t, sizeof(uint32_t)>()(
      address, val, [](uint32_t a, uint32_t b) { return a + b; });
}

inline __device__ void gpuAtomicAdd(uint64_t* address, uint64_t val) {
  AtomicAddIntegerImpl<uint64_t, sizeof(uint64_t)>()(
      address, val, [](uint64_t a, uint64_t b) { return a + b; });
}

inline __device__ int32_t gpuAtomicAdd(int32_t* address, int32_t val) {
  return atomicAdd(address, val);
}

inline __device__ void gpuAtomicAdd(int64_t* address, int64_t val) {
  static_assert(
      sizeof(unsigned long long int) == sizeof(int64_t),
      "bitwidth change is not allowed");
  atomicAdd(
      reinterpret_cast<unsigned long long int*>(address),
      static_cast<unsigned long long int>(val));
}

inline __device__ tensorplay::Half gpuAtomicAdd(tensorplay::Half* address, tensorplay::Half val) {
  return AtomicFPOp<tensorplay::Half>()(
      address, val, [](tensorplay::Half hsum, tensorplay::Half val) { return hsum + val; });
}

inline __device__ tensorplay::BFloat16 gpuAtomicAdd(
    tensorplay::BFloat16* address,
    tensorplay::BFloat16 val) {
  return AtomicFPOp<tensorplay::BFloat16>()(
      address, val, [](tensorplay::BFloat16 bsum, tensorplay::BFloat16 val) {
        return bsum + val;
      });
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
// from CUDA C Programmic Guide
inline __device__ double atomicAdd(double* address, double val)
#if defined(__clang__) && defined(__CUDA__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"
    __attribute__((enable_if(true, "")))
#pragma GCC diagnostic pop
#endif
{

  return AtomicFPOp<double>()(
      address, val, [](double val, unsigned long long int assumed) {
        return __double_as_longlong(val + __longlong_as_double(assumed));
      });
}
#endif

inline __device__ double gpuAtomicAdd(double* address, double val) {
  return atomicAdd(address, val);
}

inline __device__ float gpuAtomicAdd(float* address, float val) {
  return atomicAdd(address, val);
}

/* Note [gpuAtomicAdd vs atomicAdd]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Some extensions call atomicAdd() directly and require non-library provided
 * data type support. Only for these, we continue to provide atomicAdd
 * overloads.
 */
inline __device__ tensorplay::Half atomicAdd(tensorplay::Half* address, tensorplay::Half val) {
  return gpuAtomicAdd(address, val);
}

inline __device__ tensorplay::BFloat16 atomicAdd(
    tensorplay::BFloat16* address,
    tensorplay::BFloat16 val) {
  return gpuAtomicAdd(address, val);
}

inline __device__ void atomicAdd(uint8_t* address, uint8_t val) {
  gpuAtomicAdd(address, val);
}

inline __device__ void atomicAdd(int8_t* address, int8_t val) {
  gpuAtomicAdd(address, val);
}

inline __device__ void atomicAdd(int16_t* address, int16_t val) {
  gpuAtomicAdd(address, val);
}

inline __device__ void atomicAdd(int64_t* address, int64_t val) {
  gpuAtomicAdd(address, val);
}

inline __device__ void atomicAdd(bool* address, bool val) {
  gpuAtomicAdd(address, val);
}

inline __device__ void gpuAtomicAddNoReturn(uint8_t* address, uint8_t val) {
  gpuAtomicAdd(address, val);
}
inline __device__ void gpuAtomicAddNoReturn(int8_t* address, int8_t val) {
  gpuAtomicAdd(address, val);
}
inline __device__ void gpuAtomicAddNoReturn(int16_t* address, int16_t val) {
  gpuAtomicAdd(address, val);
}
inline __device__ void gpuAtomicAddNoReturn(int32_t* address, int32_t val) {
  gpuAtomicAdd(address, val);
}
inline __device__ void gpuAtomicAddNoReturn(int64_t* address, int64_t val) {
  gpuAtomicAdd(address, val);
}
inline __device__ void gpuAtomicAddNoReturn(bool* address, bool val) {
  gpuAtomicAdd(address, val);
}
inline __device__ void gpuAtomicAddNoReturn(tensorplay::Half* address, tensorplay::Half val) {
  gpuAtomicAdd(address, val);
}
inline __device__ void gpuAtomicAddNoReturn(
    tensorplay::BFloat16* address,
    tensorplay::BFloat16 val) {
  gpuAtomicAdd(address, val);
}
inline __device__ void gpuAtomicAddNoReturn(float* address, float val) {
  gpuAtomicAdd(address, val);
}
inline __device__ void gpuAtomicAddNoReturn(double* address, double val) {
  gpuAtomicAdd(address, val);
}

// Atomic multiplication implementation.

ATOMIC_INTEGER_IMPL(Mul)
GPU_ATOMIC_INTEGER(Mul, a* b, uint8_t)
GPU_ATOMIC_INTEGER(Mul, a* b, int8_t)
GPU_ATOMIC_INTEGER(Mul, a* b, int16_t)
GPU_ATOMIC_INTEGER(Mul, a* b, int32_t)
GPU_ATOMIC_INTEGER(Mul, a* b, int64_t)
GPU_ATOMIC_INTEGER(Mul, a* b, uint16_t)
GPU_ATOMIC_INTEGER(Mul, a* b, uint32_t)
GPU_ATOMIC_INTEGER(Mul, a* b, uint64_t)
GPU_ATOMIC_INTEGER(Mul, a && b, bool)

inline __device__ tensorplay::Half gpuAtomicMul(tensorplay::Half* address, tensorplay::Half val) {
  return AtomicFPOp<tensorplay::Half>()(
      address, val, [](tensorplay::Half bsum, tensorplay::Half val) { return bsum * val; });
}

inline __device__ tensorplay::BFloat16 gpuAtomicMul(
    tensorplay::BFloat16* address,
    tensorplay::BFloat16 val) {
  return AtomicFPOp<tensorplay::BFloat16>()(
      address, val, [](tensorplay::BFloat16 bsum, tensorplay::BFloat16 val) {
        return bsum * val;
      });
}

inline __device__ double gpuAtomicMul(double* address, double val) {
  return AtomicFPOp<double>()(
      address, val, [](double val, unsigned long long int assumed) {
        return __double_as_longlong(val * __longlong_as_double(assumed));
      });
}

// Dont use a templated function for this since the addition function defaults
// to the CUDA built-in.
inline __device__ float gpuAtomicMul(float* address, float val) {
  unsigned int* address_as_ull = (unsigned int*)address;
  unsigned int old = *address_as_ull;
  unsigned int assumed;

  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed, __float_as_int(val * __int_as_float(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

// Atomic maximum implementation.

template <typename T>
__host__ __device__ T safe_max(T a, T b) {
  T max = tp_at_isnan(b) ? b : std::max<T>(a, b);
  return max;
}

ATOMIC_INTEGER_IMPL(Max)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), uint8_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), int8_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), int16_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), int32_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), int64_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), uint16_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), uint32_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), uint64_t)
GPU_ATOMIC_INTEGER(Max, safe_max(a, b), bool)

inline __device__ tensorplay::Half gpuAtomicMax(tensorplay::Half* address, tensorplay::Half val) {
  return AtomicFPOp<tensorplay::Half>()(address, val, [](tensorplay::Half bsum, tensorplay::Half val) {
    return safe_max(bsum, val);
  });
}

inline __device__ tensorplay::BFloat16 gpuAtomicMax(
    tensorplay::BFloat16* address,
    tensorplay::BFloat16 val) {
  return AtomicFPOp<tensorplay::BFloat16>()(
      address, val, [](tensorplay::BFloat16 bsum, tensorplay::BFloat16 val) {
        return safe_max(bsum, val);
      });
}

inline __device__ double gpuAtomicMax(double* address, double val) {
  return AtomicFPOp<double>()(
      address, val, [](double val, unsigned long long int assumed) {
        return __double_as_longlong(
            safe_max(val, __longlong_as_double(assumed)));
      });
}

// Dont use a templated function for this since the addition function defaults
// to the CUDA built-in.
inline __device__ float gpuAtomicMax(float* address, float val) {
  unsigned int* address_as_ull = (unsigned int*)address;
  unsigned int old = *address_as_ull;
  unsigned int assumed;

  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __float_as_int(safe_max(val, __int_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

// Atomic minimum implementation.

template <typename T>
__host__ __device__ T safe_min(T a, T b) {
  T min = tp_at_isnan(b) ? b : std::min<T>(a, b);
  return min;
}

ATOMIC_INTEGER_IMPL(Min)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), uint8_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), int8_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), int16_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), int32_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), int64_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), uint16_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), uint32_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), uint64_t)
GPU_ATOMIC_INTEGER(Min, safe_min(a, b), bool)

inline __device__ tensorplay::Half gpuAtomicMin(tensorplay::Half* address, tensorplay::Half val) {
  return AtomicFPOp<tensorplay::Half>()(address, val, [](tensorplay::Half bsum, tensorplay::Half val) {
    return safe_min(bsum, val);
  });
}

inline __device__ tensorplay::BFloat16 gpuAtomicMin(
    tensorplay::BFloat16* address,
    tensorplay::BFloat16 val) {
  return AtomicFPOp<tensorplay::BFloat16>()(
      address, val, [](tensorplay::BFloat16 bsum, tensorplay::BFloat16 val) {
        return safe_min(bsum, val);
      });
}

inline __device__ double gpuAtomicMin(double* address, double val) {
  return AtomicFPOp<double>()(
      address, val, [](double val, unsigned long long int assumed) {
        return __double_as_longlong(
            safe_min(val, __longlong_as_double(assumed)));
      });
}

// Dont use a templated function for this since the addition function defaults
// to the CUDA built-in.
inline __device__ float gpuAtomicMin(float* address, float val) {
  unsigned int* address_as_ull = (unsigned int*)address;
  unsigned int old = *address_as_ull;
  unsigned int assumed;

  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __float_as_int(safe_min(val, __int_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

} // namespace cuda
} // namespace tensorplay
