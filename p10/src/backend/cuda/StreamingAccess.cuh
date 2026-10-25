#pragma once

// Evict-first ("streaming") loads and stores.  CUDA spells the cache hint
// __ldcs / __stcs; the AMD toolchain has no overload of them for the vector
// words these kernels move, so the HIP build takes a plain access instead,
// which keeps every value and drops only the hint.

namespace tensorplay {
namespace cuda {

template <typename T>
__device__ __forceinline__ T tp_ldcs(const T* p) {
#if defined(USE_ROCM)
    return *p;
#else
    return __ldcs(p);
#endif
}

template <typename T, typename V>
__device__ __forceinline__ void tp_stcs(T* p, const V& value) {
#if defined(USE_ROCM)
    *p = value;
#else
    __stcs(p, value);
#endif
}

} // namespace cuda
} // namespace tensorplay
