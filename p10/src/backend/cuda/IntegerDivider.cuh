#pragma once

// Division by an invariant divisor via multiply/shift.  For any N-bit
// unsigned divisor d > 0 there exist a magic multiplier m (2^N <= m < 2^(N+1))
// and a shift s with floor(n / d) == floor((m * n) / 2^(N + s)); with
// m' = m - 2^N the quotient is ((m' * n) >> N + n) >> s.  The reduced range
// (uint32_t divisor, dividend <= INT32_MAX) keeps the arithmetic in 32-bit
// registers on the device; div and mod come out of one multiplication.
//
// Only uint32_t implements the fast path; every other width falls back to
// plain division through the same interface.

#include <cstdint>

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#include <cuda_runtime.h>
#endif

namespace tensorplay {
namespace cuda {
namespace detail {

struct DivModU32 {
    uint32_t div;
    uint32_t mod;
};

struct IntDividerU32 {
    IntDividerU32() = default;

    explicit IntDividerU32(uint32_t d) : divisor(d) {
        // d == 0 never occurs: sizes are positive by construction.
        shift = 0;
        while ((1ULL << shift) < d) ++shift;
        const uint64_t one = 1;
        const uint64_t magic =
            ((one << 32) * ((one << shift) - d)) / d + 1;
        m1 = static_cast<uint32_t>(magic);
    }

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __device__ inline uint32_t div(uint32_t n) const {
        // High 32 bits of the n * m1 product, then the correction shift.
        const uint32_t t = __umulhi(n, m1);
        return (t + n) >> shift;
    }
#else
    inline uint32_t div(uint32_t n) const {
        const uint64_t t = (static_cast<uint64_t>(n) * m1) >> 32;
        return static_cast<uint32_t>((t + n) >> shift);
    }
#endif

    __host__ __device__ inline uint32_t mod(uint32_t n) const {
        return n - div(n) * divisor;
    }

    __host__ __device__ inline DivModU32 divmod(uint32_t n) const {
        const uint32_t q = div(n);
        return DivModU32{q, n - q * divisor};
    }

    uint32_t divisor = 1;
    uint32_t m1 = 0;
    uint32_t shift = 0;
};

}  // namespace detail
}  // namespace cuda
}  // namespace tensorplay
