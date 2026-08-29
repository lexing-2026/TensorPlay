#pragma once

// binary_kernel_reduce. Scalar-only variants (vectorized ops live directly in
// the sum kernels via binary_kernel_reduce_vec lambdas).

#include <cstdint>

namespace tensorplay {
inline namespace CPU_CAPABILITY {

template <typename scalar_t, typename acc_t = scalar_t>
struct SumOps {
  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }
  inline acc_t reduce(acc_t a, scalar_t b, int64_t /*idx*/) const {
    return a + static_cast<acc_t>(b);
  }
  inline acc_t project(acc_t a) const {
    return a;
  }
  inline acc_t translate_idx(acc_t a, int64_t /*idx*/) const {
    return a;
  }
};

} // namespace tensorplay::inline CPU_CAPABILITY
} // namespace tensorplay