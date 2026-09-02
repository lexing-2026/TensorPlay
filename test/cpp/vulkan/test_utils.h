//
// Shared helpers for the Vulkan C++ test suite.  Tests are skipped, not
// failed, when the build lacks the Vulkan backend or the runtime reports no
// usable adapter (e.g. CI containers without a loader or ICD).
//

#pragma once

#include <gtest/gtest.h>

#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "VulkanRuntime.h"

#include <cmath>

namespace vulkan_test {

// Convenience accessor: the C++ front end exposes device transfer through
// to(); there is no shorthand cpu() method on Tensor.
inline tensorplay::Tensor to_cpu(const tensorplay::Tensor& t) {
  return t.to(tensorplay::Device(tensorplay::DeviceType::CPU));
}

inline void skip_if_no_vulkan() {
  if (!tensorplay::vulkan::is_available()) {
    GTEST_SKIP() << "Vulkan runtime is not available";
  }
}

// Element-wise closeness check against a CPU reference computed by the
// caller.  llvmpipe and mobile GPUs both evaluate in FP32, so the default
// tolerance covers fused-multiply reassociation.  Both sides are
// materialized into contiguous host tensors first so non-row-major host
// layouts compare in logical order.
inline void expect_allclose(
    const tensorplay::Tensor& actual,
    const tensorplay::Tensor& expected,
    double rtol = 1e-5,
    double atol = 1e-6) {
  ASSERT_EQ(actual.numel(), expected.numel());
  const tensorplay::Tensor a_host = to_cpu(actual).contiguous();
  const tensorplay::Tensor e_host = to_cpu(expected).contiguous();
  ASSERT_EQ(a_host.numel(), e_host.numel());
  const float* a = a_host.data_ptr<float>();
  const float* e = e_host.data_ptr<float>();
  for (int64_t i = 0; i < e_host.numel(); ++i) {
    ASSERT_NEAR(a[i], e[i], atol + rtol * std::fabs(e[i]))
        << " mismatch at flat index " << i;
  }
}

} // namespace vulkan_test
