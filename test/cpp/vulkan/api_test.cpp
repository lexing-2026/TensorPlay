#include "test_utils.h"

#include <gtest/gtest.h>

#include "Tensor.h"
#include "VulkanRuntime.h"
#include "vulkan/Context.h"

namespace {

using namespace tensorplay;
namespace tpx_ops = tensorplay::tpx::ops;

Tensor tpx_ops_test_zeros() {
  return tpx_ops::zeros({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
}

class VulkanRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override { vulkan_test::skip_if_no_vulkan(); }
};

TEST_F(VulkanRuntimeTest, AvailabilityProbeMatchesRuntime) {
  EXPECT_TRUE(ops::is_vulkan_available());
  EXPECT_TRUE(vulkan::is_available());
}

TEST_F(VulkanRuntimeTest, DeviceCountIsPositive) {
  EXPECT_GE(vulkan::device_count(), 1);
}

TEST_F(VulkanRuntimeTest, DeviceNameIsNonEmpty) {
  EXPECT_FALSE(vulkan::device_name(0).empty());
}

TEST_F(VulkanRuntimeTest, ApiVersionIsReported) {
  // Vulkan API versions pack (major << 22) | (minor << 12) | patch.
  EXPECT_GE(vulkan::device_api_version(0), 1u << 22u);
}

TEST_F(VulkanRuntimeTest, TotalMemoryIsReported) {
  EXPECT_GT(vulkan::device_total_memory(0), 0u);
}

TEST_F(VulkanRuntimeTest, SynchronizeIsSafeWithoutWork) {
  EXPECT_NO_THROW(vulkan::synchronize(0));
}

TEST_F(VulkanRuntimeTest, TensorIsVulkanProperty) {
  Tensor t = tpx_ops_test_zeros();
  EXPECT_TRUE(t.device().is_vulkan());
  EXPECT_FALSE(t.cpu().device().is_vulkan());
}

} // namespace
