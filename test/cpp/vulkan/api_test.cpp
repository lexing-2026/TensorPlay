#include "test_utils.h"

#include <gtest/gtest.h>

#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "VulkanRuntime.h"
#include "vulkan/Context.h"
#include "backend/vulkan/api/Context.h"
#include "backend/vulkan/api/QueryPool.h"

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
  EXPECT_FALSE(t.to(Device(DeviceType::CPU)).device().is_vulkan());
}

TEST_F(VulkanRuntimeTest, TimestampQueryRoundtrip) {
  if (!tensorplay::vulkan::api::context()->adapter_ptr()->has_timestamps()) {
    GTEST_SKIP() << "Device does not report timestamp support";
  }

  auto* context = tensorplay::vulkan::api::context();
  const auto slot = context->write_timestamp();
  EXPECT_NE(slot.first, VK_NULL_HANDLE);

  // Ensure the query is submitted and completed before reading back.
  context->flush();

  uint64_t ticks = 0;
  const VkResult result = vkGetQueryPoolResults(
      context->device(),
      slot.first,
      slot.second,
      1u,
      sizeof(ticks),
      &ticks,
      sizeof(ticks),
      VK_QUERY_RESULT_64_BIT);
  EXPECT_EQ(result, VK_SUCCESS);
  // A completed query yields any 64-bit value; zero only when the pool was
  // reset mid-flight, which the flush above prevents.
  (void)ticks;

  // Reset is legal once the device is idle; the next query reuses the slot.
  EXPECT_NO_THROW(context->querypool().reset());
}

} // namespace
