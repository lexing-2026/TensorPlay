#include "test_utils.h"

#include <gtest/gtest.h>

#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace {

using namespace tensorplay;
namespace tpx_ops = tensorplay::tpx::ops;

//
// Factory ops on the Vulkan device: payloads are created as GPU textures and
// never touch host memory unless a copy is requested.
//

class VulkanFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override { vulkan_test::skip_if_no_vulkan(); }
};

TEST_F(VulkanFactoryTest, ZerosHasRequestedShapeAndDevice) {
  Tensor t = tpx_ops::zeros({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  EXPECT_EQ(t.device().type(), DeviceType::Vulkan);
  EXPECT_EQ(t.dim(), 2);
  EXPECT_EQ(t.size(0), 2);
  EXPECT_EQ(t.size(1), 3);
  EXPECT_EQ(t.dtype(), DType::Float32);
}

TEST_F(VulkanFactoryTest, ZerosValuesAreZero) {
  Tensor t = tpx_ops::zeros({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  Tensor host = t.to(Device(DeviceType::CPU));
  const float* data = host.data_ptr<float>();
  for (int64_t i = 0; i < host.numel(); ++i) {
    EXPECT_EQ(data[i], 0.0f) << " index " << i;
  }
}

TEST_F(VulkanFactoryTest, OnesValuesAreOne) {
  Tensor t = tpx_ops::ones({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  Tensor host = t.to(Device(DeviceType::CPU));
  const float* data = host.data_ptr<float>();
  for (int64_t i = 0; i < host.numel(); ++i) {
    EXPECT_EQ(data[i], 1.0f) << " index " << i;
  }
}

TEST_F(VulkanFactoryTest, FullValuesMatchFillValue) {
  Tensor t = tpx_ops::full({2, 3}, 2.5, DType::Float32, Device(DeviceType::Vulkan));
  Tensor host = t.to(Device(DeviceType::CPU));
  const float* data = host.data_ptr<float>();
  for (int64_t i = 0; i < host.numel(); ++i) {
    EXPECT_EQ(data[i], 2.5f) << " index " << i;
  }
}

TEST_F(VulkanFactoryTest, EmptyHasRequestedShape) {
  Tensor t = tpx_ops::empty({4, 5}, DType::Float32, Device(DeviceType::Vulkan));
  EXPECT_EQ(t.device().type(), DeviceType::Vulkan);
  EXPECT_EQ(t.dim(), 2);
  EXPECT_EQ(t.size(0), 4);
  EXPECT_EQ(t.size(1), 5);
}

} // namespace
