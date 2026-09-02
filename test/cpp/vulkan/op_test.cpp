#include "test_utils.h"

#include <gtest/gtest.h>

#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>

namespace {

using namespace tensorplay;
namespace tpx_ops = tensorplay::tpx::ops;

//
// Element-wise compute on the Vulkan device.  Every test pushes a CPU
// reference through the same ops on the CPU and compares numerically, so a
// shader bug shows up as a value mismatch rather than a crash.
//

class VulkanOpTest : public ::testing::Test {
 protected:
  void SetUp() override { vulkan_test::skip_if_no_vulkan(); }
};

TEST_F(VulkanOpTest, CpuToVulkanToCpuRoundtrip) {
  Tensor x = tpx_ops::arange(6., DType::Float32).reshape({2, 3});
  Tensor back = x.to(Device(DeviceType::Vulkan)).to(Device(DeviceType::CPU));
  vulkan_test::expect_allclose(back, x);
}

TEST_F(VulkanOpTest, ZerosRoundtrip) {
  Tensor x = tpx_ops::zeros({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  vulkan_test::expect_allclose(x.to(Device(DeviceType::CPU)), tpx_ops::zeros({2, 3}));
}

TEST_F(VulkanOpTest, AddTensor) {
  Tensor a = tpx_ops::ones({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  Tensor b = tpx_ops::full({2, 3}, 2., DType::Float32, Device(DeviceType::Vulkan));
  Tensor back = tpx_ops::add(a, b).to(Device(DeviceType::CPU));
  Tensor expected = tpx_ops::full({2, 3}, 3.);
  vulkan_test::expect_allclose(back, expected);
}

TEST_F(VulkanOpTest, SubTensor) {
  Tensor a = tpx_ops::full({2, 3}, 5., DType::Float32, Device(DeviceType::Vulkan));
  Tensor b = tpx_ops::ones({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  Tensor back = tpx_ops::sub(a, b).to(Device(DeviceType::CPU));
  vulkan_test::expect_allclose(back, tpx_ops::full({2, 3}, 4.));
}

TEST_F(VulkanOpTest, MulTensor) {
  Tensor a = tpx_ops::full({2, 3}, 3., DType::Float32, Device(DeviceType::Vulkan));
  Tensor b = tpx_ops::full({2, 3}, 2., DType::Float32, Device(DeviceType::Vulkan));
  Tensor back = tpx_ops::mul(a, b).to(Device(DeviceType::CPU));
  vulkan_test::expect_allclose(back, tpx_ops::full({2, 3}, 6.));
}

TEST_F(VulkanOpTest, DivTensor) {
  Tensor a = tpx_ops::full({2, 3}, 6., DType::Float32, Device(DeviceType::Vulkan));
  Tensor b = tpx_ops::full({2, 3}, 2., DType::Float32, Device(DeviceType::Vulkan));
  Tensor back = tpx_ops::div(a, b).to(Device(DeviceType::CPU));
  vulkan_test::expect_allclose(back, tpx_ops::full({2, 3}, 3.));
}

TEST_F(VulkanOpTest, AddScalar) {
  Tensor a = tpx_ops::ones({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  Tensor back = a.add(2.).to(Device(DeviceType::CPU));
  vulkan_test::expect_allclose(back, tpx_ops::full({2, 3}, 3.));
}

TEST_F(VulkanOpTest, MulScalar) {
  Tensor a = tpx_ops::full({2, 3}, 3., DType::Float32, Device(DeviceType::Vulkan));
  Tensor back = a.mul(2.).to(Device(DeviceType::CPU));
  vulkan_test::expect_allclose(back, tpx_ops::full({2, 3}, 6.));
}

TEST_F(VulkanOpTest, Exp) {
  Tensor a = tpx_ops::ones({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  Tensor back = a.exp().to(Device(DeviceType::CPU));
  vulkan_test::expect_allclose(back, tpx_ops::full({2, 3}, std::exp(1.0f)));
}

TEST_F(VulkanOpTest, Sqrt) {
  Tensor a = tpx_ops::full({2, 3}, 4., DType::Float32, Device(DeviceType::Vulkan));
  Tensor back = a.sqrt().to(Device(DeviceType::CPU));
  vulkan_test::expect_allclose(back, tpx_ops::full({2, 3}, 2.));
}

TEST_F(VulkanOpTest, AbsAndNeg) {
  Tensor a = tpx_ops::full({2, 3}, -3., DType::Float32, Device(DeviceType::Vulkan));
  vulkan_test::expect_allclose(a.abs().to(Device(DeviceType::CPU)), tpx_ops::full({2, 3}, 3.));
  vulkan_test::expect_allclose(a.neg().to(Device(DeviceType::CPU)), tpx_ops::full({2, 3}, 3.));
}

TEST_F(VulkanOpTest, Fill) {
  Tensor a = tpx_ops::zeros({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  tpx_ops::fill_(a, 1.5);
  vulkan_test::expect_allclose(a.to(Device(DeviceType::CPU)), tpx_ops::full({2, 3}, 1.5));
}

TEST_F(VulkanOpTest, InplaceAdd) {
  Tensor a = tpx_ops::ones({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  Tensor b = tpx_ops::ones({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  tpx_ops::add_(a, b);
  vulkan_test::expect_allclose(a.to(Device(DeviceType::CPU)), tpx_ops::full({2, 3}, 2.));
}

TEST_F(VulkanOpTest, InplaceAddScalar) {
  Tensor a = tpx_ops::ones({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  tpx_ops::add_(a, Scalar(2.));
  vulkan_test::expect_allclose(a.to(Device(DeviceType::CPU)), tpx_ops::full({2, 3}, 3.));
}

TEST_F(VulkanOpTest, Clamp) {
  Tensor a = tpx_ops::arange(6., DType::Float32).to(Device(DeviceType::Vulkan));
  Tensor back = tpx_ops::clamp(a, Scalar(1.5), Scalar(4.)).to(Device(DeviceType::CPU));
  Tensor expected = Tensor::tensor(std::vector<float>{1.5f, 1.5f, 2.f, 3.f, 4.f, 4.f});
  vulkan_test::expect_allclose(back, expected);
}

TEST_F(VulkanOpTest, CopyVulkanToVulkan) {
  Tensor a = tpx_ops::ones({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  Tensor b = tpx_ops::zeros({2, 3}, DType::Float32, Device(DeviceType::Vulkan));
  tpx_ops::copy_(b, a);
  vulkan_test::expect_allclose(b.to(Device(DeviceType::CPU)), tpx_ops::ones({2, 3}));
}

} // namespace
