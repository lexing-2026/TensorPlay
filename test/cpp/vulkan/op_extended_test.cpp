#include "test_utils.h"

#include <gtest/gtest.h>

#include "tensorplay/ops/TPXOpsGenerated.h"

#include "Tensor.h"

#include <cmath>
#include <functional>
#include <vector>

namespace {

using namespace tensorplay;
namespace tpx_ops = tensorplay::tpx::ops;

//
// Tests for the extended op set: matmul, softmax, reductions,
// normalization, pooling, convolution, and shape ops.  Every test pushes a
// CPU reference through the same dispatched op on the CPU and compares
// numerically, so a shader bug shows up as a value mismatch rather than a
// crash.  Ops are invoked through the tpx front end; view ops (select /
// reshape / flatten / expand) materialize through the backend, so their
// results are owning dense tensors that copy out in logical order.
//

class VulkanExtendedOpTest : public ::testing::Test {
 protected:
  void SetUp() override { vulkan_test::skip_if_no_vulkan(); }

  static Tensor vk(const Tensor& cpu_tensor) {
    return cpu_tensor.to(Device(DeviceType::Vulkan));
  }
};

TEST_F(VulkanExtendedOpTest, Mm) {
  Tensor a = Tensor::tensor(
      std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}).reshape({2, 3});
  Tensor b = Tensor::tensor(
      std::vector<float>{7.f, 8.f, 9.f, 10.f, 11.f, 12.f}).reshape({3, 2});
  vulkan_test::expect_allclose(
      tpx_ops::mm(vk(a), vk(b)).to(Device(DeviceType::CPU)), tpx_ops::mm(a, b));
}

TEST_F(VulkanExtendedOpTest, MmNonMultipleOfFour) {
  // Sizes 3 and 5 exercise texel-lane padding on every operand.
  Tensor a = Tensor::tensor(std::vector<float>{
      1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}).reshape({3, 3});
  Tensor b = Tensor::tensor(std::vector<float>{
      1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f,
      14.f, 15.f}).reshape({3, 5});
  vulkan_test::expect_allclose(
      tpx_ops::mm(vk(a), vk(b)).to(Device(DeviceType::CPU)), tpx_ops::mm(a, b));
}

TEST_F(VulkanExtendedOpTest, SoftmaxWidth) {
  Tensor x = Tensor::tensor(
      std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}).reshape({2, 3});
  vulkan_test::expect_allclose(
      tpx_ops::softmax(vk(x), -1, DType::Undefined).to(Device(DeviceType::CPU)), tpx_ops::softmax(x, -1, DType::Undefined));
}

TEST_F(VulkanExtendedOpTest, SoftmaxHeightAndBatch) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});
  vulkan_test::expect_allclose(
      tpx_ops::softmax(vk(x), 1, DType::Undefined).to(Device(DeviceType::CPU)), tpx_ops::softmax(x, 1, DType::Undefined));
  vulkan_test::expect_allclose(
      tpx_ops::softmax(vk(x), 0, DType::Undefined).to(Device(DeviceType::CPU)), tpx_ops::softmax(x, 0, DType::Undefined));
}

TEST_F(VulkanExtendedOpTest, SoftmaxChannel) {
  Tensor x = tpx_ops::arange(30., DType::Float32).reshape({2, 5, 3});
  vulkan_test::expect_allclose(
      tpx_ops::softmax(vk(x), 1, DType::Undefined).to(Device(DeviceType::CPU)), tpx_ops::softmax(x, 1, DType::Undefined));
}

TEST_F(VulkanExtendedOpTest, LogSoftmax) {
  Tensor x = Tensor::tensor(
      std::vector<float>{1.f, 2.f, 3.f, 4.f}).reshape({2, 2});
  vulkan_test::expect_allclose(
      tpx_ops::log_softmax(vk(x), -1, DType::Undefined).to(Device(DeviceType::CPU)), tpx_ops::log_softmax(x, -1, DType::Undefined));
}

TEST_F(VulkanExtendedOpTest, SumDim) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  vulkan_test::expect_allclose(
      tpx_ops::sum(vk(x), {2}, false, DType::Undefined).to(Device(DeviceType::CPU)),
      tpx_ops::sum(x, {2}, false, DType::Undefined));
  vulkan_test::expect_allclose(
      tpx_ops::sum(vk(x), {0}, true, DType::Undefined).to(Device(DeviceType::CPU)),
      tpx_ops::sum(x, {0}, true, DType::Undefined));
  vulkan_test::expect_allclose(
      tpx_ops::sum(vk(x), {0, 2}, false, DType::Undefined).to(Device(DeviceType::CPU)),
      tpx_ops::sum(x, {0, 2}, false, DType::Undefined));
}

TEST_F(VulkanExtendedOpTest, MeanDim) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  vulkan_test::expect_allclose(
      tpx_ops::mean(vk(x), {1}, false, DType::Undefined).to(Device(DeviceType::CPU)),
      tpx_ops::mean(x, {1}, false, DType::Undefined));
  vulkan_test::expect_allclose(
      tpx_ops::mean(vk(x), {1}, true, DType::Undefined).to(Device(DeviceType::CPU)),
      tpx_ops::mean(x, {1}, true, DType::Undefined));
  vulkan_test::expect_allclose(
      tpx_ops::mean(vk(x), {0, 1}, false, DType::Undefined).to(Device(DeviceType::CPU)),
      tpx_ops::mean(x, {0, 1}, false, DType::Undefined));
}

TEST_F(VulkanExtendedOpTest, VarDim) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  vulkan_test::expect_allclose(
      tpx_ops::var(vk(x), {2}, 1, false).to(Device(DeviceType::CPU)),
      tpx_ops::var(x, {2}, 1, false), 1e-4, 1e-5);
  vulkan_test::expect_allclose(
      tpx_ops::var(vk(x), {0, 1}, 0, true).to(Device(DeviceType::CPU)),
      tpx_ops::var(x, {0, 1}, 0, true), 1e-4, 1e-5);
  vulkan_test::expect_allclose(
      tpx_ops::std(vk(x), {2}, 1, false).to(Device(DeviceType::CPU)),
      tpx_ops::std(x, {2}, 1, false), 1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, LayerNormLastDim) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});
  Tensor weight = tpx_ops::ones({4});
  Tensor bias = tpx_ops::zeros({4});

  vulkan_test::expect_allclose(
      tpx_ops::layer_norm(vk(x), {4}, vk(weight), vk(bias), 1e-5).to(Device(DeviceType::CPU)),
      tpx_ops::layer_norm(x, {4}, weight, bias, 1e-5), 1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, LayerNormTwoDimsNoParams) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  vulkan_test::expect_allclose(
      tpx_ops::layer_norm(vk(x), {3, 4}, std::nullopt, std::nullopt, 1e-5)
          .to(Device(DeviceType::CPU)),
      tpx_ops::layer_norm(x, {3, 4}, std::nullopt, std::nullopt, 1e-5),
      1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, BatchNormInference) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 4, 2, 2});
  Tensor mean = Tensor::tensor(std::vector<float>{0.f, 1.f, 2.f, 3.f});
  Tensor var = Tensor::tensor(std::vector<float>{1.f, 2.f, 3.f, 4.f});
  Tensor weight = Tensor::tensor(std::vector<float>{1.f, 1.5f, 2.f, 0.5f});
  Tensor bias = Tensor::tensor(std::vector<float>{0.f, 0.1f, 0.2f, 0.3f});

  vulkan_test::expect_allclose(
      tpx_ops::batch_norm(
            vk(x), vk(weight), vk(bias), vk(mean), vk(var), false, 0.0, 1e-5)
          .to(Device(DeviceType::CPU)),
      tpx_ops::batch_norm(x, weight, bias, mean, var, false, 0.0, 1e-5),
      1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, AvgPool2d) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 1, 4, 4});

  vulkan_test::expect_allclose(
      tpx_ops::avg_pool2d(vk(x), {2, 2}, {2, 2}, {0, 0}).to(Device(DeviceType::CPU)),
      tpx_ops::avg_pool2d(x, {2, 2}, {2, 2}, {0, 0}));
}

TEST_F(VulkanExtendedOpTest, AvgPool2dPadded) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 1, 4, 4});

  vulkan_test::expect_allclose(
      tpx_ops::avg_pool2d(vk(x), {3, 3}, {1, 1}, {1, 1}).to(Device(DeviceType::CPU)),
      tpx_ops::avg_pool2d(x, {3, 3}, {1, 1}, {1, 1}));
}

TEST_F(VulkanExtendedOpTest, MaxPool2d) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 1, 4, 4});

  vulkan_test::expect_allclose(
      tpx_ops::max_pool2d(vk(x), {2, 2}, {2, 2}, {0, 0}, {1, 1}).to(Device(DeviceType::CPU)),
      tpx_ops::max_pool2d(x, {2, 2}, {2, 2}, {0, 0}, {1, 1}));
}

TEST_F(VulkanExtendedOpTest, AdaptiveAvgPool2d) {
  Tensor x = tpx_ops::arange(18., DType::Float32).reshape({1, 2, 3, 3});

  vulkan_test::expect_allclose(
      tpx_ops::adaptive_avg_pool2d(vk(x), {2, 2}).to(Device(DeviceType::CPU)),
      tpx_ops::adaptive_avg_pool2d(x, {2, 2}));
}

TEST_F(VulkanExtendedOpTest, Conv2d) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 1, 4, 4});
  Tensor w = tpx_ops::arange(4., DType::Float32).reshape({1, 1, 2, 2});
  Tensor b = tpx_ops::full({1}, 0.5);

  const std::vector<int64_t> s1{1, 1}, p0{0, 0}, d1{1, 1};
  vulkan_test::expect_allclose(
      tpx_ops::conv2d(vk(x), vk(w), std::optional<Tensor>(vk(b)), s1, p0, d1, 1).to(Device(DeviceType::CPU)),
      tpx_ops::conv2d(x, w, b, s1, p0, d1, 1), 1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, Conv2dStridedPadded) {
  Tensor x = tpx_ops::arange(36., DType::Float32).reshape({1, 1, 6, 6});
  Tensor w = tpx_ops::arange(9., DType::Float32).reshape({1, 1, 3, 3});

  const std::vector<int64_t> s2{2, 2}, p1{1, 1};
  vulkan_test::expect_allclose(
      tpx_ops::conv2d(vk(x), vk(w), std::optional<Tensor>(), s2, p1, p1, 1)
          .to(Device(DeviceType::CPU)),
      tpx_ops::conv2d(x, w, std::nullopt, s2, p1, p1, 1),
      1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, Conv2dMultiChannel) {
  Tensor x = tpx_ops::arange(32., DType::Float32).reshape({1, 2, 4, 4});
  Tensor w = tpx_ops::arange(24., DType::Float32).reshape({3, 2, 2, 2});
  Tensor b = tpx_ops::full({3}, 0.25);

  const std::vector<int64_t> s1{1, 1}, p0{0, 0}, d1{1, 1};
  vulkan_test::expect_allclose(
      tpx_ops::conv2d(vk(x), vk(w), std::optional<Tensor>(vk(b)), s1, p0, d1, 1).to(Device(DeviceType::CPU)),
      tpx_ops::conv2d(x, w, b, s1, p0, d1, 1), 1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, Conv2dDepthwise) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 2, 2, 4});
  Tensor w = tpx_ops::arange(8., DType::Float32).reshape({2, 1, 2, 2});

  const std::vector<int64_t> s1{1, 1}, p0{0, 0}, d1{1, 1};
  vulkan_test::expect_allclose(
      tpx_ops::conv2d(vk(x), vk(w), std::optional<Tensor>(), s1, p0, d1, 2)
          .to(Device(DeviceType::CPU)),
      tpx_ops::conv2d(x, w, std::nullopt, s1, p0, d1, 2),
      1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, ConvTranspose2d) {
  Tensor x = tpx_ops::arange(8., DType::Float32).reshape({1, 1, 2, 4});
  Tensor w = tpx_ops::arange(4., DType::Float32).reshape({1, 1, 2, 2});

  const std::vector<int64_t> s1{1, 1}, p0{0, 0}, op0{0, 0}, d1{1, 1};
  vulkan_test::expect_allclose(
      tpx_ops::conv_transpose2d(
          vk(x), vk(w), std::optional<Tensor>(), s1, p0, op0, 1, d1)
          .to(Device(DeviceType::CPU)),
      tpx_ops::conv_transpose2d(
          x, w, std::nullopt, s1, p0, op0, 1, d1),
      1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, TransposeAndPermute) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  vulkan_test::expect_allclose(
      tpx_ops::transpose(vk(x), 0, 2).to(Device(DeviceType::CPU)), tpx_ops::transpose(x, 0, 2));
  vulkan_test::expect_allclose(
      tpx_ops::permute(vk(x), {2, 0, 1}).to(Device(DeviceType::CPU)), tpx_ops::permute(x, {2, 0, 1}));
}

TEST_F(VulkanExtendedOpTest, Slice) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  vulkan_test::expect_allclose(
      tpx_ops::slice(vk(x), 2, 1, 3).to(Device(DeviceType::CPU)), tpx_ops::slice(x, 2, 1, 3));
  vulkan_test::expect_allclose(
      tpx_ops::slice(vk(x), 0, 0, 2).to(Device(DeviceType::CPU)), tpx_ops::slice(x, 0, 0, 2));
}

TEST_F(VulkanExtendedOpTest, Select) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  // The select op materializes through the backend's view path: the result
  // is a dense payload gathered at the selected index, so the values match
  // the CPU view's logical content.
  vulkan_test::expect_allclose(
      tpx_ops::select(vk(x), 1, 2).to(Device(DeviceType::CPU)),
      tpx_ops::select(x, 1, 2));
  vulkan_test::expect_allclose(
      tpx_ops::select(vk(x), 2, 3).to(Device(DeviceType::CPU)),
      tpx_ops::select(x, 2, 3));
  vulkan_test::expect_allclose(
      tpx_ops::select(vk(x), 0, 1).to(Device(DeviceType::CPU)),
      tpx_ops::select(x, 0, 1));
}

TEST_F(VulkanExtendedOpTest, SelectResultSurvivesCopyOutAndCompute) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});
  Tensor selected = tpx_ops::select(vk(x), 1, 1);

  // Copy-out must read the gathered payload in logical order.
  vulkan_test::expect_allclose(
      selected.to(Device(DeviceType::CPU)), tpx_ops::select(x, 1, 1));

  // Downstream compute on the materialized view stays coherent.
  vulkan_test::expect_allclose(
      tpx_ops::add(selected, selected, Scalar(1)).to(Device(DeviceType::CPU)),
      tpx_ops::add(tpx_ops::select(x, 1, 1), tpx_ops::select(x, 1, 1),
                   Scalar(1)));
}

TEST_F(VulkanExtendedOpTest, ViewAndReshapeMaterialize) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  vulkan_test::expect_allclose(
      tpx_ops::reshape(vk(x), {4, 6}).to(Device(DeviceType::CPU)),
      tpx_ops::reshape(x, {4, 6}));
  vulkan_test::expect_allclose(
      tpx_ops::reshape(vk(x), {24}).to(Device(DeviceType::CPU)),
      tpx_ops::reshape(x, {24}));

  Tensor transposed = tpx_ops::transpose(vk(x), 0, 2);
  vulkan_test::expect_allclose(
      tpx_ops::reshape(transposed, {6, 4}).to(Device(DeviceType::CPU)),
      tpx_ops::reshape(tpx_ops::transpose(x, 0, 2), {6, 4}));
}

TEST_F(VulkanExtendedOpTest, Flatten) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  vulkan_test::expect_allclose(
      tpx_ops::flatten(vk(x), 1, 2).to(Device(DeviceType::CPU)),
      tpx_ops::flatten(x, 1, 2));
  vulkan_test::expect_allclose(
      tpx_ops::flatten(vk(x), 0, -1).to(Device(DeviceType::CPU)),
      tpx_ops::flatten(x, 0, -1));
}

TEST_F(VulkanExtendedOpTest, ExpandBroadcastMaterializes) {
  Tensor x = tpx_ops::arange(4., DType::Float32).reshape({1, 4});

  vulkan_test::expect_allclose(
      tpx_ops::expand(vk(x), {3, 4}).to(Device(DeviceType::CPU)),
      tpx_ops::expand(x, {3, 4}));
}

TEST_F(VulkanExtendedOpTest, AsStridedGathers) {
  Tensor x = tpx_ops::arange(12., DType::Float32).reshape({3, 4});

  // Transposed alias through explicit strides, then materialized densely.
  Tensor strided =
      tpx_ops::as_strided(vk(x), {4, 3}, {1, 4}, std::optional<int64_t>(0));
  vulkan_test::expect_allclose(
      strided.to(Device(DeviceType::CPU)), tpx_ops::transpose(x, 0, 1));
}

TEST_F(VulkanExtendedOpTest, QuantizedArithmeticMatchesCpu) {
  Tensor a_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      Tensor::tensor(std::vector<float>{5.f, 6.f, 7.f, 8.f})
          .to(DType::Int8),
      0.1, 5);
  Tensor b_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      Tensor::tensor(std::vector<float>{2.f, 2.f, 2.f, 2.f})
          .to(DType::Int8),
      0.2, 3);
  Tensor a = vk(a_cpu);
  Tensor b = vk(b_cpu);

  const double as = 0.1, bs = 0.2, os = 0.05;
  const int64_t az = 5, bz = 3, oz = 7;
  vulkan_test::expect_allclose(
      tpx_ops::quantized_add(a, b, as, az, bs, bz, os, oz)
          .to(Device(DeviceType::CPU))
          .dequantize(),
      tpx_ops::quantized_add(a_cpu, b_cpu, as, az, bs, bz, os, oz)
          .dequantize());
  vulkan_test::expect_allclose(
      tpx_ops::quantized_sub(a, b, as, az, bs, bz, os, oz)
          .to(Device(DeviceType::CPU))
          .dequantize(),
      tpx_ops::quantized_sub(a_cpu, b_cpu, as, az, bs, bz, os, oz)
          .dequantize());
  vulkan_test::expect_allclose(
      tpx_ops::quantized_mul(a, b, as, az, bs, bz, os, oz)
          .to(Device(DeviceType::CPU))
          .dequantize(),
      tpx_ops::quantized_mul(a_cpu, b_cpu, as, az, bs, bz, os, oz)
          .dequantize());
  vulkan_test::expect_allclose(
      tpx_ops::quantized_div(a, b, as, az, bs, bz, os, oz)
          .to(Device(DeviceType::CPU))
          .dequantize(),
      tpx_ops::quantized_div(a_cpu, b_cpu, as, az, bs, bz, os, oz)
          .dequantize());

  vulkan_test::expect_allclose(
      tpx_ops::quantized_clamp(
          a, as, az, as, az, Scalar(0.05), Scalar(0.25))
          .to(Device(DeviceType::CPU))
          .dequantize(),
      tpx_ops::quantized_clamp(a_cpu, as, az, as, az, Scalar(0.05),
                               Scalar(0.25))
          .dequantize());
}

TEST_F(VulkanExtendedOpTest, QuantizedArithmeticBroadcasts) {
  // Rank expansion {4} x {2,4} and per-axis broadcast {1,4} x {3,1}.
  Tensor a_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      Tensor::tensor(std::vector<float>{5.f, 6.f, 7.f, 8.f})
          .to(DType::Int8),
      0.1, 5);
  Tensor b_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      Tensor::tensor(
          std::vector<float>{9.f, 8.f, 7.f, 6.f, 1.f, 2.f, 3.f, 4.f})
          .to(DType::Int8)
          .reshape({2, 4}),
      0.1, 5);
  const double as = 0.1, bs = 0.1, os = 0.05;
  const int64_t az = 5, bz = 5, oz = 7;

  vulkan_test::expect_allclose(
      tpx_ops::quantized_add(vk(a_cpu), vk(b_cpu), as, az, bs, bz, os, oz)
          .to(Device(DeviceType::CPU))
          .dequantize(),
      tpx_ops::quantized_add(a_cpu, b_cpu, as, az, bs, bz, os, oz)
          .dequantize());

  Tensor c_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      Tensor::tensor(std::vector<float>{1.f, 2.f, 3.f, 4.f})
          .to(DType::Int8)
          .reshape({1, 4}),
      0.1, 5);
  Tensor d_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      Tensor::tensor(std::vector<float>{1.f, 2.f, 3.f})
          .to(DType::Int8)
          .reshape({3, 1}),
      0.1, 5);
  vulkan_test::expect_allclose(
      tpx_ops::quantized_add(vk(c_cpu), vk(d_cpu), as, az, bs, bz, os, oz)
          .to(Device(DeviceType::CPU))
          .dequantize(),
      tpx_ops::quantized_add(c_cpu, d_cpu, as, az, bs, bz, os, oz)
          .dequantize());
}

TEST_F(VulkanExtendedOpTest, QuantizedMaxPool2dMatchesCpu) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 1, 4, 4});
  Tensor q_cpu = tpx_ops::quantize_per_tensor(x, 0.1, 10);
  Tensor q = vk(q_cpu);

  vulkan_test::expect_allclose(
      tpx_ops::quantized_max_pool2d(q, {2, 2}, {2, 2})
          .to(Device(DeviceType::CPU))
          .dequantize(),
      tpx_ops::quantized_max_pool2d(q_cpu, {2, 2}, {2, 2})
          .dequantize());
}

TEST_F(VulkanExtendedOpTest, QuantizedLinearMatchesCpu) {
  const int64_t M = 5, K = 7, N = 9;
  std::vector<float> xs, ws;
  for (int64_t r = 0; r < M; ++r) {
    for (int64_t k = 0; k < K; ++k) {
      xs.push_back(static_cast<float>(((r * 7 + k) % 61) - 30));
    }
  }
  for (int64_t c = 0; c < N; ++c) {
    for (int64_t k = 0; k < K; ++k) {
      ws.push_back(static_cast<float>(((c * 3 + k) % 41) - 20));
    }
  }
  Tensor x_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      Tensor::tensor(xs).to(DType::Int8).reshape({M, K}), 0.05, 3);
  Tensor w_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      Tensor::tensor(ws).to(DType::Int8).reshape({N, K}), 0.01, -1);
  Tensor w_scales_cpu = Tensor::tensor(
      [] {
        std::vector<float> v;
        for (int64_t n = 0; n < N; ++n) v.push_back(0.01f * (n + 1));
        return v;
      }());
  Tensor w_zps_cpu = Tensor::tensor(
      [] {
        std::vector<float> v;
        for (int64_t n = 0; n < N; ++n) v.push_back(static_cast<float>(n - 4));
        return v;
      }());
  Tensor bias_cpu = Tensor::tensor(
      [] {
        std::vector<float> v;
        for (int64_t n = 0; n < N; ++n) v.push_back(0.001f * n);
        return v;
      }());

  const Tensor expected = tpx_ops::quantized_linear(
      x_cpu, w_cpu, 0.05, 3, w_scales_cpu, w_zps_cpu, bias_cpu);
  const Tensor actual = tpx_ops::quantized_linear(
      vk(x_cpu), vk(w_cpu), 0.05, 3, vk(w_scales_cpu), vk(w_zps_cpu),
      vk(bias_cpu));

  vulkan_test::expect_allclose(
      actual.to(Device(DeviceType::CPU)), expected, 1e-3, 1e-4);
}

TEST_F(VulkanExtendedOpTest, QuantizedConv2dMatchesCpu) {
  // 3x3 single-group conv with padding, bias, and quantization round trip.
  Tensor x_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      tpx_ops::arange(32., DType::Float32)
          .sub(Scalar(15.5))
          .div(Scalar(2.))
          .reshape({1, 2, 4, 4})
          .to(DType::Int8),
      0.05, 3);
  Tensor w_cpu = tpx_ops::_make_per_tensor_quantized_tensor(
      Tensor::tensor(std::vector<float>{
                       1.f, -2.f, 3.f, 0.f, 1.f, 1.f, 2.f, -1.f,
                       1.f, 0.f, -1.f, 2.f, 1.f, 1.f, 1.f, 1.f,
                       -1.f, 2.f, 1.f, -1.f, 1.f, 2.f, -2.f, 0.f,
                       1.f, 1.f, -1.f, 2.f, 0.f, 1.f, 1.f, -1.f,
                       2.f, 1.f, 0.f, 1.f})
          .to(DType::Int8)
          .reshape({2, 2, 3, 3}),
      0.02, -1);
  Tensor bias_cpu = Tensor::tensor(std::vector<float>{0.5f, -0.25f});

  const double is = 0.05, ws = 0.02, os = 0.1;
  const int64_t iz = 3, wz = -1, oz = 7;

  Tensor expected = tpx_ops::quantized_conv2d(
      x_cpu, w_cpu, bias_cpu, is, iz, ws, wz, os, oz, {1, 1}, {1, 1});
  Tensor actual = tpx_ops::quantized_conv2d(
      vk(x_cpu), vk(w_cpu), vk(bias_cpu), is, iz, ws, wz, os, oz, {1, 1},
      {1, 1});

  vulkan_test::expect_allclose(
      actual.to(Device(DeviceType::CPU)).dequantize(),
      expected.dequantize());
}

namespace {

// Builds an Int8 code tensor from float values through an affine grid and
// wraps it with per-tensor qparams, matching the packed-conv prepack input.
Tensor make_qint8_codes(const std::vector<float>& values,
                        const std::vector<int64_t>& sizes, double scale,
                        int64_t zp) {
  Tensor codes = Tensor::tensor(values).to(DType::Int8).reshape(sizes);
  return tpx_ops::_make_per_tensor_quantized_tensor(codes, scale, zp);
}

Tensor make_1d_float(const std::function<float(int64_t)>& gen, int64_t n) {
  std::vector<float> v;
  v.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    v.push_back(gen(i));
  }
  return Tensor::tensor(v);
}

// Round-half-even requantization onto the output affine grid, expressed in
// the dequantized domain so the comparison sees quantized values only.
// The compute shaders requantize with roundEven, so the reference rounds the
// same way before scaling back.
Tensor requantized_grid_reference(const Tensor& acc, double out_scale) {
  Tensor out = acc.to(DType::Float32).clone();
  const float* pa = acc.data_ptr<float>();
  float* po = out.data_ptr<float>();
  for (int64_t i = 0; i < acc.numel(); ++i) {
    po[i] = static_cast<float>(std::nearbyint(
                static_cast<double>(pa[i]) / out_scale)) *
        static_cast<float>(out_scale);
  }
  return out;
}

// Dequantizes Int8 codes with per-channel parameters.  A regular weight
// [O, ...] indexes the parameters along dim 0; a transposed weight
// [in, out, ...] indexes them along dim 1.
Tensor dequantize_weight_per_channel(
    const Tensor& codes, const Tensor& scales, const Tensor& zero_points,
    bool transposed) {
  const int64_t dim0 = codes.size(0);
  const int64_t dim1 = codes.size(1);
  const int64_t per = codes.numel() / (dim0 * dim1);
  Tensor w = codes.contiguous().to(DType::Float32);
  Tensor sc = scales.contiguous().to(DType::Float32);
  Tensor zp = zero_points.contiguous().to(DType::Float32);
  float* pw = w.data_ptr<float>();
  const float* psc = sc.data_ptr<float>();
  const float* pzp = zp.data_ptr<float>();
  for (int64_t a = 0; a < dim0; ++a) {
    for (int64_t b = 0; b < dim1; ++b) {
      const float s = transposed ? psc[b] : psc[a];
      const float z = transposed ? pzp[b] : pzp[a];
      for (int64_t i = 0; i < per; ++i) {
        const int64_t idx = (a * dim1 + b) * per + i;
        pw[idx] = (pw[idx] - z) * s;
      }
    }
  }
  return w;
}

} // namespace

TEST_F(VulkanExtendedOpTest, QuantizedConv2dPrepackUnpackRoundtrip) {
  // The packed payload must rebuild into the original float weight/bias.
  const std::vector<std::array<int64_t, 4>> weight_shapes{
      {3, 5, 3, 3}, {4, 2, 1, 1}, {5, 1, 3, 3}};
  for (const auto& shape : weight_shapes) {
    const int64_t O = shape[0], C = shape[1], KH = shape[2], KW = shape[3];
    std::vector<float> ws(static_cast<size_t>(O * C * KH * KW));
    for (size_t i = 0; i < ws.size(); ++i) {
      ws[i] = static_cast<float>((static_cast<int64_t>(i) % 17) - 8);
    }
    Tensor w_cpu = make_qint8_codes(ws, {O, C, KH, KW}, 0.02, -1);
    Tensor w_scales = make_1d_float([](int64_t i) { return 0.02f + 0.001f * i; }, O);
    Tensor w_zps = make_1d_float([](int64_t i) { return static_cast<float>(i - 2); }, O);
    Tensor bias_cpu = make_1d_float([](int64_t i) { return 0.01f * i; }, O);

    auto [w_packed, b_packed] =
        tpx_ops::quantized_conv2d_prepack(w_cpu, w_scales, w_zps, bias_cpu);

    const bool depthwise = C == 1;
    auto [w_back, b_back] = tpx_ops::quantized_conv2d_unpack(
        w_packed, b_packed, {O, C, KH, KW}, false, depthwise);
    vulkan_test::expect_allclose(
        w_back,
        dequantize_weight_per_channel(w_cpu, w_scales, w_zps, false));
    vulkan_test::expect_allclose(b_back, bias_cpu);
  }

  // Transposed weights arrive as [in, out, KH, KW]; the per-channel
  // parameters index the output channels on dim 1.
  const std::array<int64_t, 4> tshape{4, 3, 2, 2};
  std::vector<float> tws(static_cast<size_t>(4 * 3 * 2 * 2));
  for (size_t i = 0; i < tws.size(); ++i) {
    tws[i] = static_cast<float>((static_cast<int64_t>(i) % 11) - 5);
  }
  Tensor tw_cpu = make_qint8_codes(tws, {4, 3, 2, 2}, 0.05, 3);
  Tensor tw_scales = make_1d_float([](int64_t i) { return 0.03f + 0.002f * i; }, 3);
  Tensor tw_zps = make_1d_float([](int64_t i) { return static_cast<float>(i + 1); }, 3);
  Tensor tb_cpu = make_1d_float([](int64_t i) { return 0.02f * i; }, 3);

  auto [tw_packed, tb_packed] = tpx_ops::quantized_conv2d_prepack(
      tw_cpu, tw_scales, tw_zps, tb_cpu, /*transposed=*/true);
  auto [tw_back, tb_back] = tpx_ops::quantized_conv2d_unpack(
      tw_packed, tb_packed, {4, 3, 2, 2}, /*transposed=*/true,
      /*depthwise=*/false);
  vulkan_test::expect_allclose(
      tw_back, dequantize_weight_per_channel(tw_cpu, tw_scales, tw_zps, true));
  vulkan_test::expect_allclose(tb_back, tb_cpu);
}

TEST_F(VulkanExtendedOpTest, QuantizedConv2dPackedMatchesScalarPath) {
  // The prepack/run path must reproduce the scalar-qparams convolution over
  // every shader selection branch: sliding window, depthwise, 1x1 pointwise,
  // and the transposed gather (with and without output padding).
  const auto run_case = [&](const std::vector<int64_t>& weight_sizes,
                            const std::vector<int64_t>& in_sizes,
                            const std::vector<int64_t>& stride,
                            const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& dilation,
                            const std::vector<int64_t>& output_padding,
                            int64_t groups, bool transposed,
                            double w_scale, int64_t w_zp) {
    const int64_t C = in_sizes[1];
    const int64_t O = transposed ? weight_sizes[1] : weight_sizes[0];
    const int64_t param_ch = transposed ? weight_sizes[1] : weight_sizes[0];
    std::vector<float> wv(static_cast<size_t>(
        weight_sizes[0] * weight_sizes[1] * weight_sizes[2] *
        weight_sizes[3]));
    for (size_t i = 0; i < wv.size(); ++i) {
      wv[i] = static_cast<float>(
          (static_cast<int64_t>(i) % 13) - 6);
    }
    Tensor w_cpu = make_qint8_codes(wv, weight_sizes, w_scale, w_zp);
    Tensor w_scales =
        make_1d_float([&](int64_t i) { return w_scale + 0.0005f * i; }, param_ch);
    Tensor w_zps = make_1d_float(
        [&](int64_t i) { return static_cast<float>(w_zp + (i % 3) - 1); },
        param_ch);
    Tensor bias_cpu = make_1d_float([](int64_t i) { return 0.05f * i; }, O);

    Tensor x_cpu = make_qint8_codes(
        [&] {
          std::vector<float> v;
          const int64_t n = in_sizes[0] * in_sizes[1] * in_sizes[2] *
              in_sizes[3];
          v.reserve(static_cast<size_t>(n));
          for (int64_t i = 0; i < n; ++i) {
            v.push_back(static_cast<float>((i % 19) - 9));
          }
          return v;
        }(),
        in_sizes, 0.1, 4);

    const double os = 0.25;
    const int64_t oz = 100;
    auto [w_packed, b_packed] = tpx_ops::quantized_conv2d_prepack(
        w_cpu, w_scales, w_zps, bias_cpu, transposed);

    Tensor packed = tpx_ops::quantized_conv2d_run(
        vk(x_cpu), w_packed.to(Device(DeviceType::Vulkan)),
        b_packed.to(Device(DeviceType::Vulkan)), weight_sizes, 0.1, 4, os, oz,
        stride, padding, dilation, output_padding, groups, transposed);
    TP_CHECK(
        packed.dtype() == DType::QUInt8,
        "quantized_conv2d_run must produce a QUInt8 tensor");

    // CPU reference: rebuild the float weight through the unpack inverse and
    // run the float convolution, then compare on the requantization grid.
    auto [w_f, b_f] = tpx_ops::quantized_conv2d_unpack(
        w_packed, b_packed, weight_sizes, transposed,
        /*depthwise=*/!transposed && weight_sizes[1] == 1);
    Tensor acc = transposed
        ? tpx_ops::conv_transpose2d(
              x_cpu.dequantize(), w_f, b_f, stride, padding, output_padding,
              groups, dilation)
        : tpx_ops::conv2d(
              x_cpu.dequantize(), w_f, b_f, stride, padding, dilation,
              groups);
    // The shader clamps the accumulator to +-inf (a no-op) and requantizes
    // with round-to-nearest-even onto the output grid; compare in the
    // dequantized domain so the test only pins the grid mapping.
    Tensor grid = requantized_grid_reference(acc, os);

    vulkan_test::expect_allclose(
        packed.to(Device(DeviceType::CPU)).dequantize(),
        grid, 1e-5, os * 0.51);
  };

  // Sliding window: 3x3, padding, stride.
  run_case({3, 2, 3, 3}, {1, 2, 5, 5}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, 1,
           false, 0.02, -1);
  run_case({2, 2, 3, 3}, {1, 2, 6, 6}, {2, 2}, {1, 1}, {1, 1}, {0, 0}, 1,
           false, 0.02, -1);
  // Depthwise: groups == out channels, one input channel each.
  run_case({4, 1, 3, 3}, {1, 4, 6, 6}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, 4,
           false, 0.03, 2);
  // Pointwise 1x1.
  run_case({5, 4, 1, 1}, {1, 4, 5, 5}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, 1,
           false, 0.02, 0);
  // Transposed: 2x2 gather with stride 2, plus output padding.
  run_case({2, 3, 2, 2}, {1, 2, 3, 3}, {2, 2}, {0, 0}, {1, 1}, {0, 0}, 1,
           true, 0.04, 1);
  run_case({2, 3, 3, 3}, {1, 2, 4, 4}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, 1,
           true, 0.04, 1);
}

TEST_F(VulkanExtendedOpTest, QuantizedQuint8Roundtrip) {
  Tensor x = tpx_ops::arange(12., DType::Float32).sub(Scalar(5.)).div(Scalar(2.));
  const double scale = 0.1;
  const int64_t zp = 25;

  Tensor q_cpu = tpx_ops::quantize_per_tensor_quint8(x, scale, zp);
  TP_CHECK(q_cpu.dtype() == DType::QUInt8, "quint8 codes must be QUInt8");
  Tensor q = vk(q_cpu);

  vulkan_test::expect_allclose(
      tpx_ops::dequantize_per_tensor_quint8(q, scale, zp)
          .to(Device(DeviceType::CPU)),
      tpx_ops::dequantize_per_tensor_quint8(q_cpu, scale, zp));

  // Round-trip error stays within one quantization step.
  Tensor back =
      tpx_ops::dequantize_per_tensor_quint8(q_cpu, scale, zp);
  vulkan_test::expect_allclose(back, x, 1e-5, scale);
}

TEST_F(VulkanExtendedOpTest, QuantizedQint32Roundtrip) {
  Tensor x = tpx_ops::arange(12., DType::Float32).sub(Scalar(5.)).div(Scalar(2.));
  const double scale = 1e-6;
  const int64_t zp = 100;

  Tensor q_cpu = tpx_ops::quantize_per_tensor_qint32(x, scale, zp);
  TP_CHECK(q_cpu.dtype() == DType::QInt32, "qint32 codes must be QInt32");
  Tensor q = vk(q_cpu);

  vulkan_test::expect_allclose(
      tpx_ops::dequantize_per_tensor_qint32(q, scale, zp)
          .to(Device(DeviceType::CPU)),
      tpx_ops::dequantize_per_tensor_qint32(q_cpu, scale, zp));

  // The wider code makes the round trip far tighter than one Int8 step.
  Tensor back =
      tpx_ops::dequantize_per_tensor_qint32(q_cpu, scale, zp);
  vulkan_test::expect_allclose(back, x, 1e-5, 1e-5);
}

TEST_F(VulkanExtendedOpTest, ViewMethodFormsMaterialize) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});
  Tensor xv = vk(x);

  // The member forms build views through as_strided; on the Vulkan device
  // they must materialize exactly like the dispatched front-end ops.
  vulkan_test::expect_allclose(
      xv.select(1, 2).to(Device(DeviceType::CPU)),
      tpx_ops::select(x, 1, 2));
  vulkan_test::expect_allclose(
      xv.slice(2, 1, 3).to(Device(DeviceType::CPU)),
      tpx_ops::slice(x, 2, 1, 3));
  vulkan_test::expect_allclose(
      xv.view({4, 6}).to(Device(DeviceType::CPU)),
      tpx_ops::reshape(x, {4, 6}));
  vulkan_test::expect_allclose(
      xv.view({4, 6}).view({24}).to(Device(DeviceType::CPU)),
      tpx_ops::reshape(x, {24}));
  vulkan_test::expect_allclose(
      xv.as_strided({1, 3, 4}, {12, 4, 1}, std::optional<int64_t>(12))
          .to(Device(DeviceType::CPU)),
      tpx_ops::slice(x, 0, 1, 2));
  vulkan_test::expect_allclose(
      xv.transpose(0, 2).view({2, 12}).to(Device(DeviceType::CPU)),
      tpx_ops::reshape(tpx_ops::transpose(x, 0, 2), {2, 12}));
}

TEST_F(VulkanExtendedOpTest, UnsqueezeSqueeze) {
  Tensor x = tpx_ops::arange(12., DType::Float32).reshape({3, 4});

  Tensor unsq = tpx_ops::unsqueeze(vk(x), 0);
  ASSERT_EQ(unsq.dim(), 3);
  vulkan_test::expect_allclose(unsq.to(Device(DeviceType::CPU)), x.reshape({1, 3, 4}));

  vulkan_test::expect_allclose(
      tpx_ops::squeeze(unsq, 0).to(Device(DeviceType::CPU)), x);
}

TEST_F(VulkanExtendedOpTest, Cat) {
  Tensor a = tpx_ops::arange(12., DType::Float32).reshape({2, 1, 2, 3});
  Tensor b = tpx_ops::full({2, 1, 2, 3}, 100.);

  vulkan_test::expect_allclose(
      tpx_ops::cat({vk(a), vk(b)}, 1).to(Device(DeviceType::CPU)), tpx_ops::cat({a, b}, 1));
  vulkan_test::expect_allclose(
      tpx_ops::cat({vk(a), vk(b)}, 3).to(Device(DeviceType::CPU)), tpx_ops::cat({a, b}, 3));
  vulkan_test::expect_allclose(
      tpx_ops::cat({vk(a), vk(b)}, 0).to(Device(DeviceType::CPU)), tpx_ops::cat({a, b}, 0));
}

TEST_F(VulkanExtendedOpTest, CatChannels) {
  Tensor a = tpx_ops::arange(16., DType::Float32).reshape({2, 2, 2, 2});
  Tensor b = tpx_ops::full({2, 3, 2, 2}, 7.);

  vulkan_test::expect_allclose(
      tpx_ops::cat({vk(a), vk(b)}, 1).to(Device(DeviceType::CPU)), tpx_ops::cat({a, b}, 1));
}

TEST_F(VulkanExtendedOpTest, Stack) {
  Tensor a = tpx_ops::arange(12., DType::Float32).reshape({3, 4});
  Tensor b = tpx_ops::full({3, 4}, 5.);

  vulkan_test::expect_allclose(
      tpx_ops::stack({vk(a), vk(b)}, 0).to(Device(DeviceType::CPU)), tpx_ops::stack({a, b}, 0));
}

TEST_F(VulkanExtendedOpTest, Pow) {
  Tensor x = Tensor::tensor(std::vector<float>{1.f, 2.f, 3.f, 4.f});
  Tensor p = Tensor::tensor(std::vector<float>{2.f, 0.5f, 2.f, 1.f});

  vulkan_test::expect_allclose(
      tpx_ops::pow(vk(x), vk(p)).to(Device(DeviceType::CPU)), tpx_ops::pow(x, p));
  vulkan_test::expect_allclose(
      tpx_ops::pow(vk(x), Scalar(2.)).to(Device(DeviceType::CPU)), tpx_ops::pow(x, Scalar(2.)));
}

TEST_F(VulkanExtendedOpTest, Lerp) {
  Tensor x = Tensor::tensor(std::vector<float>{0.f, 2.f, 4.f, 6.f});
  Tensor e = Tensor::tensor(std::vector<float>{1.f, 1.f, 1.f, 1.f});

  vulkan_test::expect_allclose(
      tpx_ops::lerp(vk(x), vk(e), Scalar(0.5)).to(Device(DeviceType::CPU)),
      tpx_ops::lerp(x, e, Scalar(0.5)));
  vulkan_test::expect_allclose(
      tpx_ops::lerp(vk(x), vk(e), vk(e)).to(Device(DeviceType::CPU)), tpx_ops::lerp(x, e, e));
}

TEST_F(VulkanExtendedOpTest, Flip) {
  Tensor x = tpx_ops::arange(24., DType::Float32).reshape({2, 3, 4});

  vulkan_test::expect_allclose(
      tpx_ops::flip(vk(x), {2}).to(Device(DeviceType::CPU)), tpx_ops::flip(x, {2}));
  vulkan_test::expect_allclose(
      tpx_ops::flip(vk(x), {0, 1}).to(Device(DeviceType::CPU)), tpx_ops::flip(x, {0, 1}));
}

TEST_F(VulkanExtendedOpTest, UpsampleNearest2d) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 1, 4, 4});

  vulkan_test::expect_allclose(
      tpx_ops::upsample_nearest2d(vk(x), {8, 8}).to(Device(DeviceType::CPU)),
      tpx_ops::upsample_nearest2d(x, {8, 8}));
}

TEST_F(VulkanExtendedOpTest, ReflectionPad2d) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 1, 4, 4});

  vulkan_test::expect_allclose(
      tpx_ops::reflection_pad_nd(vk(x), {1, 1, 1, 1}).to(Device(DeviceType::CPU)),
      tpx_ops::reflection_pad_nd(x, {1, 1, 1, 1}));
}

TEST_F(VulkanExtendedOpTest, ReplicationPad2d) {
  Tensor x = tpx_ops::arange(16., DType::Float32).reshape({1, 1, 4, 4});

  vulkan_test::expect_allclose(
      tpx_ops::replication_pad_nd(vk(x), {1, 1, 0, 0}).to(Device(DeviceType::CPU)),
      tpx_ops::replication_pad_nd(x, {1, 1, 0, 0}));
}

TEST_F(VulkanExtendedOpTest, Activations) {
  Tensor x = Tensor::tensor(std::vector<float>{-4.f, -1.f, 0.f, 1.f, 4.f, 8.f});

  vulkan_test::expect_allclose(
      tpx_ops::silu(vk(x)).to(Device(DeviceType::CPU)), tpx_ops::silu(x),
      1e-4, 1e-5);
  vulkan_test::expect_allclose(
      tpx_ops::mish(vk(x)).to(Device(DeviceType::CPU)), tpx_ops::mish(x),
      1e-4, 1e-5);
  vulkan_test::expect_allclose(
      tpx_ops::relu6(vk(x)).to(Device(DeviceType::CPU)), tpx_ops::relu6(x));
  vulkan_test::expect_allclose(
      tpx_ops::hardsigmoid(vk(x)).to(Device(DeviceType::CPU)),
      tpx_ops::hardsigmoid(x), 1e-4, 1e-5);
  vulkan_test::expect_allclose(
      tpx_ops::hardswish(vk(x)).to(Device(DeviceType::CPU)),
      tpx_ops::hardswish(x), 1e-4, 1e-5);
  vulkan_test::expect_allclose(
      tpx_ops::gelu(vk(x)).to(Device(DeviceType::CPU)), tpx_ops::gelu(x),
      1e-4, 1e-5);
  vulkan_test::expect_allclose(
      tpx_ops::gelu(vk(x), "tanh").to(Device(DeviceType::CPU)),
      tpx_ops::gelu(x, "tanh"), 1e-4, 1e-5);
  vulkan_test::expect_allclose(
      tpx_ops::leaky_relu(vk(x), Scalar(0.2)).to(Device(DeviceType::CPU)),
      tpx_ops::leaky_relu(x, Scalar(0.2)));
  vulkan_test::expect_allclose(
      tpx_ops::threshold(vk(x), Scalar(0.0), Scalar(-1.0))
          .to(Device(DeviceType::CPU)),
      tpx_ops::threshold(x, Scalar(0.0), Scalar(-1.0)));
  vulkan_test::expect_allclose(
      tpx_ops::hardshrink(vk(x), Scalar(0.5)).to(Device(DeviceType::CPU)),
      tpx_ops::hardshrink(x, Scalar(0.5)));

  // In-place variants run on the same payload.
  Tensor y = vk(x);
  tpx_ops::silu_(y);
  vulkan_test::expect_allclose(
      y.to(Device(DeviceType::CPU)), tpx_ops::silu(x), 1e-4, 1e-5);

  Tensor z = vk(x);
  tpx_ops::hardswish_(z);
  vulkan_test::expect_allclose(
      z.to(Device(DeviceType::CPU)), tpx_ops::hardswish(x), 1e-4, 1e-5);
}

TEST_F(VulkanExtendedOpTest, RandomFillsAreInRangeAndDeterministic) {
  const std::vector<int64_t> sizes{2, 3, 4};
  Tensor a = tpx_ops::empty(sizes, std::optional<DType>(DType::Float32),
                                std::optional<Device>(Device(DeviceType::Vulkan)));
  Tensor b = tpx_ops::empty(sizes, std::optional<DType>(DType::Float32),
                                std::optional<Device>(Device(DeviceType::Vulkan)));

  // Deterministic streams: identical payload shape drawn twice through the
  // same seeded generator state reproduces identical values.
  tensorplay::manual_seed(1234);
  tpx_ops::uniform_(a, -2.0, 5.0);
  tensorplay::manual_seed(1234);
  Tensor ref = tpx_ops::zeros(sizes, std::optional<DType>(DType::Float32));
  tpx_ops::uniform_(ref, -2.0, 5.0);
  vulkan_test::expect_allclose(a.to(Device(DeviceType::CPU)), ref);

  // Range contract: [from, to).
  const float* data = a.to(Device(DeviceType::CPU)).data_ptr<float>();
  for (int64_t i = 0; i < a.numel(); ++i) {
    ASSERT_GE(data[i], -2.0f);
    ASSERT_LT(data[i], 5.0f);
  }

  (void)b;
}

TEST_F(VulkanExtendedOpTest, NormalFillStatistics) {
  tensorplay::manual_seed(7);
  const std::vector<int64_t> sizes{40, 40};
  Tensor a = tpx_ops::empty(sizes, std::optional<DType>(DType::Float32),
                                std::optional<Device>(Device(DeviceType::Vulkan)));
  tpx_ops::normal_(a, 3.0, 2.0);
  Tensor host = a.to(Device(DeviceType::CPU));
  const float* data = host.data_ptr<float>();

  double sum = 0.0;
  for (int64_t i = 0; i < a.numel(); ++i) {
    sum += data[i];
  }
  const double mean = sum / a.numel();
  EXPECT_NEAR(mean, 3.0, 0.3);
}

TEST_F(VulkanExtendedOpTest, BernoulliFill) {
  tensorplay::manual_seed(99);
  const std::vector<int64_t> sizes{10, 100};
  Tensor a = tpx_ops::empty(sizes, std::optional<DType>(DType::Float32),
                                std::optional<Device>(Device(DeviceType::Vulkan)));
  tpx_ops::bernoulli_(a, 0.7);
  Tensor host = a.to(Device(DeviceType::CPU));
  const float* data = host.data_ptr<float>();

  int64_t ones = 0;
  for (int64_t i = 0; i < a.numel(); ++i) {
    ASSERT_TRUE(data[i] == 0.0f || data[i] == 1.0f) << " index " << i;
    ones += (data[i] == 1.0f);
  }
  const double ratio = static_cast<double>(ones) / a.numel();
  EXPECT_NEAR(ratio, 0.7, 0.05);
}

TEST_F(VulkanExtendedOpTest, QuantizeRoundtrip) {
  Tensor x = Tensor::tensor(std::vector<float>{
      -12.5f, -3.2f, 0.f, 0.4f, 3.7f, 12.1f, 120.f, -120.f});
  const double scale = 1.0;
  const int64_t zp = 0;

  Tensor q = tpx_ops::quantize_per_tensor(vk(x), scale, zp);
  ASSERT_EQ(q.dtype(), DType::QInt8);

  Tensor back = tpx_ops::dequantize_per_tensor(q, scale, zp)
                    .to(Device(DeviceType::CPU));
  vulkan_test::expect_allclose(
      back, tpx_ops::dequantize_per_tensor(
                tpx_ops::quantize_per_tensor(x, scale, zp), scale, zp));
}

} // namespace
