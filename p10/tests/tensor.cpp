// Tensor core semantics: creation, shape/stride accessors, views, contiguity.

#include <gtest/gtest.h>

#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>

using namespace tensorplay;
namespace ops = tensorplay::tpx::ops;

namespace {
template <typename T>
std::vector<T> values(const Tensor& t) {
    const T* data = t.data_ptr<T>();
    return std::vector<T>(data, data + t.numel());
}
} // namespace

TEST(TensorTest, CreationAndMetadata) {
    Tensor t = ops::zeros({2, 3}, DType::Float32, Device(DeviceType::CPU));
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.shape(), Size({2, 3}));
    EXPECT_EQ(t.dtype(), DType::Float32);
    EXPECT_EQ(t.device().type(), DeviceType::CPU);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.strides(), (std::vector<int64_t>{3, 1}));
    // Zero-initialized storage.
    for (float v : values<float>(t)) {
        EXPECT_EQ(v, 0.0f);
    }
}

TEST(TensorTest, EmptyTensorHasZeroNumel) {
    Tensor t = ops::empty({0, 4}, DType::Float32, Device(DeviceType::CPU));
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.dim(), 2);
}

TEST(TensorTest, ArangeValues) {
    Tensor t = ops::arange(Scalar(0), Scalar(6), Scalar(1), DType::Int64, Device(DeviceType::CPU));
    EXPECT_EQ(t.numel(), 6);
    std::vector<int64_t> expected = {0, 1, 2, 3, 4, 5};
    EXPECT_EQ(values<int64_t>(t), expected);
}

TEST(TensorTest, TransposeViewSharesStorage) {
    Tensor t = ops::reshape(ops::arange(Scalar(0), Scalar(6), Scalar(1), DType::Int64, Device(DeviceType::CPU)), {2, 3});
    Tensor v = ops::transpose(t, 0, 1);
    EXPECT_EQ(v.shape(), Size({3, 2}));
    EXPECT_EQ(v.strides(), (std::vector<int64_t>{1, 3}));
    EXPECT_FALSE(v.is_contiguous());
    // Views alias the same storage: writes through one are visible via the other.
    v.data_ptr<int64_t>()[0] = 99;
    EXPECT_EQ(t.data_ptr<int64_t>()[0], 99);
}

TEST(TensorTest, ExpandReportsZeroStride) {
    Tensor t = ops::ones({2, 1, 4}, DType::Float32, Device(DeviceType::CPU));
    Tensor e = ops::expand(t, {2, 3, 4});
    EXPECT_EQ(e.shape(), Size({2, 3, 4}));
    EXPECT_EQ(e.strides(), (std::vector<int64_t>{4, 0, 1}));
    EXPECT_EQ(e.numel(), 24);
}

TEST(TensorTest, CloneIsIndependent) {
    Tensor t = ops::ones({4}, DType::Float32, Device(DeviceType::CPU));
    Tensor c = ops::clone(t);
    c.data_ptr<float>()[0] = 7.0f;
    EXPECT_EQ(t.data_ptr<float>()[0], 1.0f);
}

TEST(TensorTest, MatmulValues) {
    Tensor a = ops::full({2, 2}, Scalar(2.0), DType::Float32, Device(DeviceType::CPU));
    Tensor b = ops::full({2, 2}, Scalar(3.0), DType::Float32, Device(DeviceType::CPU));
    Tensor r = ops::matmul(a, b);
    EXPECT_EQ(r.shape(), Size({2, 2}));
    // Each output element is 2*3 + 2*3.
    for (float v : values<float>(r)) {
        EXPECT_FLOAT_EQ(v, 12.0f);
    }
}
