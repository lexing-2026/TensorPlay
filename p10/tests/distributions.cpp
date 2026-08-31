// In-place sampling kernel semantics: stride handling, range validation,
// dtype coverage and error messages.

#include <gtest/gtest.h>

#include "DType.h"
#include "Exception.h"
#include "FileCheck.h"
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <complex>
#include <vector>

using namespace tensorplay;
namespace ops = tensorplay::tpx::ops;

namespace {

template <typename T>
std::vector<T> values(const Tensor& t) {
    const T* data = t.data_ptr<T>();
    return std::vector<T>(data, data + t.numel());
}

// Reads each element widened to double, dispatching on the storage dtype so
// narrow types are never reinterpreted as wider ones.
template <typename Fn>
void for_each_as_double(const Tensor& t, Fn&& fn) {
    switch (t.dtype()) {
        case DType::Bool:
            for (bool v : values<bool>(t)) fn(static_cast<double>(v));
            break;
        case DType::UInt8:
            for (uint8_t v : values<uint8_t>(t)) fn(static_cast<double>(v));
            break;
        case DType::Int8:
            for (int8_t v : values<int8_t>(t)) fn(static_cast<double>(v));
            break;
        case DType::Int16:
            for (int16_t v : values<int16_t>(t)) fn(static_cast<double>(v));
            break;
        case DType::Int32:
            for (int32_t v : values<int32_t>(t)) fn(static_cast<double>(v));
            break;
        case DType::Int64:
            for (int64_t v : values<int64_t>(t)) fn(static_cast<double>(v));
            break;
        case DType::Float16:
            for (Half v : values<Half>(t)) fn(static_cast<double>(static_cast<float>(v)));
            break;
        case DType::BFloat16:
            for (BFloat16 v : values<BFloat16>(t)) fn(static_cast<double>(static_cast<float>(v)));
            break;
        case DType::Float32:
            for (float v : values<float>(t)) fn(static_cast<double>(v));
            break;
        case DType::Float64:
            for (double v : values<double>(t)) fn(v);
            break;
        default:
            TP_THROW(RuntimeError, "unsupported dtype in test helper");
    }
}

} // namespace

TEST(DistributionTest, UniformFillsContiguousRange) {
    Tensor t = ops::empty({64}, DType::Float32, Device(DeviceType::CPU));
    ops::uniform_(t, -2.0, 5.0);
    for (float v : values<float>(t)) {
        EXPECT_GE(v, -2.0f);
        EXPECT_LT(v, 5.0f);
    }
}

TEST(DistributionTest, UniformOnTransposedViewWritesRespectingStrides) {
    // A transposed view is non-contiguous but fully addressable: every
    // element must be independently drawn and stay in range.
    Tensor t = ops::empty({4, 3}, DType::Float32, Device(DeviceType::CPU));
    Tensor v = ops::transpose(t, 0, 1);
    ops::uniform_(v, 0.0, 1.0);
    EXPECT_EQ(v.strides(), (std::vector<int64_t>{1, 3}));
    for (float x : values<float>(t)) {
        EXPECT_GE(x, 0.0f);
        EXPECT_LT(x, 1.0f);
    }
}

TEST(DistributionTest, UniformOnOverlappingViewIsRejected) {
    // A stride-0 dimension aliases elements, making the draw order
    // observable; overlapping in-place writes are rejected.
    Tensor t = ops::ones({2, 1, 4}, DType::Float32, Device(DeviceType::CPU));
    Tensor e = ops::expand(t, {2, 3, 4});
    bool threw = false;
    try {
        ops::uniform_(e, 0.0, 1.0);
    } catch (const RuntimeError& err) {
        threw = true;
        FileCheck().check("more than one element")->run(err.msg());
        FileCheck().check("Please clone() the tensor")->run(err.msg());
    }
    EXPECT_TRUE(threw);
}

TEST(DistributionTest, UniformHandlesComplexComponents) {
    Tensor t = ops::empty({32}, DType::ComplexFloat, Device(DeviceType::CPU));
    ops::uniform_(t, -1.0, 1.0);
    const auto* data = static_cast<const std::complex<float>*>(t.data_ptr());
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_GE(data[i].real(), -1.0f);
        EXPECT_LT(data[i].real(), 1.0f);
        EXPECT_GE(data[i].imag(), -1.0f);
        EXPECT_LT(data[i].imag(), 1.0f);
    }
}

TEST(DistributionTest, UniformEmptyTensorIsANoop) {
    Tensor t = ops::empty({0}, DType::Float32, Device(DeviceType::CPU));
    ops::uniform_(t, 0.0, 1.0);
    EXPECT_EQ(t.numel(), 0);
}

TEST(DistributionTest, UniformRangeValidationMessages) {
    Tensor t = ops::empty({4}, DType::Float32, Device(DeviceType::CPU));
    {
        bool threw = false;
        try {
            ops::uniform_(t, 10.0, 0.0);
        } catch (const RuntimeError& err) {
            threw = true;
            FileCheck().check("uniform_ expects to return a [from, to) range")->run(err.msg());
            FileCheck().check("from=10 > to=0")->run(err.msg());
        }
        EXPECT_TRUE(threw);
    }
    {
        bool threw = false;
        try {
            ops::uniform_(t, 0.0, 3.5e38);
        } catch (const RuntimeError& err) {
            threw = true;
            FileCheck().check("to is out of bounds for float")->run(err.msg());
        }
        EXPECT_TRUE(threw);
    }
    // from == to is legal and fills the constant.
    Tensor z = ops::empty({3}, DType::Float32, Device(DeviceType::CPU));
    ops::uniform_(z, 3.0, 3.0);
    for (float v : values<float>(z)) {
        EXPECT_FLOAT_EQ(v, 3.0f);
    }
}

TEST(DistributionTest, UniformIntegralDtypeIsNotImplemented) {
    Tensor t = ops::empty({4}, DType::Int64, Device(DeviceType::CPU));
    bool threw = false;
    try {
        ops::uniform_(t, 0.0, 10.0);
    } catch (const NotImplementedError& err) {
        threw = true;
        FileCheck().check("\"check_uniform_bounds\" not implemented for 'Long'")->run(err.msg());
    }
    EXPECT_TRUE(threw);
}

TEST(DistributionTest, RandomFillsAllIntegralDtypes) {
    for (DType dtype : {DType::UInt8, DType::Int8, DType::Int16, DType::Int32,
                        DType::Int64}) {
        Tensor t = ops::empty({64}, dtype, Device(DeviceType::CPU));
        ops::random_(t, 0, 10);
        for_each_as_double(t, [](double v) {
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 10.0);
        });
    }
    // Bool only supports the [0, 2) interval.
    Tensor b = ops::empty({64}, DType::Bool, Device(DeviceType::CPU));
    ops::random_(b, 0, 2);
    for_each_as_double(b, [](double v) {
        EXPECT_GE(v, 0.0);
        EXPECT_LT(v, 2.0);
    });
}

TEST(DistributionTest, RandomRangeValidationMessages) {
    Tensor t = ops::empty({4}, DType::Int64, Device(DeviceType::CPU));
    {
        bool threw = false;
        try {
            ops::random_(t, 5, 5);
        } catch (const RuntimeError& err) {
            threw = true;
            FileCheck().check("random_ expects 'from' to be less than 'to'")->run(err.msg());
        }
        EXPECT_TRUE(threw);
    }
    {
        // 200 is not representable in Int8.
        Tensor i8 = ops::empty({4}, DType::Int8, Device(DeviceType::CPU));
        bool threw = false;
        try {
            ops::random_(i8, 200, 201);
        } catch (const RuntimeError& err) {
            threw = true;
            FileCheck().check("from is out of bounds for signed char")->run(err.msg());
        }
        EXPECT_TRUE(threw);
    }
    {
        // to - 1 = 4 exceeds the representable range of Bool.
        Tensor b = ops::empty({4}, DType::Bool, Device(DeviceType::CPU));
        bool threw = false;
        try {
            ops::random_(b, 0, 5);
        } catch (const RuntimeError& err) {
            threw = true;
            FileCheck().check("to - 1 is out of bounds for bool")->run(err.msg());
        }
        EXPECT_TRUE(threw);
    }
}

TEST(DistributionTest, RandomOnTransposedViewWritesRespectingStrides) {
    Tensor t = ops::empty({4, 3}, DType::Int64, Device(DeviceType::CPU));
    Tensor v = ops::transpose(t, 0, 1);
    ops::random_(v, 3, 10);
    for (int64_t x : values<int64_t>(t)) {
        EXPECT_GE(x, 3);
        EXPECT_LT(x, 10);
    }
}

TEST(DistributionTest, RandintCoversAllDtypes) {
    for (DType dtype : {DType::Bool, DType::UInt8, DType::Int8, DType::Int16,
                        DType::Int32, DType::Int64, DType::Float16,
                        DType::BFloat16, DType::Float32, DType::Float64}) {
        Tensor t = ops::randint(0, 2, {16}, dtype, Device(DeviceType::CPU));
        EXPECT_EQ(t.dtype(), dtype);
        for_each_as_double(t, [](double v) {
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 2.0);
        });
    }
}

TEST(DistributionTest, RandintRangeValidationMessages) {
    {
        bool threw = false;
        try {
            ops::randint(5, 5, {4}, DType::Int64, Device(DeviceType::CPU));
        } catch (const RuntimeError& err) {
            threw = true;
            FileCheck().check("randint expects 'from' to be less than 'to'")->run(err.msg());
        }
        EXPECT_TRUE(threw);
    }
    {
        bool threw = false;
        try {
            ops::randint(0, 9, {4}, DType::Bool, Device(DeviceType::CPU));
        } catch (const RuntimeError& err) {
            threw = true;
            FileCheck().check("to - 1 is out of bounds for bool")->run(err.msg());
        }
        EXPECT_TRUE(threw);
    }
}

TEST(DistributionTest, IscloseUsesFullComplexValue) {
    // Equal real parts with a large imaginary gap: a real-only comparison
    // would wrongly report closeness, so zero tolerances discriminate.
    Tensor base = ops::full({1}, Scalar(std::complex<float>(1.0f, 2.0f)),
                            DType::ComplexFloat, Device(DeviceType::CPU));
    Tensor identical = base.clone();
    Tensor other = ops::full({1}, Scalar(std::complex<float>(1.0f, 3.0f)),
                             DType::ComplexFloat, Device(DeviceType::CPU));

    Tensor close_identical = ops::isclose(base, identical, 0.0, 0.0, false);
    EXPECT_TRUE(close_identical.data_ptr<bool>()[0]);
    Tensor close_other = ops::isclose(base, other, 0.0, 0.0, false);
    EXPECT_FALSE(close_other.data_ptr<bool>()[0]);
}

TEST(DistributionTest, IscloseValidatesDtypeAndTolerances) {
    Tensor f32 = ops::ones({2}, DType::Float32, Device(DeviceType::CPU));
    Tensor f64 = ops::ones({2}, DType::Float64, Device(DeviceType::CPU));
    {
        bool threw = false;
        try {
            ops::isclose(f32, f64, 1e-5, 1e-8, false);
        } catch (const RuntimeError& err) {
            threw = true;
            FileCheck().check("Float did not match Double")->run(err.msg());
        }
        EXPECT_TRUE(threw);
    }
    {
        bool threw = false;
        try {
            ops::isclose(f32, f32, -1e-5, 1e-8, false);
        } catch (const RuntimeError& err) {
            threw = true;
            FileCheck().check("rtol must be greater than or equal to zero")->run(err.msg());
        }
        EXPECT_TRUE(threw);
    }
}
