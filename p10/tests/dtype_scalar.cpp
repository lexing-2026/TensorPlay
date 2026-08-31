// DType classification, names, sizes and Scalar conversions.

#include <gtest/gtest.h>

#include "DType.h"
#include "DistributionDispatch.h"
#include "Scalar.h"

using namespace tensorplay;

TEST(DTypeTest, Classifiers) {
    EXPECT_TRUE(isFloatingType(DType::Float32));
    EXPECT_TRUE(isFloatingType(DType::Float64));
    EXPECT_TRUE(isFloatingType(DType::Float16));
    EXPECT_TRUE(isFloatingType(DType::BFloat16));
    EXPECT_FALSE(isFloatingType(DType::Int64));
    EXPECT_FALSE(isFloatingType(DType::Bool));

    EXPECT_TRUE(isIntegralType(DType::Int64));
    EXPECT_TRUE(isIntegralType(DType::Int8));
    EXPECT_FALSE(isIntegralType(DType::Bool));
    EXPECT_TRUE(isIntegralType(DType::Bool, /*includeBool=*/true));

    EXPECT_TRUE(isComplexType(DType::ComplexFloat));
    EXPECT_TRUE(isComplexType(DType::ComplexDouble));
    EXPECT_FALSE(isComplexType(DType::Float32));
}

TEST(DTypeTest, MessageSpellings) {
    // The "not implemented for 'X'" family uses enum-style spellings.
    EXPECT_STREQ(toString(DType::Int64), "Long");
    EXPECT_STREQ(toString(DType::Int8), "Char");
    EXPECT_STREQ(toString(DType::Float32), "Float");
    EXPECT_STREQ(toString(DType::BFloat16), "BFloat16");
    // The "out of bounds for X" family uses C-style spellings.
    using distribution::bounds_dtype_name;
    EXPECT_STREQ(bounds_dtype_name(DType::Int8), "signed char");
    EXPECT_STREQ(bounds_dtype_name(DType::UInt8), "unsigned char");
    EXPECT_STREQ(bounds_dtype_name(DType::Int64), "long");
    EXPECT_STREQ(bounds_dtype_name(DType::Float32), "float");
    EXPECT_STREQ(bounds_dtype_name(DType::Float64), "double");
    EXPECT_STREQ(bounds_dtype_name(DType::Float16), "c10::Half");
    EXPECT_STREQ(bounds_dtype_name(DType::Bool), "bool");
}

TEST(DTypeTest, ElementSizes) {
    EXPECT_EQ(elementSize(DType::Bool), 1u);
    EXPECT_EQ(elementSize(DType::Int8), 1u);
    EXPECT_EQ(elementSize(DType::Int32), 4u);
    EXPECT_EQ(elementSize(DType::Int64), 8u);
    EXPECT_EQ(elementSize(DType::Float32), 4u);
    EXPECT_EQ(elementSize(DType::Float64), 8u);
    EXPECT_EQ(elementSize(DType::ComplexFloat), 8u);
}

TEST(ScalarTest, Conversions) {
    // Integer literals pick the int32 overload; only int64_t makes a Long.
    Scalar si(static_cast<int64_t>(42));
    EXPECT_EQ(si.to<int64_t>(), 42);
    EXPECT_EQ(si.dtype(), DType::Int64);

    Scalar si32(42);
    EXPECT_EQ(si32.dtype(), DType::Int32);

    Scalar sd(1.5);
    EXPECT_DOUBLE_EQ(sd.toDouble(), 1.5);
    EXPECT_EQ(sd.dtype(), DType::Float64);

    Scalar sb(true);
    EXPECT_TRUE(sb.to<bool>());
    EXPECT_EQ(sb.dtype(), DType::Bool);
}

TEST(ScalarTest, UndefinedThrows) {
    Scalar s;
    EXPECT_THROW(s.toDouble(), RuntimeError);
}
