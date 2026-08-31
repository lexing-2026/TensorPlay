// Generator determinism and default-generator seeding.

#include <gtest/gtest.h>

#include "Exception.h"
#include "FileCheck.h"
#include "Generator.h"
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>

using namespace tensorplay;
namespace ops = tensorplay::tpx::ops;

namespace {
std::vector<float> draw(int n) {
    Tensor t = ops::empty({static_cast<int64_t>(n)}, DType::Float32, Device(DeviceType::CPU));
    ops::uniform_(t, 0.0, 1.0);
    const float* data = t.data_ptr<float>();
    return std::vector<float>(data, data + n);
}
} // namespace

TEST(GeneratorTest, ManualSeedReproducesSequence) {
    manual_seed(1234567890123456789ULL);
    std::vector<float> a = draw(128);
    manual_seed(1234567890123456789ULL);
    std::vector<float> b = draw(128);
    EXPECT_EQ(a, b);
}

TEST(GeneratorTest, DifferentSeedsDiverge) {
    manual_seed(1);
    std::vector<float> a = draw(64);
    manual_seed(2);
    std::vector<float> b = draw(64);
    EXPECT_NE(a, b);
}

TEST(GeneratorTest, InitialSeedRoundTrip) {
    manual_seed(0xDEADBEEFULL);
    EXPECT_EQ(default_generator().initial_seed(), 0xDEADBEEFULL);
}
