#include <gtest/gtest.h>

// Entry point provided by GTest::gtest_main; this translation unit keeps the
// test binary's dependencies anchored in one place.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
