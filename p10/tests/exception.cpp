// Native exception types: hierarchy, message capture and diagnostics.

#include <gtest/gtest.h>

#include "Exception.h"
#include "FileCheck.h"

#include <string>

using namespace tensorplay;

namespace {
std::string capture(std::function<void()> fn) {
    try {
        fn();
    } catch (const Exception& e) {
        return e.msg();
    }
    return "";
}
} // namespace

TEST(ExceptionTest, RuntimeErrorCarriesMessage) {
    std::string msg = capture([] {
        TP_THROW(RuntimeError, "something went wrong: ", 42);
    });
    EXPECT_NE(msg.find("something went wrong: 42"), std::string::npos);
}

TEST(ExceptionTest, NotImplementedErrorIsAnException) {
    // NotImplementedError inherits from Exception directly (Python maps it
    // to the builtin NotImplementedError, not RuntimeError).
    bool caught = false;
    try {
        TP_THROW(NotImplementedError, "not available yet");
    } catch (const NotImplementedError& e) {
        caught = true;
        EXPECT_NE(e.msg().find("not available yet"), std::string::npos);
    }
    EXPECT_TRUE(caught);
}

TEST(ExceptionTest, DeviceMismatchIsRuntimeError) {
    try {
        TP_THROW(DeviceMismatchError, "cross-device");
    } catch (const RuntimeError&) {
        SUCCEED();
    }
}

TEST(ExceptionTest, ThrowIfOnlyFiresOnCondition) {
    std::string msg = capture([] {
        int v = 7;
        TP_THROW_IF(v > 5, ValueError, "value too large: ", v);
    });
    EXPECT_NE(msg.find("value too large: 7"), std::string::npos);

    std::string empty = capture([] {
        TP_THROW_IF(false, ValueError, "never");
    });
    EXPECT_TRUE(empty.empty());
}

TEST(ExceptionTest, ValueErrorType) {
    std::string msg = capture([] {
        TP_THROW(ValueError, "bad value");
    });
    FileCheck().check("bad value")->run(msg);
}
