// Native FileCheck: directive semantics and diagnostics.

#include <gtest/gtest.h>

#include "FileCheck.h"

#include <string>

#include "Exception.h"

using tensorplay::FileCheck;

namespace {

// Runs the check and, on failure, returns the thrown message; returns an
// empty string when the run succeeds.
std::string run_and_capture(std::function<void()> fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST(FileCheckTest, BasicMatchAndChaining) {
    FileCheck().check("foo")->run("foo bar");
    FileCheck().check("a")->check("b")->run("a\nb");
    // Sequential checks continue after the previous match.
    FileCheck().check("a")->check("a")->run("a b a");
}

TEST(FileCheckTest, BasicFailureMessage) {
    std::string msg = run_and_capture([] {
        FileCheck().check("foo")->run("bar baz");
    });
    EXPECT_NE(msg.find("Expected to find \"foo\" but did not find it"), std::string::npos);
    EXPECT_NE(msg.find("Searched string:"), std::string::npos);
    EXPECT_NE(msg.find("From CHECK: foo"), std::string::npos);
}

TEST(FileCheckTest, SequentialOrderEnforced) {
    // Each check resumes after the previous match, so reversed order fails.
    std::string msg = run_and_capture([] {
        FileCheck().check("b")->check("a")->run("a\nb");
    });
    EXPECT_NE(msg.find("Expected to find \"a\""), std::string::npos);
}

TEST(FileCheckTest, CheckNext) {
    FileCheck().check("a")->check_next("b")->run("a\nb");
    std::string msg = run_and_capture([] {
        FileCheck().check("a")->check_next("b")->run("a\nx\nb");
    });
    EXPECT_NE(msg.find("Expected to not find \"\\n\""), std::string::npos);
    EXPECT_NE(msg.find("From CHECK-NEXT: b"), std::string::npos);
}

TEST(FileCheckTest, CheckSame) {
    FileCheck().check("a")->check_same("b")->run("a b");
    std::string msg = run_and_capture([] {
        FileCheck().check("a")->check_same("b")->run("a\nb");
    });
    EXPECT_NE(msg.find("Expected to not find \"\\n\""), std::string::npos);
}

TEST(FileCheckTest, CheckNot) {
    FileCheck().check("a")->check_not("b")->run("a c");
    // NOT covers only the region up to the next match.
    FileCheck().check("a")->check("c")->check_not("b")->run("a\nb\nc");
    std::string msg = run_and_capture([] {
        FileCheck().check("a")->check_not("b")->run("a b");
    });
    EXPECT_NE(msg.find("Expected to not find \"b\" but found it"), std::string::npos);
}

TEST(FileCheckTest, CheckCount) {
    FileCheck().check_count("x", 2, /*exactly=*/true)->run("x\ny\nx");
    FileCheck().check_count("x", 2, /*exactly=*/false)->run("x\ny\nx\nx");
    std::string msg = run_and_capture([] {
        FileCheck().check_count("x", 3, /*exactly=*/true)->run("x\ny\nx");
    });
    EXPECT_NE(msg.find("Expected to find \"x\""), std::string::npos);
}

TEST(FileCheckTest, CheckDag) {
    // DAG groups match in any order.
    FileCheck().check_dag("b")->check_dag("a")->run("b\na");
    std::string msg = run_and_capture([] {
        FileCheck().check_dag("b")->check_dag("a")->run("a\nc");
    });
    EXPECT_NE(msg.find("Expected to find \"b\""), std::string::npos);
}

TEST(FileCheckTest, CheckLabelResetsScope) {
    FileCheck().check("L1")->check("x")->check("L2")->check("x")->run("L1\nx\nL2\nx");
}

TEST(FileCheckTest, CheckSourceHighlighted) {
    // The '~' run on the following line must cover exactly the match span.
    FileCheck().check_source_highlighted("foo")->run("some foo here\n     ~~~\nnext");
    std::string longer = run_and_capture([] {
        FileCheck().check_source_highlighted("foo")->run("some foo here\n    ~~~~~~\nnext");
    });
    EXPECT_NE(longer.find("Expected to not find \"~\" but found it"), std::string::npos);
    std::string missing = run_and_capture([] {
        FileCheck().check_source_highlighted("foo")->run("some foo here\nnext line");
    });
    EXPECT_NE(missing.find("highlighted but it is not"), std::string::npos);
}

TEST(FileCheckTest, CheckRegex) {
    FileCheck().check_regex("[0-9]+ items")->run("there are 42 items");
    std::string msg = run_and_capture([] {
        FileCheck().check_regex("[0-9]+ items")->run("no numbers");
    });
    EXPECT_NE(msg.find("Expected to find regex \"[0-9]+ items\""), std::string::npos);
}

TEST(FileCheckTest, NoChecksIsAnError) {
    std::string msg = run_and_capture([] {
        FileCheck().run("anything");
    });
    EXPECT_NE(msg.find("No checks have been added"), std::string::npos);
}

TEST(FileCheckTest, AppliedToNativeErrorText) {
    // Typical usage: assert on the shape of native runtime error messages.
    using namespace tensorplay;
    try {
        TP_THROW(RuntimeError, "unsupported operation: more than one element of the "
                               "written-to tensor refers to a single memory location");
    } catch (const RuntimeError& e) {
        FileCheck().check("more than one element")->check("single memory location")->run(e.msg());
    }
}
