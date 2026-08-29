#pragma once

// Native FileCheck-style input checker for test suites. Supports the
// directive set and grouping semantics used by the test harness:
// literal pattern matching with ordered, same-line, next-line, negative,
// counted, order-independent (DAG), highlighted and regex checks.

#include <optional>
#include <string>
#include <vector>

#include "Macros.h"

namespace tensorplay {

enum class CheckType {
    Check,
    CheckNext,
    CheckSame,
    CheckNot,
    CheckCount,
    CheckDag,
    CheckSourceHighlighted,
    CheckRegex,
};

struct P10_API FileCheckEntry {
    FileCheckEntry(CheckType type, std::string str, std::optional<size_t> count = std::nullopt)
        : type_(type), count_(count), search_str_(std::move(str)) {}

    CheckType type_;
    std::optional<size_t> count_;
    const std::string search_str_;
};

P10_API std::string filecheck_entry_to_string(const FileCheckEntry& entry);

class P10_API FileCheck {
public:
    FileCheck();
    ~FileCheck();

    FileCheck(const FileCheck&) = delete;
    FileCheck& operator=(const FileCheck&) = delete;

    // All check* methods return *this so calls can be chained.
    FileCheck* check(const std::string& str);
    FileCheck* check_not(const std::string& str);
    FileCheck* check_same(const std::string& str);
    FileCheck* check_next(const std::string& str);
    FileCheck* check_count(const std::string& str, size_t count, bool exactly = false);
    FileCheck* check_dag(const std::string& str);
    FileCheck* check_source_highlighted(const std::string& str);
    FileCheck* check_regex(const std::string& str);

    // Applies the accumulated checks against the input text. Throws
    // std::runtime_error with a diagnostic on the first violation.
    void run(const std::string& test_file);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace tensorplay
