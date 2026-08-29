#include "FileCheck.h"
#include "Exception.h"

#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>

namespace tensorplay {

// ---------------------------------------------------------------------------
// Small read-only text source: offsets, line lookup and highlighted printing.
// ---------------------------------------------------------------------------
namespace {

struct CheckSource {
    std::string text;
    std::vector<size_t> line_starts;

    explicit CheckSource(std::string t) : text(std::move(t)) {
        line_starts.push_back(0);
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '\n') {
                line_starts.push_back(i + 1);
            }
        }
    }

    size_t size() const { return text.size(); }
    char char_at(size_t i) const { return text[i]; }

    size_t num_lines() const { return line_starts.size(); }

    // Line containing the offset.
    size_t lineno_for_offset(size_t offset) const {
        auto it = std::upper_bound(line_starts.begin(), line_starts.end(), offset);
        return (it - line_starts.begin()) - 1;
    }

    size_t offset_for_line(size_t lineno) const { return line_starts[lineno]; }

    size_t line_end(size_t lineno) const {
        size_t end = (lineno + 1 < line_starts.size()) ? line_starts[lineno + 1] - 1
                                                       : text.size();
        return end;
    }

    // Renders the given [start, end) range with a `~` underline marker,
    // following the diagnostic style used by runtime error messages.
    std::string highlight(size_t start, size_t end) const {
        std::ostringstream ss;
        size_t line = lineno_for_offset(start);
        size_t ls = offset_for_line(line);
        size_t le = line_end(line);
        ss.write(text.data() + ls, static_cast<std::streamsize>(le - ls));
        ss << '\n';
        ss << std::string(start - ls, ' ');
        size_t span = std::min(end, le) > start ? std::min(end, le) - start : 1;
        ss << std::string(span, '~');
        ss << " <--- HERE";
        return ss.str();
    }
};

std::string quote(const std::string& s) {
    std::ostringstream ss;
    ss << '"';
    for (char c : s) {
        switch (c) {
            case '"': ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\n': ss << "\\n"; break;
            case '\t': ss << "\\t"; break;
            default: ss << c;
        }
    }
    ss << '"';
    return ss.str();
}

size_t text_find(const CheckSource& source, const std::string& sub, size_t start) {
    if (start > source.size()) return std::string::npos;
    return source.text.find(sub, start);
}

size_t regex_find(const CheckSource& source, const std::string& pattern, size_t start) {
    if (start > source.size()) return std::string::npos;
    try {
        std::regex re(pattern);
        std::smatch m;
        if (std::regex_search(source.text.cbegin() + static_cast<long>(start),
                              source.text.cend(), m, re)) {
            return start + static_cast<size_t>(m.position(0));
        }
    } catch (const std::regex_error&) {
        TP_THROW(RuntimeError, "invalid regex in check: ", quote(pattern));
    }
    return std::string::npos;
}

// A matched extent [begin, end) in the source.
struct Span {
    size_t begin;
    size_t end;
};

} // namespace

struct FileCheck::Impl {
    std::vector<FileCheckEntry> checks;
    std::vector<std::vector<FileCheckEntry>> groups;
    bool has_run = false;

    void add_check(FileCheckEntry entry) {
        // Consecutive CHECK_NOTs and consecutive CHECK_DAGs are evaluated as
        // one group; any other check starts its own group.
        if (groups.empty() ||
            (entry.type_ != CheckType::CheckNot && entry.type_ != CheckType::CheckDag)) {
            groups.push_back({entry});
        } else {
            auto& last_group = groups.back();
            if (last_group.at(0).type_ == entry.type_) {
                last_group.push_back(entry);
            } else {
                groups.push_back({entry});
            }
        }
        checks.push_back(entry);
        has_run = false;
    }

    [[noreturn]] void fail_find(const CheckSource& source, size_t search_start,
                                const std::string& sub, const FileCheckEntry& entry) const {
        std::ostringstream ss;
        ss << "Expected to find " << quote(sub) << " but did not find it\n";
        ss << "Searched string:\n";
        ss << source.highlight(search_start, search_start + sub.size()) << '\n';
        ss << "From " << filecheck_entry_to_string(entry) << '\n';
        TP_THROW(RuntimeError, ss.str());
    }

    [[noreturn]] void fail_not_find(const CheckSource& source, size_t pos, size_t len,
                                    const std::string& sub, const FileCheckEntry& entry) const {
        std::ostringstream ss;
        ss << "Expected to not find " << quote(sub) << " but found it\n";
        ss << source.highlight(pos, pos + len) << '\n';
        ss << "From " << filecheck_entry_to_string(entry) << '\n';
        TP_THROW(RuntimeError, ss.str());
    }

    size_t assert_find(const CheckSource& source, size_t start, const std::string& sub,
                       const FileCheckEntry& entry) const {
        size_t pos = text_find(source, sub, start);
        if (pos == std::string::npos || (pos + sub.size()) > source.size()) {
            fail_find(source, start, sub, entry);
        }
        return pos;
    }

    size_t assert_find_regex(const CheckSource& source, size_t start,
                             const std::string& pattern, const FileCheckEntry& entry) const {
        size_t pos = regex_find(source, pattern, start);
        if (pos == std::string::npos) {
            std::ostringstream ss;
            ss << "Expected to find regex " << quote(pattern) << " but did not find it\n";
            ss << "Searched string:\n";
            ss << source.highlight(start, start + pattern.size()) << '\n';
            ss << "From " << filecheck_entry_to_string(entry) << '\n';
            TP_THROW(RuntimeError, ss.str());
        }
        return pos;
    }

    void assert_not_find(const CheckSource& source, size_t start, size_t end,
                         const std::string& sub, const FileCheckEntry& entry) const {
        if (end < start) return;
        size_t pos = text_find(source, sub, start);
        if (pos != std::string::npos && (pos + sub.size()) <= end) {
            fail_not_find(source, pos, sub.size(), sub, entry);
        }
    }

    void do_check_not(const std::vector<FileCheckEntry>& nots, const CheckSource& source,
                      const Span& prev, const Span& next) const {
        for (const auto& check : nots) {
            assert_not_find(source, prev.end, next.begin, check.search_str_, check);
        }
    }

    // Verifies the pattern sits inside a highlighted region: the line below
    // the match must be filled with '~' across the whole match span, and the
    // '~' run must end exactly at the span edges. Does not advance the
    // search position.
    void do_check_source_highlighted(const FileCheckEntry& check, const CheckSource& source,
                                     size_t start_offset) const {
        const std::string& sub = check.search_str_;
        size_t search_from = start_offset;
        bool found_token_at_least_once = false;
        while (search_from <= source.size()) {
            size_t pos = text_find(source, sub, search_from);
            if (pos == std::string::npos) break;
            found_token_at_least_once = true;

            size_t lineno = source.lineno_for_offset(pos);
            size_t col = pos - source.offset_for_line(lineno);
            size_t highlight_lineno = lineno + 1;

            if (highlight_lineno >= source.num_lines()) {
                fail_highlight(source, pos, sub);
            }
            size_t hl_start = source.offset_for_line(highlight_lineno) + col;
            size_t hl_end = std::min(hl_start + sub.size(), source.size());
            if (hl_end >= source.size()) {
                fail_highlight(source, pos, sub);
            }
            bool found_highlight = true;
            for (size_t i = hl_start; i < hl_end; ++i) {
                if (source.char_at(i) != '~') found_highlight = false;
            }
            if (found_highlight) {
                assert_not_find(source, hl_start - 1, hl_start, "~", check);
                assert_not_find(source, hl_end, hl_end + 1, "~", check);
                return;
            }
            search_from = pos + 1;
        }
        if (!found_token_at_least_once) {
            fail_find(source, start_offset, sub, check);
        }
        fail_highlight(source, start_offset, sub);
    }

    [[noreturn]] void fail_highlight(const CheckSource& source, size_t pos,
                                     const std::string& sub) const {
        std::ostringstream ss;
        ss << "Expected to find " << quote(sub) << "highlighted but it is not.\n";
        ss << source.highlight(pos, pos + sub.size()) << '\n';
        TP_THROW(RuntimeError, ss.str());
    }

    Span match_dag_group(const std::vector<FileCheckEntry>& group, const CheckSource& source,
                         const Span& prev) const {
        size_t group_beg = std::string::npos;
        size_t group_end = 0;
        for (const auto& check : group) {
            size_t pos = assert_find(source, prev.end, check.search_str_, check);
            group_beg = std::min(pos, group_beg);
            group_end = std::max(pos + check.search_str_.size(), group_end);
        }
        return Span{group_beg, group_end};
    }

    Span match_group(const std::vector<FileCheckEntry>& group, const CheckSource& source,
                     const Span& prev) const {
        const FileCheckEntry& check = group.at(0);
        size_t start_range = prev.end;
        size_t end_range = start_range;

        switch (check.type_) {
            case CheckType::CheckDag:
                return match_dag_group(group, source, prev);
            case CheckType::Check: {
                start_range = assert_find(source, start_range, check.search_str_, check);
                end_range = start_range + check.search_str_.size();
                break;
            }
            case CheckType::CheckSame: {
                size_t pos = assert_find(source, start_range, check.search_str_, check);
                assert_not_find(source, prev.end, pos, "\n", check);
                start_range = pos;
                end_range = pos + check.search_str_.size();
                break;
            }
            case CheckType::CheckNext: {
                size_t line_end = assert_find(source, start_range, "\n", check);
                size_t pos = assert_find(source, line_end + 1, check.search_str_, check);
                assert_not_find(source, line_end + 1, pos, "\n", check);
                start_range = pos;
                end_range = pos + check.search_str_.size();
                break;
            }
            case CheckType::CheckCount: {
                size_t count = check.count_.value();
                size_t group_start_range = std::string::npos;
                for (size_t i = 0; i < count; ++i) {
                    start_range = assert_find(source, start_range, check.search_str_, check);
                    group_start_range = std::min(start_range, group_start_range);
                    end_range = start_range + check.search_str_.size();
                    start_range = end_range;
                }
                start_range = group_start_range;
                break;
            }
            case CheckType::CheckSourceHighlighted: {
                do_check_source_highlighted(check, source, start_range);
                break;
            }
            case CheckType::CheckRegex: {
                start_range = assert_find_regex(source, start_range, check.search_str_, check);
                end_range = start_range + check.search_str_.size();
                break;
            }
            default:
                TP_THROW(RuntimeError, "unexpected check kind in group");
        }
        return Span{start_range, end_range};
    }

    void do_checks(const CheckSource& source) const {
        Span prev{0, 0};
        for (size_t i = 0; i < groups.size(); i++) {
            const auto& curr_group = groups[i];
            CheckType type = curr_group.at(0).type_;
            if (type != CheckType::CheckNot) {
                prev = match_group(curr_group, source, prev);
            } else {
                if (i + 1 < groups.size()) {
                    const auto& next_group = groups[i + 1];
                    Span after_not = match_group(next_group, source, prev);
                    do_check_not(curr_group, source, prev, after_not);
                    prev = after_not;
                    ++i; // the group after the NOTs was consumed as its bound
                } else {
                    Span end_of_file{source.size() + 1, source.size() + 1};
                    do_check_not(curr_group, source, prev, end_of_file);
                }
            }
        }
    }

    void run(const std::string& test_file) {
        has_run = true;
        if (groups.empty() || groups[0].empty()) {
            TP_THROW(RuntimeError,
                     "No checks have been added to this instance of"
                     "Filecheck! Check for bad input.");
        }
        CheckSource source(test_file);
        do_checks(source);
    }
};


std::string filecheck_entry_to_string(const FileCheckEntry& c) {
    std::ostringstream out;
    switch (c.type_) {
        case CheckType::Check: out << "CHECK"; break;
        case CheckType::CheckNext: out << "CHECK-NEXT"; break;
        case CheckType::CheckSame: out << "CHECK-SAME"; break;
        case CheckType::CheckNot: out << "CHECK-NOT"; break;
        case CheckType::CheckDag: out << "CHECK-DAG"; break;
        case CheckType::CheckCount: out << "CHECK-COUNT-" << c.count_.value(); break;
        case CheckType::CheckSourceHighlighted: out << "CHECK-SOURCE-HIGHLIGHTED"; break;
        case CheckType::CheckRegex: out << "CHECK-REGEX"; break;
    }
    out << ": " << c.search_str_;
    return out.str();
}

FileCheck::FileCheck() : impl_(new Impl()) {}

FileCheck::~FileCheck() {
    if (!impl_->has_run) {
        std::cout << "You have not run this instance of FileCheck!\n";
        std::cout << "FileCheck checks:\n";
        for (const auto& c : impl_->checks) {
            std::cout << '\t' << filecheck_entry_to_string(c) << '\n';
        }
    }
    delete impl_;
}

FileCheck* FileCheck::check(const std::string& str) {
    impl_->add_check(FileCheckEntry(CheckType::Check, str));
    return this;
}

FileCheck* FileCheck::check_not(const std::string& str) {
    impl_->add_check(FileCheckEntry(CheckType::CheckNot, str));
    return this;
}

FileCheck* FileCheck::check_same(const std::string& str) {
    impl_->add_check(FileCheckEntry(CheckType::CheckSame, str));
    return this;
}

FileCheck* FileCheck::check_next(const std::string& str) {
    impl_->add_check(FileCheckEntry(CheckType::CheckNext, str));
    return this;
}

FileCheck* FileCheck::check_count(const std::string& str, size_t count, bool exactly) {
    TP_CHECK(count != 0 || exactly, "Count == 0 && !exactly doesn't do anything");
    if (count) {
        impl_->add_check(FileCheckEntry(CheckType::CheckCount, str, count));
    }
    if (exactly) {
        impl_->add_check(FileCheckEntry(CheckType::CheckNot, str));
    }
    return this;
}

FileCheck* FileCheck::check_dag(const std::string& str) {
    impl_->add_check(FileCheckEntry(CheckType::CheckDag, str));
    return this;
}

FileCheck* FileCheck::check_source_highlighted(const std::string& str) {
    impl_->add_check(FileCheckEntry(CheckType::CheckSourceHighlighted, str));
    return this;
}

FileCheck* FileCheck::check_regex(const std::string& str) {
    impl_->add_check(FileCheckEntry(CheckType::CheckRegex, str));
    return this;
}

void FileCheck::run(const std::string& test_file) {
    impl_->run(test_file);
}

} // namespace tensorplay
