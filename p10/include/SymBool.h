#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <utility>

#include "Macros.h"
#include "SymNodeImpl.h"

namespace tensorplay {

class SymInt;
class SymFloat;

class P10_API SymBool {
public:
    SymBool() noexcept : data_(false) {}
    SymBool(bool value) noexcept : data_(value) {}
    explicit SymBool(SymNode node);

    SymBool(const SymBool&) = default;
    SymBool(SymBool&&) noexcept = default;
    SymBool& operator=(const SymBool&) = default;
    SymBool& operator=(SymBool&&) noexcept = default;
    ~SymBool() = default;

    SymNodeImpl* toSymNodeImplUnowned() const noexcept { return ptr_.get(); }
    SymNode toSymNodeImpl() const;
    SymNode toSymNode() const { return toSymNodeImpl(); }
    SymNode wrap_node(const SymNode& base) const;

    bool expect_bool() const;
    std::optional<bool> maybe_as_bool() const;
    bool as_bool_unchecked() const noexcept { return data_; }
    bool is_heap_allocated() const noexcept { return static_cast<bool>(ptr_); }
    bool is_symbolic() const;

    SymBool sym_and(const SymBool& other) const;
    SymBool sym_or(const SymBool& other) const;
    SymBool sym_xor(const SymBool& other) const;
    SymBool sym_not() const;
    SymBool sym_eq(const SymBool& other) const;
    SymBool sym_ne(const SymBool& other) const;

    SymBool operator&(const SymBool& other) const { return sym_and(other); }
    SymBool operator|(const SymBool& other) const { return sym_or(other); }
    SymBool operator||(const SymBool& other) const { return sym_or(other); }
    SymBool operator~() const { return sym_not(); }

    bool equals(const SymBool& other) const;
    bool operator==(const SymBool& other) const { return equals(other); }
    bool operator!=(const SymBool& other) const { return !equals(other); }

    bool guard_bool(const char* file, int64_t line) const;
    bool expect_true(const char* file, int64_t line) const;
    bool guard_size_oblivious(const char* file, int64_t line) const;
    bool statically_known_true(const char* file, int64_t line) const;
    bool guard_or_false(const char* file, int64_t line) const;
    bool guard_or_true(const char* file, int64_t line) const;
    bool has_hint() const;

    SymInt toSymInt() const;
    SymFloat toSymFloat() const;

private:
    bool data_ = false;
    SymNode ptr_;
};

P10_API std::ostream& operator<<(std::ostream& os, const SymBool& value);

inline bool guard_size_oblivious(
    bool value, const char*, int64_t) noexcept {
    return value;
}

inline bool guard_size_oblivious(
    const SymBool& value, const char* file, int64_t line) {
    return value.guard_size_oblivious(file, line);
}

inline bool guard_or_false(bool value, const char*, int64_t) noexcept {
    return value;
}

inline bool guard_or_false(
    const SymBool& value, const char* file, int64_t line) {
    return value.guard_or_false(file, line);
}

inline bool statically_known_true(bool value, const char*, int64_t) noexcept {
    return value;
}

inline bool statically_known_true(
    const SymBool& value, const char* file, int64_t line) {
    return value.statically_known_true(file, line);
}

inline bool guard_or_true(bool value, const char*, int64_t) noexcept {
    return value;
}

inline bool guard_or_true(
    const SymBool& value, const char* file, int64_t line) {
    return value.guard_or_true(file, line);
}

#define TP_SYM_CHECK(cond, ...) \
    TP_CHECK((cond).expect_true(__FILE__, __LINE__), __VA_ARGS__)

#define TP_SYM_INTERNAL_ASSERT(cond, ...) \
    TP_CHECK((cond).expect_true(__FILE__, __LINE__), __VA_ARGS__)

#define TP_GUARD_SIZE_OBLIVIOUS(cond) \
    ::tensorplay::guard_size_oblivious((cond), __FILE__, __LINE__)

#define TP_STATICALLY_KNOWN_TRUE(cond) \
    ::tensorplay::statically_known_true((cond), __FILE__, __LINE__)

#define TP_GUARD_OR_FALSE(cond) \
    ::tensorplay::guard_or_false((cond), __FILE__, __LINE__)

#define TP_GUARD_OR_TRUE(cond) \
    ::tensorplay::guard_or_true((cond), __FILE__, __LINE__)

} // namespace tensorplay
