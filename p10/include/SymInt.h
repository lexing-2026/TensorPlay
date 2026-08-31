#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <type_traits>

#include "Macros.h"
#include "SymBool.h"
#include "SymNodeImpl.h"

namespace tensorplay {

class SymFloat;

class P10_API SymInt {
public:
    enum Unchecked {
        UNCHECKED,
    };

    SymInt() noexcept : data_(0) {}
    SymInt(int64_t value);
    SymInt(SymNode node);
    SymInt(Unchecked, int64_t value) noexcept : data_(value) {}

    SymInt(const SymInt& other) noexcept;
    SymInt(SymInt&& other) noexcept;
    SymInt& operator=(const SymInt& other) noexcept;
    SymInt& operator=(SymInt&& other) noexcept;
    ~SymInt();

    SymNodeImpl* toSymNodeImplUnowned() const noexcept;
    SymNode toSymNode() const;
    SymNode wrap_node(const SymNode& base) const;
    SymNodeImpl* release() &&;

    int64_t expect_int() const;
    int64_t guard_int(const char* file, int64_t line) const;
    bool has_hint() const;
    bool is_symbolic() const;
    bool is_heap_allocated() const noexcept;
    std::optional<int64_t> maybe_as_int() const;
    int64_t as_int_unchecked() const noexcept { return data_; }
    void unsafe_set_data(size_t value) noexcept {
        data_ = static_cast<int64_t>(value);
    }

    SymInt operator+(const SymInt& other) const;
    SymInt operator-(const SymInt& other) const;
    SymInt operator*(const SymInt& other) const;
    SymInt operator/(const SymInt& other) const;
    SymInt operator%(const SymInt& other) const;
    SymInt operator&(const SymInt& other) const;
    SymInt operator|(const SymInt& other) const;
    SymInt operator^(const SymInt& other) const;
    SymInt operator<<(const SymInt& other) const;
    SymInt operator>>(const SymInt& other) const;
    void operator+=(const SymInt& other);
    void operator-=(const SymInt& other);
    void operator*=(const SymInt& other);
    void operator/=(const SymInt& other);

    SymInt clone() const;
    SymBool sym_eq(const SymInt& other) const;
    SymBool sym_ne(const SymInt& other) const;
    SymBool sym_lt(const SymInt& other) const;
    SymBool sym_le(const SymInt& other) const;
    SymBool sym_gt(const SymInt& other) const;
    SymBool sym_ge(const SymInt& other) const;

    bool operator==(const SymInt& other) const {
        return sym_eq(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator!=(const SymInt& other) const {
        return sym_ne(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator<(const SymInt& other) const {
        return sym_lt(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator<=(const SymInt& other) const {
        return sym_le(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator>(const SymInt& other) const {
        return sym_gt(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator>=(const SymInt& other) const {
        return sym_ge(other).guard_bool(__FILE__, __LINE__);
    }

    SymInt min(const SymInt& other) const;
    SymInt max(const SymInt& other) const;
    SymInt pow_by_natural(const SymInt& exponent) const;
    SymInt ceil() const;
    SymInt floor() const;
    SymInt trunc() const;
    SymInt round() const;
    SymFloat sym_float() const;
    bool is_same(const SymInt& other) const;
    operator SymFloat() const;

    static bool check_range(int64_t value) noexcept {
        return value > kMaxUnrepresentableInt;
    }

    static constexpr int64_t min_representable_int() noexcept {
        return kMaxUnrepresentableInt + 1;
    }

private:
    void promote_to_negative();
    void release_() noexcept;
    SymInt binary_slow(const SymInt& other, int operation) const;
    SymBool compare_slow(const SymInt& other, int operation) const;
    std::optional<int64_t> maybe_as_int_slow() const;

    static constexpr uint64_t kMask =
        (1ULL << 63) | (1ULL << 62) | (1ULL << 61);
    static constexpr uint64_t kIsSymbolic = (1ULL << 63) | (1ULL << 61);
    static constexpr int64_t kMaxUnrepresentableInt =
        -1LL & static_cast<int64_t>(~(1ULL << 62));

    int64_t data_ = 0;
};

template <typename Container>
inline SymInt multiply_integers(const Container& values) {
    return std::accumulate(
        values.begin(), values.end(), SymInt(1),
        [](const SymInt& left, const SymInt& right) { return left * right; });
}

template <typename Iterator>
inline SymInt multiply_integers(Iterator begin, Iterator end) {
    return std::accumulate(
        begin, end, SymInt(1),
        [](const SymInt& left, const SymInt& right) { return left * right; });
}

#define TP_DECLARE_SYMINT_INT_OP(scalar_type) \
    P10_API SymInt operator+(const SymInt&, scalar_type); \
    P10_API SymInt operator-(const SymInt&, scalar_type); \
    P10_API SymInt operator*(const SymInt&, scalar_type); \
    P10_API SymInt operator/(const SymInt&, scalar_type); \
    P10_API SymInt operator+(scalar_type, const SymInt&); \
    P10_API SymInt operator-(scalar_type, const SymInt&); \
    P10_API SymInt operator*(scalar_type, const SymInt&); \
    P10_API SymInt operator/(scalar_type, const SymInt&); \
    P10_API SymInt operator%(const SymInt&, scalar_type); \
    P10_API SymInt operator%(scalar_type, const SymInt&); \
    P10_API bool operator==(const SymInt&, scalar_type); \
    P10_API bool operator!=(const SymInt&, scalar_type); \
    P10_API bool operator<(const SymInt&, scalar_type); \
    P10_API bool operator<=(const SymInt&, scalar_type); \
    P10_API bool operator>(const SymInt&, scalar_type); \
    P10_API bool operator>=(const SymInt&, scalar_type); \
    P10_API bool operator==(scalar_type, const SymInt&); \
    P10_API bool operator!=(scalar_type, const SymInt&); \
    P10_API bool operator<(scalar_type, const SymInt&); \
    P10_API bool operator<=(scalar_type, const SymInt&); \
    P10_API bool operator>(scalar_type, const SymInt&); \
    P10_API bool operator>=(scalar_type, const SymInt&)

TP_DECLARE_SYMINT_INT_OP(int64_t);
TP_DECLARE_SYMINT_INT_OP(int32_t);
TP_DECLARE_SYMINT_INT_OP(uint64_t);
TP_DECLARE_SYMINT_INT_OP(uint32_t);

#undef TP_DECLARE_SYMINT_INT_OP

P10_API SymFloat operator+(const SymInt&, double);
P10_API SymFloat operator-(const SymInt&, double);
P10_API SymFloat operator*(const SymInt&, double);
P10_API SymFloat operator/(const SymInt&, double);
P10_API SymFloat operator+(double, const SymInt&);
P10_API SymFloat operator-(double, const SymInt&);
P10_API SymFloat operator*(double, const SymInt&);
P10_API SymFloat operator/(double, const SymInt&);

P10_API SymInt operator-(const SymInt& value);
P10_API std::ostream& operator<<(std::ostream& os, const SymInt& value);

inline bool sym_eq(int64_t left, int64_t right) noexcept { return left == right; }
inline SymBool sym_eq(const SymInt& left, const SymInt& right) {
    return left.sym_eq(right);
}
inline bool sym_ne(int64_t left, int64_t right) noexcept { return left != right; }
inline SymBool sym_ne(const SymInt& left, const SymInt& right) {
    return left.sym_ne(right);
}
inline bool sym_lt(int64_t left, int64_t right) noexcept { return left < right; }
inline SymBool sym_lt(const SymInt& left, const SymInt& right) {
    return left.sym_lt(right);
}
inline bool sym_le(int64_t left, int64_t right) noexcept { return left <= right; }
inline SymBool sym_le(const SymInt& left, const SymInt& right) {
    return left.sym_le(right);
}
inline bool sym_gt(int64_t left, int64_t right) noexcept { return left > right; }
inline SymBool sym_gt(const SymInt& left, const SymInt& right) {
    return left.sym_gt(right);
}
inline bool sym_ge(int64_t left, int64_t right) noexcept { return left >= right; }
inline SymBool sym_ge(const SymInt& left, const SymInt& right) {
    return left.sym_ge(right);
}

} // namespace tensorplay

namespace std {
template <>
class numeric_limits<tensorplay::SymInt> {
public:
    static constexpr bool is_specialized = true;
    static constexpr int64_t max() noexcept {
        return numeric_limits<int64_t>::max();
    }
    static constexpr int64_t min() noexcept {
        return numeric_limits<int64_t>::min();
    }
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = true;
};
} // namespace std
