#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <utility>

#include "Macros.h"
#include "SymBool.h"

namespace tensorplay {

class SymInt;

class P10_API SymFloat {
public:
    SymFloat() noexcept : data_(0.0) {}
    SymFloat(double value) noexcept : data_(value) {}
    explicit SymFloat(SymNode node);

    SymFloat(const SymFloat&) = default;
    SymFloat(SymFloat&&) noexcept = default;
    SymFloat& operator=(const SymFloat&) = default;
    SymFloat& operator=(SymFloat&&) noexcept = default;
    ~SymFloat() = default;

    SymNodeImpl* toSymNodeImplUnowned() const noexcept { return ptr_.get(); }
    SymNode toSymNodeImpl() const;
    SymNode toSymNode() const { return toSymNodeImpl(); }
    SymNode wrap_node(const SymNode& base) const;

    double expect_float() const;
    std::optional<double> maybe_as_float() const;
    double as_float_unchecked() const noexcept { return data_; }
    bool is_symbolic() const noexcept { return static_cast<bool>(ptr_); }
    bool has_hint() const;

    SymFloat operator+(const SymFloat& other) const;
    SymFloat operator-(const SymFloat& other) const;
    SymFloat operator*(const SymFloat& other) const;
    SymFloat operator/(const SymFloat& other) const;
    SymFloat operator%(const SymFloat& other) const;
    SymFloat operator-() const;
    SymFloat operator+() const;
    SymFloat abs() const;
    SymFloat pow(const SymFloat& exponent) const;
    SymFloat floor_div(const SymFloat& other) const;
    SymInt ceil() const;
    SymInt floor() const;
    SymInt trunc() const;
    SymInt round() const;
    SymFloat round(const SymInt& ndigits) const;
    SymBool is_integer() const;
    SymInt sym_int() const;

    SymBool sym_eq(const SymFloat& other) const;
    SymBool sym_ne(const SymFloat& other) const;
    SymBool sym_lt(const SymFloat& other) const;
    SymBool sym_le(const SymFloat& other) const;
    SymBool sym_gt(const SymFloat& other) const;
    SymBool sym_ge(const SymFloat& other) const;

    bool operator==(const SymFloat& other) const {
        return sym_eq(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator!=(const SymFloat& other) const {
        return sym_ne(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator<(const SymFloat& other) const {
        return sym_lt(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator<=(const SymFloat& other) const {
        return sym_le(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator>(const SymFloat& other) const {
        return sym_gt(other).guard_bool(__FILE__, __LINE__);
    }
    bool operator>=(const SymFloat& other) const {
        return sym_ge(other).guard_bool(__FILE__, __LINE__);
    }

    SymFloat min(const SymFloat& other) const;
    SymFloat max(const SymFloat& other) const;
    SymFloat sqrt() const;
    SymFloat cos() const;
    SymFloat cosh() const;
    SymFloat sin() const;
    SymFloat sinh() const;
    SymFloat tan() const;
    SymFloat tanh() const;
    SymFloat asin() const;
    SymFloat acos() const;
    SymFloat atan() const;
    SymFloat log2() const;
    double guard_float(const char* file, int64_t line) const;

private:
    double data_ = 0.0;
    SymNode ptr_;
};

P10_API std::ostream& operator<<(std::ostream& os, const SymFloat& value);

} // namespace tensorplay
