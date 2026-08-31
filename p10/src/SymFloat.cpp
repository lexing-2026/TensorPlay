#include "SymFloat.h"

#include "SymInt.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace tensorplay {

SymFloat::SymFloat(SymNode node)
    : data_(std::numeric_limits<double>::quiet_NaN()), ptr_(std::move(node)) {
    TP_CHECK_TYPE(ptr_ && ptr_->is_float(),
                  "symbolic floating node has an incompatible type");
}

SymNode SymFloat::toSymNodeImpl() const {
    TP_CHECK(is_symbolic(), "floating value is not symbolic");
    return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymNode SymFloat::wrap_node(const SymNode& base) const {
    if (is_symbolic()) return toSymNodeImpl();
    return base->wrap_float(as_float_unchecked());
}

std::optional<double> SymFloat::maybe_as_float() const {
    if (!is_symbolic()) return data_;
    return toSymNodeImplUnowned()->constant_float();
}

double SymFloat::expect_float() const {
    if (auto value = maybe_as_float()) return *value;
    TP_THROW(RuntimeError, "expected a concrete floating value");
}

namespace {

struct NormalizedFloat {
    SymNode left;
    SymNode right;
};

NormalizedFloat normalize(const SymFloat& left, const SymFloat& right) {
    SymNode a;
    SymNode b;
    if (left.is_symbolic()) a = left.toSymNodeImpl();
    if (right.is_symbolic()) b = right.toSymNodeImpl();
    SymNodeImpl* base = a ? a.get() : b.get();
    TP_CHECK(base != nullptr, "at least one symbolic floating value is required");
    if (!a) a = base->wrap_float(left.as_float_unchecked());
    if (!b) b = base->wrap_float(right.as_float_unchecked());
    return {std::move(a), std::move(b)};
}

} // namespace

SymFloat SymFloat::operator+(const SymFloat& other) const {
    if (!is_symbolic() && !other.is_symbolic()) return SymFloat(data_ + other.data_);
    auto values = normalize(*this, other);
    return SymFloat(values.left->add(values.right));
}

SymFloat SymFloat::operator-(const SymFloat& other) const {
    if (!is_symbolic() && !other.is_symbolic()) return SymFloat(data_ - other.data_);
    auto values = normalize(*this, other);
    return SymFloat(values.left->sub(values.right));
}

SymFloat SymFloat::operator*(const SymFloat& other) const {
    if (!is_symbolic() && !other.is_symbolic()) return SymFloat(data_ * other.data_);
    auto values = normalize(*this, other);
    return SymFloat(values.left->mul(values.right));
}

SymFloat SymFloat::operator/(const SymFloat& other) const {
    if (!is_symbolic() && !other.is_symbolic()) {
        TP_CHECK_VALUE(other.data_ != 0.0,
                       "symbolic floating division by zero");
        return SymFloat(data_ / other.data_);
    }
    auto values = normalize(*this, other);
    return SymFloat(values.left->truediv(values.right));
}

SymFloat SymFloat::operator%(const SymFloat& other) const {
    if (!is_symbolic() && !other.is_symbolic()) {
        TP_CHECK_VALUE(other.data_ != 0.0,
                       "symbolic floating remainder by zero");
        double result = std::fmod(data_, other.data_);
        if (result != 0.0 && ((result < 0.0) != (other.data_ < 0.0))) {
            result += other.data_;
        }
        return SymFloat(result);
    }
    auto values = normalize(*this, other);
    return SymFloat(values.left->mod(values.right));
}

SymFloat SymFloat::operator-() const {
    if (!is_symbolic()) return SymFloat(-data_);
    return SymFloat(toSymNodeImplUnowned()->neg());
}

SymFloat SymFloat::operator+() const {
    if (!is_symbolic()) return SymFloat(data_);
    return SymFloat(toSymNodeImplUnowned()->pos());
}

SymFloat SymFloat::abs() const {
    if (!is_symbolic()) return SymFloat(std::fabs(data_));
    return SymFloat(toSymNodeImplUnowned()->abs());
}

SymFloat SymFloat::pow(const SymFloat& exponent) const {
    if (!is_symbolic() && !exponent.is_symbolic()) {
        TP_CHECK_VALUE(data_ >= 0.0,
                       "symbolic floating power requires a non-negative base");
        return SymFloat(std::pow(data_, exponent.data_));
    }
    auto condition = sym_ge(SymFloat(0));
    TP_CHECK(condition.expect_true(__FILE__, __LINE__),
             "symbolic floating power requires a non-negative base");
    auto values = normalize(*this, exponent);
    return SymFloat(values.left->float_pow(values.right));
}

SymFloat SymFloat::floor_div(const SymFloat& other) const {
    if (!is_symbolic() && !other.is_symbolic()) {
        TP_CHECK_VALUE(other.data_ != 0.0,
                       "symbolic floating division by zero");
        return SymFloat(std::floor(data_ / other.data_));
    }
    auto values = normalize(*this, other);
    return SymFloat(values.left->floordiv(values.right));
}

namespace {

SymNode float_node(const SymFloat& value) {
    if (value.is_symbolic()) return value.toSymNodeImpl();
    return make_constant_float(value.as_float_unchecked());
}

} // namespace

SymInt SymFloat::ceil() const {
    return SymInt(float_node(*this)->ceil());
}

SymInt SymFloat::floor() const {
    return SymInt(float_node(*this)->floor());
}

SymInt SymFloat::trunc() const {
    return SymInt(float_node(*this)->trunc());
}

SymInt SymFloat::round() const {
    return SymInt(float_node(*this)->round());
}

SymFloat SymFloat::round(const SymInt& ndigits) const {
    SymNode left = float_node(*this);
    SymNode right;
    if (auto value = ndigits.maybe_as_int()) {
        right = left->wrap_int(*value);
    } else {
        right = ndigits.toSymNode();
    }
    return SymFloat(left->round(right));
}

SymBool SymFloat::is_integer() const {
    if (!is_symbolic()) {
        return SymBool(std::isfinite(data_) && std::trunc(data_) == data_);
    }
    return SymBool(toSymNodeImplUnowned()->is_integer());
}

SymInt SymFloat::sym_int() const {
    return SymInt(float_node(*this)->sym_int());
}

#define TP_FLOAT_COMPARE(name, method, op) \
    SymBool SymFloat::name(const SymFloat& other) const { \
        if (!is_symbolic() && !other.is_symbolic()) return SymBool(data_ op other.data_); \
        auto values = normalize(*this, other); \
        return SymBool(values.left->method(values.right)); \
    }

TP_FLOAT_COMPARE(sym_eq, eq, ==)
TP_FLOAT_COMPARE(sym_ne, ne, !=)
TP_FLOAT_COMPARE(sym_lt, lt, <)
TP_FLOAT_COMPARE(sym_le, le, <=)
TP_FLOAT_COMPARE(sym_gt, gt, >)
TP_FLOAT_COMPARE(sym_ge, ge, >=)

#undef TP_FLOAT_COMPARE

SymFloat SymFloat::min(const SymFloat& other) const {
    if (!is_symbolic() && !other.is_symbolic()) return SymFloat(std::min(data_, other.data_));
    auto values = normalize(*this, other);
    return SymFloat(values.left->sym_min(values.right));
}

SymFloat SymFloat::max(const SymFloat& other) const {
    if (!is_symbolic() && !other.is_symbolic()) return SymFloat(std::max(data_, other.data_));
    auto values = normalize(*this, other);
    return SymFloat(values.left->sym_max(values.right));
}

SymFloat SymFloat::sqrt() const {
    if (!is_symbolic()) {
        TP_CHECK_VALUE(data_ >= 0.0, "sqrt domain error");
        return SymFloat(std::sqrt(data_));
    }
    return SymFloat(toSymNodeImplUnowned()->sqrt());
}

#define TP_FLOAT_MATH(name) \
    SymFloat SymFloat::name() const { \
        if (!is_symbolic()) return SymFloat(std::name(data_)); \
        return SymFloat(toSymNodeImplUnowned()->name()); \
    }

TP_FLOAT_MATH(cos)
TP_FLOAT_MATH(cosh)
TP_FLOAT_MATH(sin)
TP_FLOAT_MATH(sinh)
TP_FLOAT_MATH(tan)
TP_FLOAT_MATH(tanh)
TP_FLOAT_MATH(asin)
TP_FLOAT_MATH(acos)
TP_FLOAT_MATH(atan)
TP_FLOAT_MATH(log2)

#undef TP_FLOAT_MATH

double SymFloat::guard_float(const char* file, int64_t line) const {
    if (!is_symbolic()) return data_;
    return toSymNodeImplUnowned()->guard_float(file, line);
}

bool SymFloat::has_hint() const {
    if (!is_symbolic()) return true;
    return toSymNodeImplUnowned()->has_hint();
}

std::ostream& operator<<(std::ostream& os, const SymFloat& value) {
    if (value.is_symbolic()) {
        os << value.toSymNodeImplUnowned()->str();
    } else {
        os << value.as_float_unchecked();
    }
    return os;
}

} // namespace tensorplay
