#include "SymFloat.h"

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
    if (!is_symbolic() && !other.is_symbolic()) return SymFloat(data_ / other.data_);
    auto values = normalize(*this, other);
    return SymFloat(values.left->truediv(values.right));
}

SymFloat SymFloat::operator-() const {
    if (!is_symbolic()) return SymFloat(-data_);
    return SymFloat(toSymNodeImplUnowned()->neg());
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
    if (!is_symbolic()) return SymFloat(std::sqrt(data_));
    auto values = normalize(*this, SymFloat(0.5));
    return SymFloat(values.left->pow(values.right));
}

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
