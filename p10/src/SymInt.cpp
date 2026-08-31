#include "SymInt.h"

#include <limits>
#include <utility>

#include "SymFloat.h"

namespace tensorplay {

namespace {

int64_t checked_add(int64_t left, int64_t right) {
    const __int128 result = static_cast<__int128>(left) + right;
    TP_CHECK_VALUE(result >= std::numeric_limits<int64_t>::min() &&
                       result <= std::numeric_limits<int64_t>::max(),
                   "symbolic integer addition overflow");
    return static_cast<int64_t>(result);
}

int64_t checked_sub(int64_t left, int64_t right) {
    const __int128 result = static_cast<__int128>(left) - right;
    TP_CHECK_VALUE(result >= std::numeric_limits<int64_t>::min() &&
                       result <= std::numeric_limits<int64_t>::max(),
                   "symbolic integer subtraction overflow");
    return static_cast<int64_t>(result);
}

int64_t checked_mul(int64_t left, int64_t right) {
    const __int128 result = static_cast<__int128>(left) * right;
    TP_CHECK_VALUE(result >= std::numeric_limits<int64_t>::min() &&
                       result <= std::numeric_limits<int64_t>::max(),
                   "symbolic integer multiplication overflow");
    return static_cast<int64_t>(result);
}

int64_t checked_left_shift(int64_t left, int64_t right) {
    TP_CHECK_VALUE(right >= 0,
                   "symbolic integer shift count must be non-negative");
    TP_CHECK_VALUE(right < 127,
                   "symbolic integer shift count is too large");
    const __int128 multiplier = static_cast<__int128>(1) << right;
    const __int128 result = static_cast<__int128>(left) * multiplier;
    TP_CHECK_VALUE(
        result >= std::numeric_limits<int64_t>::min() &&
            result <= std::numeric_limits<int64_t>::max(),
        "symbolic integer left shift overflow");
    return static_cast<int64_t>(result);
}

int64_t checked_right_shift(int64_t left, int64_t right) {
    TP_CHECK_VALUE(right >= 0,
                   "symbolic integer shift count must be non-negative");
    if (right >= 63) return left < 0 ? -1 : 0;
    const int64_t divisor = static_cast<int64_t>(1ULL << right);
    int64_t quotient = left / divisor;
    const int64_t remainder = left % divisor;
    if (remainder != 0 && left < 0) --quotient;
    return quotient;
}

int64_t bitwise_and(int64_t left, int64_t right) {
    return static_cast<int64_t>(static_cast<uint64_t>(left) &
                                static_cast<uint64_t>(right));
}

int64_t bitwise_or(int64_t left, int64_t right) {
    return static_cast<int64_t>(static_cast<uint64_t>(left) |
                                static_cast<uint64_t>(right));
}

int64_t bitwise_xor(int64_t left, int64_t right) {
    return static_cast<int64_t>(static_cast<uint64_t>(left) ^
                                static_cast<uint64_t>(right));
}

int64_t checked_pow(int64_t base, int64_t exponent) {
    TP_CHECK_VALUE(exponent >= 0,
                   "symbolic integer exponent must be non-negative");
    int64_t result = 1;
    uint64_t remaining = static_cast<uint64_t>(exponent);
    while (remaining != 0) {
        if (remaining & 1U) result = checked_mul(result, base);
        remaining >>= 1U;
        if (remaining != 0) base = checked_mul(base, base);
    }
    return result;
}

} // namespace

SymInt::SymInt(int64_t value) : data_(value) {
    if (is_heap_allocated()) promote_to_negative();
}

SymInt::SymInt(SymNode node) {
    TP_CHECK_TYPE(node && node->is_int(),
                  "symbolic integer node has an incompatible type");
    const uint64_t pointer = reinterpret_cast<uintptr_t>(node.get());
    TP_CHECK_VALUE((pointer & kMask) == 0,
                   "symbolic node pointer cannot be represented");
    data_ = static_cast<int64_t>((pointer & ~kMask) | kIsSymbolic);
    static_cast<void>(std::move(node).release());
}

SymInt::SymInt(const SymInt& other) noexcept : data_(other.data_) {
    if (other.is_heap_allocated()) toSymNodeImplUnowned()->incref();
}

SymInt::SymInt(SymInt&& other) noexcept : data_(other.data_) {
    other.data_ = 0;
}

SymInt& SymInt::operator=(const SymInt& other) noexcept {
    if (this == &other) return *this;
    release_();
    data_ = other.data_;
    if (other.is_heap_allocated()) toSymNodeImplUnowned()->incref();
    return *this;
}

SymInt& SymInt::operator=(SymInt&& other) noexcept {
    if (this == &other) return *this;
    release_();
    data_ = other.data_;
    if (other.is_heap_allocated()) other.data_ = 0;
    return *this;
}

SymInt::~SymInt() { release_(); }

bool SymInt::is_heap_allocated() const noexcept {
    return !check_range(data_);
}

SymNodeImpl* SymInt::toSymNodeImplUnowned() const noexcept {
    const uint64_t unextended = static_cast<uint64_t>(data_) & ~kMask;
    const uint64_t sign = 1ULL << 61;
    const uint64_t extended = (unextended ^ sign) - sign;
    return reinterpret_cast<SymNodeImpl*>(static_cast<uintptr_t>(extended));
}

void SymInt::release_() noexcept {
    if (is_heap_allocated()) {
        toSymNodeImplUnowned()->decref();
    }
}

SymNodeImpl* SymInt::release() && {
    TP_CHECK(is_heap_allocated(), "symbolic integer is stored inline");
    auto* result = toSymNodeImplUnowned();
    data_ = 0;
    return result;
}

void SymInt::promote_to_negative() {
    SymInt promoted(make_constant_int(data_));
    data_ = promoted.data_;
    promoted.data_ = 0;
}

SymNode SymInt::toSymNode() const {
    TP_CHECK(is_heap_allocated(), "symbolic integer is stored inline");
    return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymNode SymInt::wrap_node(const SymNode& base) const {
    if (auto value = maybe_as_int()) return base->wrap_int(*value);
    return toSymNode();
}

std::optional<int64_t> SymInt::maybe_as_int_slow() const {
    if (auto value = toSymNodeImplUnowned()->constant_int()) return value;
    return toSymNodeImplUnowned()->maybe_as_int();
}

std::optional<int64_t> SymInt::maybe_as_int() const {
    if (!is_heap_allocated()) return data_;
    return maybe_as_int_slow();
}

bool SymInt::is_symbolic() const {
    return is_heap_allocated() &&
           !toSymNodeImplUnowned()->constant_int().has_value();
}

bool SymInt::has_hint() const {
    if (!is_heap_allocated()) return true;
    return toSymNodeImplUnowned()->has_hint();
}

int64_t SymInt::expect_int() const {
    if (auto value = maybe_as_int()) return *value;
    TP_THROW(RuntimeError, "expected a concrete integer value");
}

int64_t SymInt::guard_int(const char* file, int64_t line) const {
    if (auto value = maybe_as_int()) return *value;
    return toSymNodeImplUnowned()->guard_int(file, line);
}

SymInt SymInt::binary_slow(const SymInt& other, int operation) const {
    SymNode left;
    SymNode right;
    if (!maybe_as_int()) left = toSymNode();
    if (!other.maybe_as_int()) right = other.toSymNode();
    if (!left) left = right->wrap_int(*maybe_as_int());
    if (!right) right = left->wrap_int(*other.maybe_as_int());
    SymNode result;
    switch (operation) {
        case 0: result = left->add(right); break;
        case 1: result = left->sub(right); break;
        case 2: result = left->mul(right); break;
        case 3: result = left->floordiv(right); break;
        case 4: result = left->mod(right); break;
        case 5: result = left->sym_min(right); break;
        case 6: result = left->sym_max(right); break;
        case 7: result = left->bitwise_and(right); break;
        case 8: result = left->bitwise_or(right); break;
        case 9: result = left->bitwise_xor(right); break;
        case 10: result = left->lshift(right); break;
        case 11: result = left->rshift(right); break;
        default: TP_THROW(RuntimeError, "unknown symbolic integer operation");
    }
    return SymInt(std::move(result));
}

SymBool SymInt::compare_slow(const SymInt& other, int operation) const {
    SymNode left;
    SymNode right;
    if (!maybe_as_int()) left = toSymNode();
    if (!other.maybe_as_int()) right = other.toSymNode();
    if (!left) left = right->wrap_int(*maybe_as_int());
    if (!right) right = left->wrap_int(*other.maybe_as_int());
    SymNode result;
    switch (operation) {
        case 0: result = left->eq(right); break;
        case 1: result = left->ne(right); break;
        case 2: result = left->lt(right); break;
        case 3: result = left->le(right); break;
        case 4: result = left->gt(right); break;
        case 5: result = left->ge(right); break;
        default: TP_THROW(RuntimeError, "unknown symbolic integer comparison");
    }
    return SymBool(std::move(result));
}

SymInt SymInt::operator+(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymInt(checked_add(*left, *right));
    }
    return binary_slow(other, 0);
}

SymInt SymInt::operator-(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymInt(checked_sub(*left, *right));
    }
    return binary_slow(other, 1);
}

SymInt SymInt::operator*(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymInt(checked_mul(*left, *right));
    }
    return binary_slow(other, 2);
}

SymInt SymInt::operator/(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) {
            TP_CHECK_VALUE(*right != 0, "symbolic integer division by zero");
            TP_CHECK_VALUE(!(*left == std::numeric_limits<int64_t>::min() && *right == -1),
                           "symbolic integer division overflow");
            int64_t quotient = *left / *right;
            const int64_t remainder = *left % *right;
            if (remainder != 0 && ((remainder < 0) != (*right < 0))) --quotient;
            return SymInt(quotient);
        }
    }
    return binary_slow(other, 3);
}

SymInt SymInt::operator%(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) {
            TP_CHECK_VALUE(*right != 0, "symbolic integer remainder by zero");
            TP_CHECK_VALUE(!(*left == std::numeric_limits<int64_t>::min() &&
                             *right == -1),
                           "symbolic integer remainder overflow");
            return SymInt(*left % *right);
        }
    }
    return binary_slow(other, 4);
}

SymInt SymInt::operator&(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) {
            return SymInt(bitwise_and(*left, *right));
        }
    }
    return binary_slow(other, 7);
}

SymInt SymInt::operator|(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) {
            return SymInt(bitwise_or(*left, *right));
        }
    }
    return binary_slow(other, 8);
}

SymInt SymInt::operator^(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) {
            return SymInt(bitwise_xor(*left, *right));
        }
    }
    return binary_slow(other, 9);
}

SymInt SymInt::operator<<(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) {
            return SymInt(checked_left_shift(*left, *right));
        }
    }
    return binary_slow(other, 10);
}

SymInt SymInt::operator>>(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) {
            return SymInt(checked_right_shift(*left, *right));
        }
    }
    return binary_slow(other, 11);
}

void SymInt::operator+=(const SymInt& other) { *this = *this + other; }
void SymInt::operator-=(const SymInt& other) { *this = *this - other; }
void SymInt::operator*=(const SymInt& other) { *this = *this * other; }
void SymInt::operator/=(const SymInt& other) { *this = *this / other; }

SymInt SymInt::clone() const {
    if (auto value = maybe_as_int()) return SymInt(*value);
    return SymInt(toSymNodeImplUnowned()->clone());
}

SymBool SymInt::sym_eq(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymBool(*left == *right);
    }
    return compare_slow(other, 0);
}

SymBool SymInt::sym_ne(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymBool(*left != *right);
    }
    return compare_slow(other, 1);
}

SymBool SymInt::sym_lt(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymBool(*left < *right);
    }
    return compare_slow(other, 2);
}

SymBool SymInt::sym_le(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymBool(*left <= *right);
    }
    return compare_slow(other, 3);
}

SymBool SymInt::sym_gt(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymBool(*left > *right);
    }
    return compare_slow(other, 4);
}

SymBool SymInt::sym_ge(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymBool(*left >= *right);
    }
    return compare_slow(other, 5);
}

SymInt SymInt::min(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymInt(std::min(*left, *right));
    }
    return binary_slow(other, 5);
}

SymInt SymInt::max(const SymInt& other) const {
    if (auto left = maybe_as_int()) {
        if (auto right = other.maybe_as_int()) return SymInt(std::max(*left, *right));
    }
    return binary_slow(other, 6);
}

SymInt SymInt::pow_by_natural(const SymInt& exponent) const {
    if (auto value = exponent.maybe_as_int()) {
        TP_CHECK_VALUE(*value >= 0,
                       "symbolic integer exponent must be non-negative");
        if (auto base = maybe_as_int()) return SymInt(checked_pow(*base, *value));
    } else {
        TP_CHECK(exponent.sym_ge(SymInt(0)).expect_true(__FILE__, __LINE__),
                 "symbolic integer exponent must be non-negative");
    }

    SymNode left;
    SymNode right;
    if (!maybe_as_int()) left = toSymNode();
    if (!exponent.maybe_as_int()) right = exponent.toSymNode();
    SymNodeImpl* base = left ? left.get() : right.get();
    TP_CHECK(base != nullptr, "at least one symbolic integer is required");
    if (!left) left = base->wrap_int(*maybe_as_int());
    if (!right) right = base->wrap_int(*exponent.maybe_as_int());
    return SymInt(left->pow_by_natural(right));
}

SymInt SymInt::ceil() const {
    return clone();
}

SymInt SymInt::floor() const {
    return clone();
}

SymInt SymInt::trunc() const {
    return clone();
}

SymInt SymInt::round() const {
    return clone();
}

SymFloat SymInt::sym_float() const {
    return static_cast<SymFloat>(*this);
}

bool SymInt::is_same(const SymInt& other) const {
    if (is_heap_allocated() != other.is_heap_allocated()) return false;
    if (!is_heap_allocated()) return data_ == other.data_;
    return toSymNodeImplUnowned() == other.toSymNodeImplUnowned();
}

SymInt::operator SymFloat() const {
    if (auto value = maybe_as_int()) return SymFloat(static_cast<double>(*value));
    return SymFloat(toSymNodeImplUnowned()->sym_float());
}

SymInt operator-(const SymInt& value) {
    if (auto concrete = value.maybe_as_int()) {
        if (*concrete == std::numeric_limits<int64_t>::min()) return SymInt(*concrete);
        return SymInt(-*concrete);
    }
    return SymInt(value.toSymNodeImplUnowned()->neg());
}

std::ostream& operator<<(std::ostream& os, const SymInt& value) {
    if (value.is_heap_allocated()) {
        os << value.toSymNodeImplUnowned()->str();
    } else {
        os << value.as_int_unchecked();
    }
    return os;
}

#define TP_DEFINE_INT_OPS(scalar_type) \
    SymInt operator+(const SymInt& left, scalar_type right) { return left + SymInt(static_cast<int64_t>(right)); } \
    SymInt operator-(const SymInt& left, scalar_type right) { return left - SymInt(static_cast<int64_t>(right)); } \
    SymInt operator*(const SymInt& left, scalar_type right) { return left * SymInt(static_cast<int64_t>(right)); } \
    SymInt operator/(const SymInt& left, scalar_type right) { return left / SymInt(static_cast<int64_t>(right)); } \
    SymInt operator+(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) + right; } \
    SymInt operator-(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) - right; } \
    SymInt operator*(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) * right; } \
    SymInt operator/(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) / right; } \
    SymInt operator%(const SymInt& left, scalar_type right) { return left % SymInt(static_cast<int64_t>(right)); } \
    SymInt operator%(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) % right; } \
    bool operator==(const SymInt& left, scalar_type right) { return left == SymInt(static_cast<int64_t>(right)); } \
    bool operator!=(const SymInt& left, scalar_type right) { return left != SymInt(static_cast<int64_t>(right)); } \
    bool operator<(const SymInt& left, scalar_type right) { return left < SymInt(static_cast<int64_t>(right)); } \
    bool operator<=(const SymInt& left, scalar_type right) { return left <= SymInt(static_cast<int64_t>(right)); } \
    bool operator>(const SymInt& left, scalar_type right) { return left > SymInt(static_cast<int64_t>(right)); } \
    bool operator>=(const SymInt& left, scalar_type right) { return left >= SymInt(static_cast<int64_t>(right)); } \
    bool operator==(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) == right; } \
    bool operator!=(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) != right; } \
    bool operator<(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) < right; } \
    bool operator<=(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) <= right; } \
    bool operator>(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) > right; } \
    bool operator>=(scalar_type left, const SymInt& right) { return SymInt(static_cast<int64_t>(left)) >= right; }

TP_DEFINE_INT_OPS(int64_t)
TP_DEFINE_INT_OPS(int32_t)
TP_DEFINE_INT_OPS(uint64_t)
TP_DEFINE_INT_OPS(uint32_t)

#undef TP_DEFINE_INT_OPS

SymFloat operator+(const SymInt& left, double right) {
    return static_cast<SymFloat>(left) + SymFloat(right);
}
SymFloat operator-(const SymInt& left, double right) {
    return static_cast<SymFloat>(left) - SymFloat(right);
}
SymFloat operator*(const SymInt& left, double right) {
    return static_cast<SymFloat>(left) * SymFloat(right);
}
SymFloat operator/(const SymInt& left, double right) {
    return static_cast<SymFloat>(left) / SymFloat(right);
}
SymFloat operator+(double left, const SymInt& right) {
    return SymFloat(left) + static_cast<SymFloat>(right);
}
SymFloat operator-(double left, const SymInt& right) {
    return SymFloat(left) - static_cast<SymFloat>(right);
}
SymFloat operator*(double left, const SymInt& right) {
    return SymFloat(left) * static_cast<SymFloat>(right);
}
SymFloat operator/(double left, const SymInt& right) {
    return SymFloat(left) / static_cast<SymFloat>(right);
}

} // namespace tensorplay
