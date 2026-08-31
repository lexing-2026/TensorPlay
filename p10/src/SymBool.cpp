#include "SymBool.h"

#include "SymFloat.h"
#include "SymInt.h"

#include <utility>

namespace tensorplay {

SymBool::SymBool(SymNode node) : data_(false), ptr_(std::move(node)) {
    TP_CHECK_TYPE(ptr_ && ptr_->is_bool(),
                  "symbolic boolean node has an incompatible type");
}

SymNode SymBool::toSymNodeImpl() const {
    TP_CHECK(is_heap_allocated(), "boolean value is not symbolic");
    return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymNode SymBool::wrap_node(const SymNode& base) const {
    if (auto value = maybe_as_bool()) {
        return base->wrap_bool(*value);
    }
    return toSymNodeImpl();
}

std::optional<bool> SymBool::maybe_as_bool() const {
    if (!is_heap_allocated()) return data_;
    return toSymNodeImplUnowned()->constant_bool();
}

bool SymBool::is_symbolic() const {
    return is_heap_allocated() &&
           !toSymNodeImplUnowned()->constant_bool().has_value();
}

namespace {

struct NormalizedBool {
    SymNode left;
    SymNode right;
};

NormalizedBool normalize(const SymBool& left, const SymBool& right) {
    SymNode a;
    SymNode b;
    if (left.is_symbolic()) a = left.toSymNodeImpl();
    if (right.is_symbolic()) b = right.toSymNodeImpl();
    SymNodeImpl* base = a ? a.get() : b.get();
    TP_CHECK(base != nullptr, "at least one symbolic boolean is required");
    if (!a) a = base->wrap_bool(left.expect_bool());
    if (!b) b = base->wrap_bool(right.expect_bool());
    return {std::move(a), std::move(b)};
}

} // namespace

SymBool SymBool::sym_and(const SymBool& other) const {
    if (auto left = maybe_as_bool()) {
        if (auto right = other.maybe_as_bool()) return SymBool(*left && *right);
    }
    auto values = normalize(*this, other);
    return SymBool(values.left->sym_and(values.right));
}

SymBool SymBool::sym_or(const SymBool& other) const {
    if (auto left = maybe_as_bool()) {
        if (auto right = other.maybe_as_bool()) return SymBool(*left || *right);
    }
    auto values = normalize(*this, other);
    return SymBool(values.left->sym_or(values.right));
}

SymBool SymBool::sym_xor(const SymBool& other) const {
    if (auto left = maybe_as_bool()) {
        if (auto right = other.maybe_as_bool()) return SymBool(*left != *right);
    }
    auto values = normalize(*this, other);
    return SymBool(values.left->sym_xor(values.right));
}

SymBool SymBool::sym_not() const {
    if (auto value = maybe_as_bool()) return SymBool(!*value);
    return SymBool(toSymNodeImplUnowned()->sym_not());
}

SymBool SymBool::sym_eq(const SymBool& other) const {
    if (auto left = maybe_as_bool()) {
        if (auto right = other.maybe_as_bool()) return SymBool(*left == *right);
    }
    auto values = normalize(*this, other);
    return SymBool(values.left->eq(values.right));
}

SymBool SymBool::sym_ne(const SymBool& other) const {
    if (auto left = maybe_as_bool()) {
        if (auto right = other.maybe_as_bool()) return SymBool(*left != *right);
    }
    auto values = normalize(*this, other);
    return SymBool(values.left->ne(values.right));
}

bool SymBool::equals(const SymBool& other) const {
    if (auto left = maybe_as_bool()) {
        if (auto right = other.maybe_as_bool()) return *left == *right;
    }
    auto values = normalize(*this, other);
    return values.left->eq(values.right)->guard_bool("symbolic", 0);
}

bool SymBool::expect_bool() const {
    if (auto value = maybe_as_bool()) return *value;
    TP_THROW(RuntimeError, "expected a concrete boolean value");
}

bool SymBool::guard_bool(const char* file, int64_t line) const {
    if (auto value = maybe_as_bool()) return *value;
    return toSymNodeImplUnowned()->guard_bool(file, line);
}

bool SymBool::expect_true(const char* file, int64_t line) const {
    return guard_bool(file, line);
}

bool SymBool::guard_size_oblivious(const char* file, int64_t line) const {
    if (auto value = maybe_as_bool()) return *value;
    return toSymNodeImplUnowned()->guard_size_oblivious(file, line);
}

bool SymBool::statically_known_true(const char* file, int64_t line) const {
    if (auto value = maybe_as_bool()) return *value;
    return toSymNodeImplUnowned()->statically_known_true(file, line);
}

bool SymBool::guard_or_false(const char* file, int64_t line) const {
    if (auto value = maybe_as_bool()) return *value;
    return toSymNodeImplUnowned()->guard_or_false(file, line);
}

bool SymBool::guard_or_true(const char* file, int64_t line) const {
    if (auto value = maybe_as_bool()) return *value;
    return toSymNodeImplUnowned()->guard_or_true(file, line);
}

bool SymBool::has_hint() const {
    if (maybe_as_bool()) return true;
    return toSymNodeImplUnowned()->has_hint();
}

SymInt SymBool::toSymInt() const {
    if (auto value = maybe_as_bool()) return SymInt(*value ? 1 : 0);
    SymNode node = toSymNodeImpl();
    return SymInt(node->sym_ite(node->wrap_int(1), node->wrap_int(0)));
}

SymFloat SymBool::toSymFloat() const {
    if (auto value = maybe_as_bool()) return SymFloat(*value ? 1.0 : 0.0);
    return SymFloat(toSymNodeImplUnowned()->sym_float());
}

std::ostream& operator<<(std::ostream& os, const SymBool& value) {
    if (auto concrete = value.maybe_as_bool()) {
        os << (*concrete ? "true" : "false");
    } else {
        os << value.toSymNodeImplUnowned()->str();
    }
    return os;
}

} // namespace tensorplay
