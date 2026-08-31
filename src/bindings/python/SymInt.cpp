#include "python_bindings.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename T>
std::string value_string(const T& value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

py::object not_implemented() {
    return py::reinterpret_borrow<py::object>(Py_NotImplemented);
}

bool is_symint(py::handle value) {
    return value && py::isinstance<SymInt>(value);
}

bool is_symbool(py::handle value) {
    return value && py::isinstance<SymBool>(value);
}

bool is_symfloat(py::handle value) {
    return value && py::isinstance<SymFloat>(value);
}

bool is_bool_value(py::handle value) {
    return value && PyBool_Check(value.ptr());
}

bool is_int_like(py::handle value) {
    return is_symint(value) || (value && PyLong_Check(value.ptr()));
}

bool is_float_like(py::handle value) {
    return is_symfloat(value) || (value && PyFloat_Check(value.ptr()));
}

bool is_bool_like(py::handle value) {
    return is_symbool(value) || is_bool_value(value);
}

bool is_number_like(py::handle value) {
    return is_int_like(value) || is_float_like(value) || is_bool_like(value);
}

SymInt as_symint(py::handle value) {
    if (is_symint(value)) return value.cast<SymInt>();
    if (is_int_like(value)) return SymInt(py::cast<int64_t>(value));
    throw py::type_error("expected an integer-like symbolic value");
}

SymFloat as_symfloat(py::handle value) {
    if (is_symfloat(value)) return value.cast<SymFloat>();
    if (is_symint(value)) return value.cast<SymInt>().sym_float();
    if (is_symbool(value)) return value.cast<SymBool>().toSymFloat();
    if (value && PyFloat_Check(value.ptr())) {
        return SymFloat(py::cast<double>(value));
    }
    if (value && PyLong_Check(value.ptr())) {
        return SymFloat(static_cast<double>(py::cast<int64_t>(value)));
    }
    throw py::type_error("expected a numeric symbolic value");
}

SymBool as_symbool(py::handle value) {
    if (is_symbool(value)) return value.cast<SymBool>();
    if (is_bool_value(value)) return SymBool(py::cast<bool>(value));
    throw py::type_error("expected a boolean symbolic value");
}

tensorplay::SymNode node_for(py::handle value, tensorplay::SymNodeImpl* base) {
    if (is_symint(value)) {
        const SymInt converted = value.cast<SymInt>();
        if (auto concrete = converted.maybe_as_int()) {
            return base ? base->wrap_int(*concrete)
                        : tensorplay::make_constant_int(*concrete);
        }
        return converted.toSymNode();
    }
    if (is_symfloat(value)) {
        const SymFloat converted = value.cast<SymFloat>();
        if (auto concrete = converted.maybe_as_float()) {
            return base ? base->wrap_float(*concrete)
                        : tensorplay::make_constant_float(*concrete);
        }
        return converted.toSymNode();
    }
    if (is_symbool(value)) {
        const SymBool converted = value.cast<SymBool>();
        if (auto concrete = converted.maybe_as_bool()) {
            return base ? base->wrap_bool(*concrete)
                        : tensorplay::make_constant_bool(*concrete);
        }
        return converted.toSymNode();
    }
    if (is_bool_value(value)) {
        const bool concrete = py::cast<bool>(value);
        return base ? base->wrap_bool(concrete)
                    : tensorplay::make_constant_bool(concrete);
    }
    if (value && PyLong_Check(value.ptr())) {
        const int64_t concrete = py::cast<int64_t>(value);
        return base ? base->wrap_int(concrete)
                    : tensorplay::make_constant_int(concrete);
    }
    if (value && PyFloat_Check(value.ptr())) {
        const double concrete = py::cast<double>(value);
        return base ? base->wrap_float(concrete)
                    : tensorplay::make_constant_float(concrete);
    }
    throw py::type_error("expected a symbolic scalar value");
}

py::object pack_node(tensorplay::SymNode node) {
    if (!node) throw std::runtime_error("symbolic operation returned no value");
    if (node->is_int()) return py::cast(SymInt(std::move(node)));
    if (node->is_bool()) return py::cast(SymBool(std::move(node)));
    if (node->is_float()) return py::cast(SymFloat(std::move(node)));
    throw std::runtime_error("symbolic operation returned an invalid type");
    return py::none();
}

py::object pack_symbool(const SymBool& value) {
    return py::cast(value);
}

SymFloat int_true_divide(const SymInt& left, const SymInt& right) {
    if (auto left_value = left.maybe_as_int()) {
        if (auto right_value = right.maybe_as_int()) {
            if (*right_value == 0) {
                throw py::value_error(
                    "integer division or modulo by zero");
            }
            return SymFloat(static_cast<double>(*left_value) /
                            static_cast<double>(*right_value));
        }
    }
    tensorplay::SymNode lhs;
    tensorplay::SymNode rhs;
    if (!left.maybe_as_int()) lhs = left.toSymNode();
    if (!right.maybe_as_int()) rhs = right.toSymNode();
    tensorplay::SymNodeImpl* base = lhs ? lhs.get() : rhs.get();
    if (!base) throw std::runtime_error("integer division has no operands");
    if (!lhs) lhs = base->wrap_int(*left.maybe_as_int());
    if (!rhs) rhs = base->wrap_int(*right.maybe_as_int());
    return SymFloat(lhs->int_truediv(rhs));
}

py::object int_power(const SymInt& base, const SymInt& exponent) {
    if (auto value = exponent.maybe_as_int()) {
        if (*value >= 0) return py::cast(base.pow_by_natural(exponent));
        return py::cast(base.sym_float().pow(exponent.sym_float()));
    }
    const bool nonnegative = exponent.sym_ge(SymInt(0)).guard_bool("", 0);
    if (nonnegative) return py::cast(base.pow_by_natural(exponent));
    return py::cast(base.sym_float().pow(exponent.sym_float()));
}

SymBool compare_int(const SymInt& left, const SymInt& right, int operation) {
    switch (operation) {
        case 0: return left.sym_eq(right);
        case 1: return left.sym_ne(right);
        case 2: return left.sym_lt(right);
        case 3: return left.sym_le(right);
        case 4: return left.sym_gt(right);
        case 5: return left.sym_ge(right);
        default: throw py::value_error("unknown symbolic comparison");
    }
}

SymBool compare_float(
    const SymFloat& left, const SymFloat& right, int operation) {
    switch (operation) {
        case 0: return left.sym_eq(right);
        case 1: return left.sym_ne(right);
        case 2: return left.sym_lt(right);
        case 3: return left.sym_le(right);
        case 4: return left.sym_gt(right);
        case 5: return left.sym_ge(right);
        default: throw py::value_error("unknown symbolic comparison");
    }
}

py::object symint_compare(
    const SymInt& left, py::handle other, int operation) {
    if (is_int_like(other)) {
        return pack_symbool(compare_int(left, as_symint(other), operation));
    }
    if (is_float_like(other)) {
        return pack_symbool(compare_float(
            left.sym_float(), as_symfloat(other), operation));
    }
    return not_implemented();
}

py::object symfloat_compare(
    const SymFloat& left, py::handle other, int operation) {
    if (!is_number_like(other)) return not_implemented();
    return pack_symbool(compare_float(left, as_symfloat(other), operation));
}

py::object symbool_compare(
    const SymBool& left, py::handle other, int operation) {
    if (is_bool_like(other)) {
        const SymBool right = as_symbool(other);
        return pack_symbool(operation == 0 ? left.sym_eq(right)
                                           : left.sym_ne(right));
    }
    if (is_int_like(other)) {
        const SymInt lhs = left.toSymInt();
        return pack_symbool(compare_int(lhs, as_symint(other), operation));
    }
    return not_implemented();
}

py::object symint_binary(
    const SymInt& left, py::handle other, int operation) {
    if (is_int_like(other)) {
        const SymInt right = as_symint(other);
        switch (operation) {
            case 0: return py::cast(left + right);
            case 1: return py::cast(left - right);
            case 2: return py::cast(left * right);
            case 3: return py::cast(left / right);
            case 4: return py::cast(left % right);
            case 5: return py::cast(left & right);
            case 6: return py::cast(left | right);
            case 7: return py::cast(left ^ right);
            case 8: return py::cast(left << right);
            case 9: return py::cast(left >> right);
            case 10: return py::cast(left.min(right));
            case 11: return py::cast(left.max(right));
            default: break;
        }
    }
    if (is_float_like(other)) {
        const SymFloat lhs = left.sym_float();
        const SymFloat rhs = as_symfloat(other);
        switch (operation) {
            case 0: return py::cast(lhs + rhs);
            case 1: return py::cast(lhs - rhs);
            case 2: return py::cast(lhs * rhs);
            case 3: return py::cast(lhs / rhs);
            case 4: return py::cast(lhs % rhs);
            case 10: return py::cast(lhs.min(rhs));
            case 11: return py::cast(lhs.max(rhs));
            default: break;
        }
    }
    return not_implemented();
}

py::object symint_reverse_binary(
    const SymInt& right, py::handle other, int operation) {
    if (is_int_like(other)) {
        const SymInt left = as_symint(other);
        switch (operation) {
            case 0: return py::cast(left + right);
            case 1: return py::cast(left - right);
            case 2: return py::cast(left * right);
            case 3: return py::cast(left / right);
            case 4: return py::cast(left % right);
            case 5: return py::cast(left & right);
            case 6: return py::cast(left | right);
            case 7: return py::cast(left ^ right);
            case 8: return py::cast(left << right);
            case 9: return py::cast(left >> right);
            case 10: return py::cast(left.min(right));
            case 11: return py::cast(left.max(right));
            default: break;
        }
    }
    if (is_float_like(other)) {
        const SymFloat left = as_symfloat(other);
        const SymFloat rhs = right.sym_float();
        switch (operation) {
            case 0: return py::cast(left + rhs);
            case 1: return py::cast(left - rhs);
            case 2: return py::cast(left * rhs);
            case 3: return py::cast(left / rhs);
            case 4: return py::cast(left % rhs);
            case 10: return py::cast(left.min(rhs));
            case 11: return py::cast(left.max(rhs));
            default: break;
        }
    }
    return not_implemented();
}

py::object symint_truediv(const SymInt& left, py::handle other) {
    if (is_int_like(other)) {
        return py::cast(int_true_divide(left, as_symint(other)));
    }
    if (is_float_like(other)) {
        return py::cast(left.sym_float() / as_symfloat(other));
    }
    return not_implemented();
}

py::object symint_reverse_truediv(const SymInt& right, py::handle other) {
    if (is_int_like(other)) {
        return py::cast(int_true_divide(as_symint(other), right));
    }
    if (is_float_like(other)) {
        return py::cast(as_symfloat(other) / right.sym_float());
    }
    return not_implemented();
}

py::object symint_floordiv(const SymInt& left, py::handle other) {
    if (is_int_like(other)) return py::cast(left / as_symint(other));
    if (is_float_like(other)) {
        return py::cast(left.sym_float().floor_div(as_symfloat(other)));
    }
    return not_implemented();
}

py::object symint_reverse_floordiv(const SymInt& right, py::handle other) {
    if (is_int_like(other)) return py::cast(as_symint(other) / right);
    if (is_float_like(other)) {
        return py::cast(as_symfloat(other).floor_div(right.sym_float()));
    }
    return not_implemented();
}

py::object symint_mod(const SymInt& left, py::handle other) {
    if (is_int_like(other)) return py::cast(left % as_symint(other));
    if (is_float_like(other)) {
        return py::cast(left.sym_float() % as_symfloat(other));
    }
    return not_implemented();
}

py::object symint_reverse_mod(const SymInt& right, py::handle other) {
    if (is_int_like(other)) return py::cast(as_symint(other) % right);
    if (is_float_like(other)) {
        return py::cast(as_symfloat(other) % right.sym_float());
    }
    return not_implemented();
}

py::object symint_pow(const SymInt& base, py::handle exponent) {
    if (is_float_like(exponent)) {
        return py::cast(base.sym_float().pow(as_symfloat(exponent)));
    }
    if (is_int_like(exponent)) return int_power(base, as_symint(exponent));
    return not_implemented();
}

py::object symint_reverse_pow(const SymInt& exponent, py::handle base) {
    if (is_float_like(base)) {
        return py::cast(as_symfloat(base).pow(exponent.sym_float()));
    }
    if (is_int_like(base)) return int_power(as_symint(base), exponent);
    return not_implemented();
}

py::object symfloat_binary(
    const SymFloat& left, py::handle other, int operation) {
    if (!is_number_like(other)) return not_implemented();
    const SymFloat right = as_symfloat(other);
    switch (operation) {
        case 0: return py::cast(left + right);
        case 1: return py::cast(left - right);
        case 2: return py::cast(left * right);
        case 3: return py::cast(left / right);
        case 4: return py::cast(left % right);
        case 5: return py::cast(left.min(right));
        case 6: return py::cast(left.max(right));
        default: return not_implemented();
    }
}

py::object symfloat_reverse_binary(
    const SymFloat& right, py::handle other, int operation) {
    if (!is_number_like(other)) return not_implemented();
    const SymFloat left = as_symfloat(other);
    switch (operation) {
        case 0: return py::cast(left + right);
        case 1: return py::cast(left - right);
        case 2: return py::cast(left * right);
        case 3: return py::cast(left / right);
        case 4: return py::cast(left % right);
        case 5: return py::cast(left.min(right));
        case 6: return py::cast(left.max(right));
        default: return not_implemented();
    }
}

py::object symbool_numeric_binary(
    const SymBool& left, py::handle other, int operation, bool reverse) {
    if (is_float_like(other)) {
        const SymFloat lhs = left.toSymFloat();
        const SymFloat rhs = as_symfloat(other);
        switch (operation) {
            case 0: return py::cast(reverse ? rhs + lhs : lhs + rhs);
            case 1: return py::cast(reverse ? rhs - lhs : lhs - rhs);
            case 2: return py::cast(reverse ? rhs * lhs : lhs * rhs);
            default: return not_implemented();
        }
    }
    if (is_bool_like(other)) {
        const SymInt lhs = left.toSymInt();
        const SymInt rhs = as_symbool(other).toSymInt();
        switch (operation) {
            case 0: return py::cast(reverse ? rhs + lhs : lhs + rhs);
            case 1: return py::cast(reverse ? rhs - lhs : lhs - rhs);
            case 2: return py::cast(reverse ? rhs * lhs : lhs * rhs);
            default: return not_implemented();
        }
    }
    if (is_int_like(other)) {
        const SymInt lhs = left.toSymInt();
        const SymInt rhs = as_symint(other);
        switch (operation) {
            case 0: return py::cast(reverse ? rhs + lhs : lhs + rhs);
            case 1: return py::cast(reverse ? rhs - lhs : lhs - rhs);
            case 2: return py::cast(reverse ? rhs * lhs : lhs * rhs);
            default: return not_implemented();
        }
    }
    return not_implemented();
}

py::object sym_ite(
    py::handle condition, py::handle then_value, py::handle else_value) {
    if (!is_bool_like(condition)) {
        throw py::type_error("sym_ite expects a boolean condition");
    }
    if (is_bool_value(condition)) {
        if (Py_TYPE(then_value.ptr()) != Py_TYPE(else_value.ptr())) {
            throw py::type_error("sym_ite branches must have the same type");
        }
        return py::reinterpret_borrow<py::object>(
            py::cast<bool>(condition) ? then_value : else_value);
    }
    if (!is_number_like(then_value) || !is_number_like(else_value)) {
        throw py::type_error(
            "sym_ite expects scalar branches for a symbolic condition");
    }
    if (Py_TYPE(then_value.ptr()) != Py_TYPE(else_value.ptr())) {
        throw py::type_error("sym_ite branches must have the same type");
    }
    const SymBool cond = as_symbool(condition);
    tensorplay::SymNode condition_node = cond.maybe_as_bool()
        ? tensorplay::make_constant_bool(*cond.maybe_as_bool())
        : cond.toSymNode();
    tensorplay::SymNodeImpl* base = condition_node.get();
    tensorplay::SymNode then_node = node_for(then_value, base);
    tensorplay::SymNode else_node = node_for(else_value, base);
    if (then_node->value_type() != else_node->value_type()) {
        throw py::type_error("sym_ite branches must have the same type");
    }
    return pack_node(condition_node->sym_ite(then_node, else_node));
}

template <SymFloat (SymFloat::*Method)() const>
py::object symint_math(const SymInt& value) {
    return py::cast((value.sym_float().*Method)());
}

template <SymFloat (SymFloat::*Method)() const>
py::object symfloat_math(const SymFloat& value) {
    return py::cast((value.*Method)());
}

template <SymFloat (SymFloat::*Method)() const>
py::object sym_math_scalar(py::handle value, const char* name) {
    if (is_symint(value)) {
        return py::cast((value.cast<SymInt>().sym_float().*Method)());
    }
    if (is_symfloat(value)) {
        return py::cast((value.cast<SymFloat>().*Method)());
    }
    if (value && (PyLong_Check(value.ptr()) || PyFloat_Check(value.ptr()))) {
        return py::module_::import("math").attr(name)(value);
    }
    return not_implemented();
}

}  // namespace

void init_symint(py::module_& m) {
    auto symint = py::class_<SymInt>(m, "SymInt");
    auto symbool = py::class_<SymBool>(m, "SymBool");
    auto symfloat = py::class_<SymFloat>(m, "SymFloat");

    symint
        .def(py::init<int64_t>())
        .def_static(
            "symbolic",
            [](std::string name, std::optional<int64_t> hint) {
                return SymInt(tensorplay::make_symbolic_int(
                    std::move(name), hint));
            },
            py::arg("name"), py::arg("hint") = std::nullopt)
        .def("expect_int", &SymInt::expect_int)
        .def("guard_int",
             [](const SymInt& value, const std::string& file, int64_t line) {
                 return value.guard_int(file.c_str(), line);
             },
             py::arg("file") = "", py::arg("line") = 0)
        .def("has_hint", &SymInt::has_hint)
        .def("is_symbolic", &SymInt::is_symbolic)
        .def("is_heap_allocated", &SymInt::is_heap_allocated)
        .def("maybe_as_int", &SymInt::maybe_as_int)
        .def("clone", &SymInt::clone)
        .def("is_same", &SymInt::is_same)
        .def("min", &SymInt::min)
        .def("max", &SymInt::max)
        .def("pow_by_natural", &SymInt::pow_by_natural)
        .def("ceil", &SymInt::ceil)
        .def("floor", &SymInt::floor)
        .def("trunc", &SymInt::trunc)
        .def("round", &SymInt::round)
        .def("sym_float", &SymInt::sym_float)
        .def("sym_eq", &SymInt::sym_eq)
        .def("sym_ne", &SymInt::sym_ne)
        .def("sym_lt", &SymInt::sym_lt)
        .def("sym_le", &SymInt::sym_le)
        .def("sym_gt", &SymInt::sym_gt)
        .def("sym_ge", &SymInt::sym_ge)
        .def("to_sym_float", &SymInt::sym_float)
        .def("__add__", [](const SymInt& left, py::object right) {
            return symint_binary(left, right, 0);
        }, py::is_operator())
        .def("__radd__", [](const SymInt& right, py::object left) {
            return symint_reverse_binary(right, left, 0);
        }, py::is_operator())
        .def("__sub__", [](const SymInt& left, py::object right) {
            return symint_binary(left, right, 1);
        }, py::is_operator())
        .def("__rsub__", [](const SymInt& right, py::object left) {
            return symint_reverse_binary(right, left, 1);
        }, py::is_operator())
        .def("__mul__", [](const SymInt& left, py::object right) {
            return symint_binary(left, right, 2);
        }, py::is_operator())
        .def("__rmul__", [](const SymInt& right, py::object left) {
            return symint_reverse_binary(right, left, 2);
        }, py::is_operator())
        .def("__truediv__", [](const SymInt& left, py::object right) {
            return symint_truediv(left, right);
        }, py::is_operator())
        .def("__rtruediv__", [](const SymInt& right, py::object left) {
            return symint_reverse_truediv(right, left);
        }, py::is_operator())
        .def("__floordiv__", [](const SymInt& left, py::object right) {
            return symint_floordiv(left, right);
        }, py::is_operator())
        .def("__rfloordiv__", [](const SymInt& right, py::object left) {
            return symint_reverse_floordiv(right, left);
        }, py::is_operator())
        .def("__mod__", [](const SymInt& left, py::object right) {
            return symint_mod(left, right);
        }, py::is_operator())
        .def("__rmod__", [](const SymInt& right, py::object left) {
            return symint_reverse_mod(right, left);
        }, py::is_operator())
        .def("__pow__", [](const SymInt& base, py::object exponent) {
            return symint_pow(base, exponent);
        }, py::is_operator())
        .def("__rpow__", [](const SymInt& exponent, py::object base) {
            return symint_reverse_pow(exponent, base);
        }, py::is_operator())
        .def("__float_truediv__", [](const SymInt& left, py::object right) {
            if (!is_number_like(right)) return not_implemented();
            return py::cast(left.sym_float() / as_symfloat(right));
        })
        .def("__rfloat_truediv__", [](const SymInt& right, py::object left) {
            if (!is_number_like(left)) return not_implemented();
            return py::cast(as_symfloat(left) / right.sym_float());
        })
        .def("__int_truediv__", [](const SymInt& left, py::object right) {
            if (!is_int_like(right)) return not_implemented();
            return py::cast(int_true_divide(left, as_symint(right)));
        })
        .def("__rint_truediv__", [](const SymInt& right, py::object left) {
            if (!is_int_like(left)) return not_implemented();
            return py::cast(int_true_divide(as_symint(left), right));
        })
        .def("__int_floordiv__", [](const SymInt& left, py::object right) {
            if (!is_int_like(right)) return not_implemented();
            return py::cast(left / as_symint(right));
        })
        .def("__rint_floordiv__", [](const SymInt& right, py::object left) {
            if (!is_int_like(left)) return not_implemented();
            return py::cast(as_symint(left) / right);
        })
        .def("__pow_by_natural__", [](const SymInt& base, py::object exponent) {
            if (!is_int_like(exponent)) return not_implemented();
            return py::cast(base.pow_by_natural(as_symint(exponent)));
        })
        .def("__rpow_by_natural__", [](const SymInt& exponent, py::object base) {
            if (!is_int_like(base)) return not_implemented();
            return py::cast(as_symint(base).pow_by_natural(exponent));
        })
        .def("__and__", [](const SymInt& left, py::object right) {
            if (!is_int_like(right)) return not_implemented();
            return py::cast(left & as_symint(right));
        }, py::is_operator())
        .def("__rand__", [](const SymInt& right, py::object left) {
            if (!is_int_like(left)) return not_implemented();
            return py::cast(as_symint(left) & right);
        }, py::is_operator())
        .def("__or__", [](const SymInt& left, py::object right) {
            if (!is_int_like(right)) return not_implemented();
            return py::cast(left | as_symint(right));
        }, py::is_operator())
        .def("__ror__", [](const SymInt& right, py::object left) {
            if (!is_int_like(left)) return not_implemented();
            return py::cast(as_symint(left) | right);
        }, py::is_operator())
        .def("__xor__", [](const SymInt& left, py::object right) {
            if (!is_int_like(right)) return not_implemented();
            return py::cast(left ^ as_symint(right));
        }, py::is_operator())
        .def("__rxor__", [](const SymInt& right, py::object left) {
            if (!is_int_like(left)) return not_implemented();
            return py::cast(as_symint(left) ^ right);
        }, py::is_operator())
        .def("__lshift__", [](const SymInt& left, py::object right) {
            if (!is_int_like(right)) return not_implemented();
            return py::cast(left << as_symint(right));
        }, py::is_operator())
        .def("__rlshift__", [](const SymInt& right, py::object left) {
            if (!is_int_like(left)) return not_implemented();
            return py::cast(as_symint(left) << right);
        }, py::is_operator())
        .def("__rshift__", [](const SymInt& left, py::object right) {
            if (!is_int_like(right)) return not_implemented();
            return py::cast(left >> as_symint(right));
        }, py::is_operator())
        .def("__rrshift__", [](const SymInt& right, py::object left) {
            if (!is_int_like(left)) return not_implemented();
            return py::cast(as_symint(left) >> right);
        }, py::is_operator())
        .def("__neg__", [](const SymInt& value) { return -value; },
             py::is_operator())
        .def("__pos__", [](const SymInt& value) {
            if (auto concrete = value.maybe_as_int()) return SymInt(*concrete);
            return SymInt(value.toSymNodeImplUnowned()->pos());
        }, py::is_operator())
        .def("__abs__", [](const SymInt& value) {
            if (auto concrete = value.maybe_as_int()) {
                return SymInt(tensorplay::make_constant_int(*concrete)->abs());
            }
            return SymInt(value.toSymNodeImplUnowned()->abs());
        }, py::is_operator())
        .def("__ceil__", &SymInt::ceil)
        .def("__floor__", &SymInt::floor)
        .def("__trunc__", &SymInt::trunc)
        .def("__round__", [](const SymInt& value, py::object ndigits) {
            if (!ndigits.is_none() && !is_int_like(ndigits)) {
                throw py::type_error("round precision must be an integer");
            }
            return value.round();
        }, py::arg("ndigits") = py::none())
        .def("__sym_float__", &SymInt::sym_float)
        .def("__sym_int__", [](const SymInt& value) { return value; })
        .def("__sym_min__", [](const SymInt& left, py::object right) {
            if (is_int_like(right)) return py::cast(left.min(as_symint(right)));
            if (is_float_like(right)) {
                return py::cast(left.sym_float().min(as_symfloat(right)));
            }
            return not_implemented();
        })
        .def("__sym_max__", [](const SymInt& left, py::object right) {
            if (is_int_like(right)) return py::cast(left.max(as_symint(right)));
            if (is_float_like(right)) {
                return py::cast(left.sym_float().max(as_symfloat(right)));
            }
            return not_implemented();
        })
        .def("__sym_sqrt__", &symint_math<&SymFloat::sqrt>)
        .def("__sym_cos__", &symint_math<&SymFloat::cos>)
        .def("__sym_cosh__", &symint_math<&SymFloat::cosh>)
        .def("__sym_sin__", &symint_math<&SymFloat::sin>)
        .def("__sym_sinh__", &symint_math<&SymFloat::sinh>)
        .def("__sym_tan__", &symint_math<&SymFloat::tan>)
        .def("__sym_tanh__", &symint_math<&SymFloat::tanh>)
        .def("__sym_asin__", &symint_math<&SymFloat::asin>)
        .def("__sym_acos__", &symint_math<&SymFloat::acos>)
        .def("__sym_atan__", &symint_math<&SymFloat::atan>)
        .def("__sym_log2__", &symint_math<&SymFloat::log2>)
        .def("as_integer_ratio", [](const SymInt& value) {
            return py::make_tuple(value, 1);
        })
        .def("bit_length", [](const SymInt& value) {
            uint64_t magnitude;
            const int64_t concrete = value.expect_int();
            if (concrete < 0) {
                magnitude = static_cast<uint64_t>(-(concrete + 1)) + 1;
            } else {
                magnitude = static_cast<uint64_t>(concrete);
            }
            int64_t result = 0;
            while (magnitude != 0) {
                ++result;
                magnitude >>= 1;
            }
            return result;
        })
        .def("conjugate", [](const SymInt& value) { return value; })
        .def_property_readonly("hint", [](const SymInt& value) -> py::object {
            if (!value.has_hint()) return py::none();
            return py::int_(value.guard_int("", 0));
        })
        .def_property_readonly("constant", [](const SymInt& value) -> py::object {
            if (auto concrete = value.maybe_as_int()) return py::int_(*concrete);
            return py::none();
        })
        .def("__eq__", [](const SymInt& left, py::object right) {
            return symint_compare(left, right, 0);
        }, py::is_operator())
        .def("__ne__", [](const SymInt& left, py::object right) {
            return symint_compare(left, right, 1);
        }, py::is_operator())
        .def("__lt__", [](const SymInt& left, py::object right) {
            return symint_compare(left, right, 2);
        }, py::is_operator())
        .def("__le__", [](const SymInt& left, py::object right) {
            return symint_compare(left, right, 3);
        }, py::is_operator())
        .def("__gt__", [](const SymInt& left, py::object right) {
            return symint_compare(left, right, 4);
        }, py::is_operator())
        .def("__ge__", [](const SymInt& left, py::object right) {
            return symint_compare(left, right, 5);
        }, py::is_operator())
        .def("__int__", &SymInt::expect_int)
        .def("__index__", &SymInt::expect_int)
        .def("__bool__", [](const SymInt& value) {
            return value.sym_ne(SymInt(0)).guard_bool("", 0);
        })
        .def("__hash__", [](const SymInt& value) {
            return static_cast<py::ssize_t>(
                std::hash<int64_t>{}(value.expect_int()));
        })
        .def("__str__", [](const SymInt& value) { return value_string(value); })
        .def("__repr__", [](const SymInt& value) { return value_string(value); });

    symbool
        .def(py::init<bool>())
        .def_static(
            "symbolic",
            [](std::string name, std::optional<bool> hint) {
                return SymBool(tensorplay::make_symbolic_bool(
                    std::move(name), hint));
            },
            py::arg("name"), py::arg("hint") = std::nullopt)
        .def("expect_bool", &SymBool::expect_bool)
        .def("guard_bool",
             [](const SymBool& value, const std::string& file, int64_t line) {
                 return value.guard_bool(file.c_str(), line);
             },
             py::arg("file") = "", py::arg("line") = 0)
        .def("guard_size_oblivious",
             [](const SymBool& value, const std::string& file, int64_t line) {
                 return value.guard_size_oblivious(file.c_str(), line);
             },
             py::arg("file") = "", py::arg("line") = 0)
        .def("statically_known_true",
             [](const SymBool& value, const std::string& file, int64_t line) {
                 return value.statically_known_true(file.c_str(), line);
             },
             py::arg("file") = "", py::arg("line") = 0)
        .def("guard_or_false",
             [](const SymBool& value, const std::string& file, int64_t line) {
                 return value.guard_or_false(file.c_str(), line);
             },
             py::arg("file") = "", py::arg("line") = 0)
        .def("guard_or_true",
             [](const SymBool& value, const std::string& file, int64_t line) {
                 return value.guard_or_true(file.c_str(), line);
             },
             py::arg("file") = "", py::arg("line") = 0)
        .def("has_hint", &SymBool::has_hint)
        .def("is_symbolic", &SymBool::is_symbolic)
        .def("maybe_as_bool", &SymBool::maybe_as_bool)
        .def("to_sym_int", &SymBool::toSymInt)
        .def("to_sym_float", &SymBool::toSymFloat)
        .def("sym_and", &SymBool::sym_and)
        .def("sym_or", &SymBool::sym_or)
        .def("sym_xor", &SymBool::sym_xor)
        .def("sym_not", &SymBool::sym_not)
        .def("__and__", [](const SymBool& left, py::object right) {
            if (!is_bool_like(right)) return not_implemented();
            return py::cast(left.sym_and(as_symbool(right)));
        }, py::is_operator())
        .def("__rand__", [](const SymBool& right, py::object left) {
            if (!is_bool_like(left)) return not_implemented();
            return py::cast(as_symbool(left).sym_and(right));
        }, py::is_operator())
        .def("__or__", [](const SymBool& left, py::object right) {
            if (!is_bool_like(right)) return not_implemented();
            return py::cast(left.sym_or(as_symbool(right)));
        }, py::is_operator())
        .def("__ror__", [](const SymBool& right, py::object left) {
            if (!is_bool_like(left)) return not_implemented();
            return py::cast(as_symbool(left).sym_or(right));
        }, py::is_operator())
        .def("__xor__", [](const SymBool& left, py::object right) {
            if (!is_bool_like(right)) return not_implemented();
            return py::cast(left.sym_xor(as_symbool(right)));
        }, py::is_operator())
        .def("__rxor__", [](const SymBool& right, py::object left) {
            if (!is_bool_like(left)) return not_implemented();
            return py::cast(as_symbool(left).sym_xor(right));
        }, py::is_operator())
        .def("__add__", [](const SymBool& left, py::object right) {
            return symbool_numeric_binary(left, right, 0, false);
        }, py::is_operator())
        .def("__radd__", [](const SymBool& right, py::object left) {
            return symbool_numeric_binary(right, left, 0, true);
        }, py::is_operator())
        .def("__sub__", [](const SymBool& left, py::object right) {
            return symbool_numeric_binary(left, right, 1, false);
        }, py::is_operator())
        .def("__rsub__", [](const SymBool& right, py::object left) {
            return symbool_numeric_binary(right, left, 1, true);
        }, py::is_operator())
        .def("__mul__", [](const SymBool& left, py::object right) {
            return symbool_numeric_binary(left, right, 2, false);
        }, py::is_operator())
        .def("__rmul__", [](const SymBool& right, py::object left) {
            return symbool_numeric_binary(right, left, 2, true);
        }, py::is_operator())
        .def("__sym_not__", &SymBool::sym_not)
        .def("__sym_float__", &SymBool::toSymFloat)
        .def("__sym_ite__", [](const SymBool& condition, py::object then_value,
                                py::object else_value) {
            return sym_ite(py::cast(condition), then_value, else_value);
        })
        .def("__eq__", [](const SymBool& left, py::object right) {
            return symbool_compare(left, right, 0);
        }, py::is_operator())
        .def("__ne__", [](const SymBool& left, py::object right) {
            return symbool_compare(left, right, 1);
        }, py::is_operator())
        .def("__bool__", [](const SymBool& value) {
            return value.guard_bool("", 0);
        })
        .def("__int__", [](const SymBool& value) {
            return static_cast<int64_t>(value.guard_bool("", 0));
        })
        .def("__hash__", [](const SymBool& value) {
            return static_cast<py::ssize_t>(
                std::hash<bool>{}(value.expect_bool()));
        })
        .def_property_readonly("hint", [](const SymBool& value) -> py::object {
            if (!value.has_hint()) return py::none();
            return py::bool_(value.guard_bool("", 0));
        })
        .def_property_readonly("constant", [](const SymBool& value) -> py::object {
            if (auto concrete = value.maybe_as_bool()) return py::bool_(*concrete);
            return py::none();
        })
        .def("__str__", [](const SymBool& value) { return value_string(value); })
        .def("__repr__", [](const SymBool& value) { return value_string(value); });

    symfloat
        .def(py::init<double>())
        .def_static(
            "symbolic",
            [](std::string name, std::optional<double> hint) {
                return SymFloat(tensorplay::make_symbolic_float(
                    std::move(name), hint));
            },
            py::arg("name"), py::arg("hint") = std::nullopt)
        .def("expect_float", &SymFloat::expect_float)
        .def("guard_float",
             [](const SymFloat& value, const std::string& file, int64_t line) {
                 return value.guard_float(file.c_str(), line);
             },
             py::arg("file") = "", py::arg("line") = 0)
        .def("has_hint", &SymFloat::has_hint)
        .def("is_symbolic", &SymFloat::is_symbolic)
        .def("maybe_as_float", &SymFloat::maybe_as_float)
        .def("min", &SymFloat::min)
        .def("max", &SymFloat::max)
        .def("sqrt", &SymFloat::sqrt)
        .def("cos", &SymFloat::cos)
        .def("cosh", &SymFloat::cosh)
        .def("sin", &SymFloat::sin)
        .def("sinh", &SymFloat::sinh)
        .def("tan", &SymFloat::tan)
        .def("tanh", &SymFloat::tanh)
        .def("asin", &SymFloat::asin)
        .def("acos", &SymFloat::acos)
        .def("atan", &SymFloat::atan)
        .def("log2", &SymFloat::log2)
        .def("ceil", &SymFloat::ceil)
        .def("floor", &SymFloat::floor)
        .def("trunc", &SymFloat::trunc)
        .def("round", [](const SymFloat& value, py::object ndigits) -> py::object {
            if (ndigits.is_none()) return py::cast(value.round());
            if (!is_int_like(ndigits)) {
                throw py::type_error("round precision must be an integer");
            }
            return py::cast(value.round(as_symint(ndigits)));
        }, py::arg("ndigits") = py::none())
        .def("is_integer", &SymFloat::is_integer)
        .def("sym_int", &SymFloat::sym_int)
        .def("sym_float", [](const SymFloat& value) { return value; })
        .def("sym_eq", &SymFloat::sym_eq)
        .def("sym_ne", &SymFloat::sym_ne)
        .def("sym_lt", &SymFloat::sym_lt)
        .def("sym_le", &SymFloat::sym_le)
        .def("sym_gt", &SymFloat::sym_gt)
        .def("sym_ge", &SymFloat::sym_ge)
        .def("__add__", [](const SymFloat& left, py::object right) {
            return symfloat_binary(left, right, 0);
        }, py::is_operator())
        .def("__radd__", [](const SymFloat& right, py::object left) {
            return symfloat_reverse_binary(right, left, 0);
        }, py::is_operator())
        .def("__sub__", [](const SymFloat& left, py::object right) {
            return symfloat_binary(left, right, 1);
        }, py::is_operator())
        .def("__rsub__", [](const SymFloat& right, py::object left) {
            return symfloat_reverse_binary(right, left, 1);
        }, py::is_operator())
        .def("__mul__", [](const SymFloat& left, py::object right) {
            return symfloat_binary(left, right, 2);
        }, py::is_operator())
        .def("__rmul__", [](const SymFloat& right, py::object left) {
            return symfloat_reverse_binary(right, left, 2);
        }, py::is_operator())
        .def("__truediv__", [](const SymFloat& left, py::object right) {
            return symfloat_binary(left, right, 3);
        }, py::is_operator())
        .def("__rtruediv__", [](const SymFloat& right, py::object left) {
            return symfloat_reverse_binary(right, left, 3);
        }, py::is_operator())
        .def("__floordiv__", [](const SymFloat& left, py::object right) {
            if (!is_number_like(right)) return not_implemented();
            return py::cast(left.floor_div(as_symfloat(right)));
        }, py::is_operator())
        .def("__rfloordiv__", [](const SymFloat& right, py::object left) {
            if (!is_number_like(left)) return not_implemented();
            return py::cast(as_symfloat(left).floor_div(right));
        }, py::is_operator())
        .def("__mod__", [](const SymFloat& left, py::object right) {
            return symfloat_binary(left, right, 4);
        }, py::is_operator())
        .def("__rmod__", [](const SymFloat& right, py::object left) {
            return symfloat_reverse_binary(right, left, 4);
        }, py::is_operator())
        .def("__pow__", [](const SymFloat& base, py::object exponent) {
            if (!is_number_like(exponent)) return not_implemented();
            return py::cast(base.pow(as_symfloat(exponent)));
        }, py::is_operator())
        .def("__rpow__", [](const SymFloat& exponent, py::object base) {
            if (!is_number_like(base)) return not_implemented();
            return py::cast(as_symfloat(base).pow(exponent));
        }, py::is_operator())
        .def("__float_truediv__", [](const SymFloat& left, py::object right) {
            if (!is_number_like(right)) return not_implemented();
            return py::cast(left / as_symfloat(right));
        })
        .def("__rfloat_truediv__", [](const SymFloat& right, py::object left) {
            if (!is_number_like(left)) return not_implemented();
            return py::cast(as_symfloat(left) / right);
        })
        .def("__float_pow__", [](const SymFloat& base, py::object exponent) {
            if (!is_number_like(exponent)) return not_implemented();
            return py::cast(base.pow(as_symfloat(exponent)));
        })
        .def("__rfloat_pow__", [](const SymFloat& exponent, py::object base) {
            if (!is_number_like(base)) return not_implemented();
            return py::cast(as_symfloat(base).pow(exponent));
        })
        .def("__int_floordiv__", [](const SymFloat& left, py::object right) {
            if (!is_number_like(right)) return not_implemented();
            return py::cast(left.floor_div(as_symfloat(right)));
        })
        .def("__rint_floordiv__", [](const SymFloat& right, py::object left) {
            if (!is_number_like(left)) return not_implemented();
            return py::cast(as_symfloat(left).floor_div(right));
        })
        .def("__neg__", [](const SymFloat& value) { return -value; },
             py::is_operator())
        .def("__pos__", [](const SymFloat& value) { return +value; },
             py::is_operator())
        .def("__abs__", &SymFloat::abs, py::is_operator())
        .def("__ceil__", &SymFloat::ceil)
        .def("__floor__", &SymFloat::floor)
        .def("__trunc__", &SymFloat::trunc)
        .def("__round__", [](const SymFloat& value, py::object ndigits) -> py::object {
            if (ndigits.is_none()) return py::cast(value.round());
            if (!is_int_like(ndigits)) {
                throw py::type_error("round precision must be an integer");
            }
            return py::cast(value.round(as_symint(ndigits)));
        }, py::arg("ndigits") = py::none())
        .def("__sym_int__", &SymFloat::sym_int)
        .def("__sym_float__", [](const SymFloat& value) { return value; })
        .def("__sym_min__", [](const SymFloat& left, py::object right) {
            if (!is_number_like(right)) return not_implemented();
            return py::cast(left.min(as_symfloat(right)));
        })
        .def("__sym_max__", [](const SymFloat& left, py::object right) {
            if (!is_number_like(right)) return not_implemented();
            return py::cast(left.max(as_symfloat(right)));
        })
        .def("__sym_sqrt__", &symfloat_math<&SymFloat::sqrt>)
        .def("__sym_cos__", &symfloat_math<&SymFloat::cos>)
        .def("__sym_cosh__", &symfloat_math<&SymFloat::cosh>)
        .def("__sym_sin__", &symfloat_math<&SymFloat::sin>)
        .def("__sym_sinh__", &symfloat_math<&SymFloat::sinh>)
        .def("__sym_tan__", &symfloat_math<&SymFloat::tan>)
        .def("__sym_tanh__", &symfloat_math<&SymFloat::tanh>)
        .def("__sym_asin__", &symfloat_math<&SymFloat::asin>)
        .def("__sym_acos__", &symfloat_math<&SymFloat::acos>)
        .def("__sym_atan__", &symfloat_math<&SymFloat::atan>)
        .def("__sym_log2__", &symfloat_math<&SymFloat::log2>)
        .def("as_integer_ratio", [](const SymFloat& value) {
            return py::float_(value.expect_float()).attr("as_integer_ratio")();
        })
        .def("conjugate", [](const SymFloat& value) { return value; })
        .def("hex", [](const SymFloat& value) {
            return py::float_(value.expect_float()).attr("hex")();
        })
        .def_property_readonly("hint", [](const SymFloat& value) -> py::object {
            if (!value.has_hint()) return py::none();
            return py::float_(value.guard_float("", 0));
        })
        .def_property_readonly("constant", [](const SymFloat& value) -> py::object {
            if (auto concrete = value.maybe_as_float()) return py::float_(*concrete);
            return py::none();
        })
        .def("__eq__", [](const SymFloat& left, py::object right) {
            return symfloat_compare(left, right, 0);
        }, py::is_operator())
        .def("__ne__", [](const SymFloat& left, py::object right) {
            return symfloat_compare(left, right, 1);
        }, py::is_operator())
        .def("__lt__", [](const SymFloat& left, py::object right) {
            return symfloat_compare(left, right, 2);
        }, py::is_operator())
        .def("__le__", [](const SymFloat& left, py::object right) {
            return symfloat_compare(left, right, 3);
        }, py::is_operator())
        .def("__gt__", [](const SymFloat& left, py::object right) {
            return symfloat_compare(left, right, 4);
        }, py::is_operator())
        .def("__ge__", [](const SymFloat& left, py::object right) {
            return symfloat_compare(left, right, 5);
        }, py::is_operator())
        .def("__bool__", [](const SymFloat& value) {
            if (auto concrete = value.maybe_as_float()) return *concrete != 0.0;
            return value.guard_float("", 0) != 0.0;
        })
        .def("__int__", [](const SymFloat& value) {
            return value.trunc().expect_int();
        })
        .def("__float__", &SymFloat::expect_float)
        .def("__hash__", [](const SymFloat& value) {
            return static_cast<py::ssize_t>(
                std::hash<double>{}(value.expect_float()));
        })
        .def("__str__", [](const SymFloat& value) { return value_string(value); })
        .def("__repr__", [](const SymFloat& value) { return value_string(value); });

    m.def("sym_float", [](py::object value) -> py::object {
        if (is_symfloat(value)) return value;
        if (is_symint(value)) return py::cast(value.cast<SymInt>().sym_float());
        if (is_symbool(value)) return py::cast(value.cast<SymBool>().toSymFloat());
        if (value && (PyLong_Check(value.ptr()) || PyFloat_Check(value.ptr()))) {
            return py::float_(value);
        }
        return not_implemented();
    });
    m.def("sym_int", [](py::object value) -> py::object {
        if (is_symint(value)) return value;
        if (is_symbool(value)) return py::cast(value.cast<SymBool>().toSymInt());
        if (is_symfloat(value)) return py::cast(value.cast<SymFloat>().sym_int());
        if (value && (PyLong_Check(value.ptr()) || PyFloat_Check(value.ptr()))) {
            return py::module_::import("builtins").attr("int")(value);
        }
        return not_implemented();
    });
    m.def("sym_not", [](py::object value) -> py::object {
        if (is_symbool(value)) return py::cast(value.cast<SymBool>().sym_not());
        if (is_bool_value(value)) return py::bool_(!py::cast<bool>(value));
        return not_implemented();
    });
    m.def("sym_min", [](py::object left, py::object right) -> py::object {
        if (is_float_like(left) || is_float_like(right)) {
            if (is_symbool(left) || is_symbool(right) ||
                !is_number_like(left) || !is_number_like(right)) {
                return not_implemented();
            }
            if (!is_symint(left) && !is_symfloat(left) &&
                !is_symint(right) && !is_symfloat(right)) {
                return py::float_(py::module_::import("builtins")
                                      .attr("min")(left, right));
            }
            return py::cast(as_symfloat(left).min(as_symfloat(right)));
        }
        if (is_int_like(left) && is_int_like(right)) {
            if (!is_symint(left) && !is_symint(right)) {
                return py::module_::import("builtins").attr("min")(left, right);
            }
            return py::cast(as_symint(left).min(as_symint(right)));
        }
        return not_implemented();
    });
    m.def("sym_max", [](py::object left, py::object right) -> py::object {
        if (is_float_like(left) || is_float_like(right)) {
            if (is_symbool(left) || is_symbool(right) ||
                !is_number_like(left) || !is_number_like(right)) {
                return not_implemented();
            }
            if (!is_symint(left) && !is_symfloat(left) &&
                !is_symint(right) && !is_symfloat(right)) {
                return py::float_(py::module_::import("builtins")
                                      .attr("max")(left, right));
            }
            return py::cast(as_symfloat(left).max(as_symfloat(right)));
        }
        if (is_int_like(left) && is_int_like(right)) {
            if (!is_symint(left) && !is_symint(right)) {
                return py::module_::import("builtins").attr("max")(left, right);
            }
            return py::cast(as_symint(left).max(as_symint(right)));
        }
        return not_implemented();
    });
    m.def("sym_ite", &sym_ite);
    m.def("sym_sum", [](py::args args) -> py::object {
        py::list values;
        if (args.size() == 1 &&
            (PyList_Check(args[0].ptr()) || PyTuple_Check(args[0].ptr()))) {
            py::sequence sequence = py::reinterpret_borrow<py::sequence>(args[0]);
            for (py::handle value : sequence) values.append(value);
        } else {
            for (py::handle value : args) values.append(value);
        }
        bool has_symbolic = false;
        SymInt result(0);
        for (py::handle value : values) {
            if (!is_int_like(value)) {
                return py::module_::import("builtins").attr("sum")(values);
            }
            if (is_symint(value)) {
                has_symbolic = true;
                result = result + value.cast<SymInt>();
            } else if (has_symbolic) {
                result = result + as_symint(value);
            }
        }
        if (!has_symbolic) {
            return py::module_::import("builtins").attr("sum")(values);
        }
        return py::cast(result);
    });

    m.def("sym_sqrt", [](py::handle value) {
        return sym_math_scalar<&SymFloat::sqrt>(
            value, "sqrt");
    });
    m.def("sym_cos", [](py::handle value) {
        return sym_math_scalar<&SymFloat::cos>(
            value, "cos");
    });
    m.def("sym_cosh", [](py::handle value) {
        return sym_math_scalar<&SymFloat::cosh>(
            value, "cosh");
    });
    m.def("sym_sin", [](py::handle value) {
        return sym_math_scalar<&SymFloat::sin>(
            value, "sin");
    });
    m.def("sym_sinh", [](py::handle value) {
        return sym_math_scalar<&SymFloat::sinh>(
            value, "sinh");
    });
    m.def("sym_tan", [](py::handle value) {
        return sym_math_scalar<&SymFloat::tan>(
            value, "tan");
    });
    m.def("sym_tanh", [](py::handle value) {
        return sym_math_scalar<&SymFloat::tanh>(
            value, "tanh");
    });
    m.def("sym_asin", [](py::handle value) {
        return sym_math_scalar<&SymFloat::asin>(
            value, "asin");
    });
    m.def("sym_acos", [](py::handle value) {
        return sym_math_scalar<&SymFloat::acos>(
            value, "acos");
    });
    m.def("sym_atan", [](py::handle value) {
        return sym_math_scalar<&SymFloat::atan>(
            value, "atan");
    });
    m.def("sym_log2", [](py::handle value) {
        return sym_math_scalar<&SymFloat::log2>(
            value, "log2");
    });

    py::implicitly_convertible<int64_t, SymInt>();
    py::implicitly_convertible<bool, SymBool>();
    py::implicitly_convertible<double, SymFloat>();
}
