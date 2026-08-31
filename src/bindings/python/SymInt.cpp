#include "python_bindings.h"

#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace {

template <typename T>
std::string value_string(const T& value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
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
        .def("sym_eq", &SymInt::sym_eq)
        .def("sym_ne", &SymInt::sym_ne)
        .def("sym_lt", &SymInt::sym_lt)
        .def("sym_le", &SymInt::sym_le)
        .def("sym_gt", &SymInt::sym_gt)
        .def("sym_ge", &SymInt::sym_ge)
        .def("to_sym_float",
             [](const SymInt& value) { return static_cast<SymFloat>(value); })
        .def("__add__",
             [](const SymInt& left, const SymInt& right) { return left + right; },
             py::is_operator())
        .def("__sub__",
             [](const SymInt& left, const SymInt& right) { return left - right; },
             py::is_operator())
        .def("__mul__",
             [](const SymInt& left, const SymInt& right) { return left * right; },
             py::is_operator())
        .def("__truediv__",
             [](const SymInt& left, int64_t right) {
                 return left / SymInt(right);
             },
             py::is_operator())
        .def("__truediv__",
             [](const SymInt& left, double right) { return left / right; },
             py::is_operator())
        .def("__floordiv__",
             [](const SymInt& left, const SymInt& right) { return left / right; },
             py::is_operator())
        .def("__mod__",
             [](const SymInt& left, const SymInt& right) { return left % right; },
             py::is_operator())
        .def("__neg__", [](const SymInt& value) { return -value; },
             py::is_operator())
        .def("__eq__",
             [](const SymInt& left, const SymInt& right) { return left == right; },
             py::is_operator())
        .def("__ne__",
             [](const SymInt& left, const SymInt& right) { return left != right; },
             py::is_operator())
        .def("__lt__",
             [](const SymInt& left, const SymInt& right) { return left < right; },
             py::is_operator())
        .def("__le__",
             [](const SymInt& left, const SymInt& right) { return left <= right; },
             py::is_operator())
        .def("__gt__",
             [](const SymInt& left, const SymInt& right) { return left > right; },
             py::is_operator())
        .def("__ge__",
             [](const SymInt& left, const SymInt& right) { return left >= right; },
             py::is_operator())
        .def("__int__", &SymInt::expect_int)
        .def("__index__", &SymInt::expect_int)
        .def("__bool__", [](const SymInt& value) {
            return value.sym_ne(SymInt(0)).guard_bool("", 0);
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
        .def("sym_and", &SymBool::sym_and)
        .def("sym_or", &SymBool::sym_or)
        .def("sym_not", &SymBool::sym_not)
        .def("__and__", &SymBool::operator&, py::is_operator())
        .def("__or__", &SymBool::operator|, py::is_operator())
        .def("__invert__", &SymBool::operator~, py::is_operator())
        .def("__eq__", &SymBool::operator==, py::is_operator())
        .def("__ne__", &SymBool::operator!=, py::is_operator())
        .def("__bool__", [](const SymBool& value) {
            return value.guard_bool("", 0);
        })
        .def("__int__", [](const SymBool& value) {
            return static_cast<int64_t>(value.guard_bool("", 0));
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
        .def("sym_eq", &SymFloat::sym_eq)
        .def("sym_ne", &SymFloat::sym_ne)
        .def("sym_lt", &SymFloat::sym_lt)
        .def("sym_le", &SymFloat::sym_le)
        .def("sym_gt", &SymFloat::sym_gt)
        .def("sym_ge", &SymFloat::sym_ge)
        .def("__add__",
             [](const SymFloat& left, const SymFloat& right) { return left + right; },
             py::is_operator())
        .def("__sub__",
             [](const SymFloat& left, const SymFloat& right) { return left - right; },
             py::is_operator())
        .def("__mul__",
             [](const SymFloat& left, const SymFloat& right) { return left * right; },
             py::is_operator())
        .def("__truediv__",
             [](const SymFloat& left, const SymFloat& right) { return left / right; },
             py::is_operator())
        .def("__neg__", [](const SymFloat& value) { return -value; },
             py::is_operator())
        .def("__eq__",
             [](const SymFloat& left, const SymFloat& right) { return left == right; },
             py::is_operator())
        .def("__ne__",
             [](const SymFloat& left, const SymFloat& right) { return left != right; },
             py::is_operator())
        .def("__lt__",
             [](const SymFloat& left, const SymFloat& right) { return left < right; },
             py::is_operator())
        .def("__le__",
             [](const SymFloat& left, const SymFloat& right) { return left <= right; },
             py::is_operator())
        .def("__gt__",
             [](const SymFloat& left, const SymFloat& right) { return left > right; },
             py::is_operator())
        .def("__ge__",
             [](const SymFloat& left, const SymFloat& right) { return left >= right; },
             py::is_operator())
        .def("__float__", &SymFloat::expect_float)
        .def("__str__", [](const SymFloat& value) { return value_string(value); })
        .def("__repr__", [](const SymFloat& value) { return value_string(value); });

    py::implicitly_convertible<int64_t, SymInt>();
    py::implicitly_convertible<bool, SymBool>();
    py::implicitly_convertible<double, SymFloat>();
}
