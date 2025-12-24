#include "python_bindings.h"

void init_size(nb::module_& m) {
    nb::class_<Size>(m, "Size")
        .def(nb::init<std::vector<int64_t>>())
        .def("__len__", &Size::size)
        .def("__getitem__", [](const Size& s, int64_t i) {
            if (i < 0) i += s.size();
            if (i < 0 || i >= (int64_t)s.size()) throw nb::index_error();
            return s[i];
        })
        .def("__iter__", [](const Size& s) {
            return nb::make_iterator(nb::type<Size>(), "iterator", s.begin(), s.end());
        }, nb::keep_alive<0, 1>())
        .def("__repr__", &Size::toString)
        .def("__str__", &Size::toString)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("__eq__", [](const Size& s, const nb::tuple& other) {
            if (s.size() != other.size()) return false;
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] != nb::cast<int64_t>(other[i])) return false;
            }
            return true;
        })
        .def("__eq__", [](const Size& s, const nb::list& other) {
             if (s.size() != other.size()) return false;
             for (size_t i = 0; i < s.size(); ++i) {
                 if (s[i] != nb::cast<int64_t>(other[i])) return false;
             }
             return true;
        });
}