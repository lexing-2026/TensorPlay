#include "python_bindings.h"

void init_size(py::module_& m) {
    py::class_<Size>(m, "Size")
        .def(py::init<std::vector<int64_t>>())
        .def("__len__", &Size::size)
        .def("__getitem__", [](const Size& s, int64_t i) {
            if (i < 0) i += s.size();
            if (i < 0 || i >= (int64_t)s.size()) throw py::index_error();
            return s[i];
        })
        .def("__iter__", [](const Size& s) {
            return py::make_iterator(s.begin(), s.end());
        }, py::keep_alive<0, 1>())
        .def("__repr__", &Size::toString)
        .def("__str__", &Size::toString)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__eq__", [](const Size& s, const py::tuple& other) {
            if (s.size() != other.size()) return false;
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] != py::cast<int64_t>(other[i])) return false;
            }
            return true;
        })
        .def("__eq__", [](const Size& s, const py::list& other) {
             if (s.size() != other.size()) return false;
             for (size_t i = 0; i < s.size(); ++i) {
                 if (s[i] != py::cast<int64_t>(other[i])) return false;
             }
             return true;
        });
}