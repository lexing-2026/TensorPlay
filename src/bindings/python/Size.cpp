#include "python_bindings.h"

void init_size(py::module_& m) {
    py::class_<Size>(m, "Size")
        .def(py::init<std::vector<int64_t>>())
        .def("__len__", &Size::size)
        .def("__getitem__", [](const Size& s, py::object idx) -> py::object {
            try {
                int64_t i = py::cast<int64_t>(idx);
                if (i < 0) i += s.size();
                if (i < 0 || i >= (int64_t)s.size()) throw py::index_error();
                return py::cast(s[i]);
            } catch (py::cast_error&) {
            } catch (const py::reference_cast_error&) {
            }
            // ... and slices (shape[:-1], shape[-2:], shape[::2]).
            if (py::isinstance<py::slice>(idx)) {
                auto sl = py::cast<py::slice>(idx);
                ssize_t start = 0, stop = 0, step = 0, slicelen = 0;
                if (!sl.compute(static_cast<ssize_t>(s.size()), &start, &stop,
                                &step, &slicelen))
                    throw py::error_already_set();
                std::vector<int64_t> out;
                out.reserve(static_cast<size_t>(slicelen));
                for (ssize_t i = 0; i < slicelen; ++i)
                    out.push_back(s[static_cast<int64_t>(start + i * step)]);
                return py::cast(Size(out));
            }
            throw py::type_error("Size indices must be integers or slices");
        })
        .def("__iter__", [](const Size& s) {
            return py::make_iterator(s.begin(), s.end());
        }, py::keep_alive<0, 1>())
        .def("__repr__", &Size::toString)
        .def("__str__", &Size::toString)
        .def("__add__", [](const Size& s, const py::sequence& other) {
            std::vector<int64_t> out(s.begin(), s.end());
            for (auto item : other) out.push_back(py::cast<int64_t>(item));
            return Size(out);
        })
        .def("__radd__", [](const Size& s, const py::sequence& other) {
            std::vector<int64_t> out;
            for (auto item : other) out.push_back(py::cast<int64_t>(item));
            out.insert(out.end(), s.begin(), s.end());
            return Size(out);
        })
        .def("__hash__", [](const Size& s) {
            // match tuple hashing semantics well enough for dict keys
            py::tuple t(s.size());
            for (size_t i = 0; i < s.size(); ++i) t[i] = py::cast(s[i]);
            return py::hash(t);
        })
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