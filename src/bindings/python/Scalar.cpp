#include "python_bindings.h"
#include <complex>

void init_scalar(py::module_& m) {
    py::class_<Scalar>(m, "Scalar")
        // ATen alignment: overload order matters. bool first (exact match),
        // then int64_t so Python ints stay integral, double last.
        .def(py::init<bool>())
        .def(py::init<int64_t>())
        .def(py::init<double>())
        .def(py::init<std::complex<float>>())
        .def(py::init<std::complex<double>>())
        .def("__repr__", &Scalar::toString)
        .def("__float__", [](const Scalar& s) { return s.to<double>(); })
        .def("__int__", [](const Scalar& s) { return s.to<int64_t>(); })
        .def("__bool__", [](const Scalar& s) { return s.to<bool>(); });

    // ATen alignment: keep Python ints integral (int64_t) instead of collapsing
    // them to double, so ops like pow can reject int ** negative-int (torch
    // raises "Integers to negative integer powers are not allowed").
    py::implicitly_convertible<int64_t, Scalar>();
    py::implicitly_convertible<double, Scalar>();
    py::implicitly_convertible<bool, Scalar>();
    py::implicitly_convertible<std::complex<float>, Scalar>();
    py::implicitly_convertible<std::complex<double>, Scalar>();
}
