#include "python_bindings.h"

void init_scalar(nb::module_& m) {
    nb::class_<Scalar>(m, "Scalar")
        .def(nb::init<double>())
        .def(nb::init<int64_t>())
        .def(nb::init<bool>())
        .def("__repr__", &Scalar::toString)
        .def("__float__", [](const Scalar& s) { return s.to<double>(); })
        .def("__int__", [](const Scalar& s) { return s.to<int64_t>(); })
        .def("__bool__", [](const Scalar& s) { return s.to<bool>(); });

    nb::implicitly_convertible<double, Scalar>();
    nb::implicitly_convertible<int64_t, Scalar>();
    nb::implicitly_convertible<bool, Scalar>();
}