#include "python_bindings.h"
#include <complex>

void init_scalar(py::module_& m) {
    py::class_<Scalar>(m, "Scalar")
        // One raw-args __init__ instead of typed pybind overloads: a bad
        // argument raises a one-line error instead of an aggregate dump of
        // constructor signatures.
        .def("__init__", [](Scalar& self, py::args args, py::kwargs kwargs) {
            if (kwargs.size() > 0 || args.size() != 1) {
                throw std::invalid_argument(
                    "Scalar() takes exactly one Number argument");
            }
            PyObject* v = args[0].ptr();
            if (PyBool_Check(v)) {
                new (&self) Scalar(v == Py_True);
            } else if (PyLong_Check(v)) {
                new (&self) Scalar(static_cast<int64_t>(PyLong_AsLongLong(v)));
            } else if (PyFloat_Check(v)) {
                new (&self) Scalar(PyFloat_AS_DOUBLE(v));
            } else if (PyComplex_Check(v)) {
                new (&self) Scalar(std::complex<double>(
                    PyComplex_RealAsDouble(v), PyComplex_ImagAsDouble(v)));
            } else if (py::isinstance<Scalar>(args[0])) {
                new (&self) Scalar(args[0].cast<const Scalar&>());
            } else {
                throw std::invalid_argument(
                    std::string("Scalar() expects a Number, not ")
                    + Py_TYPE(v)->tp_name);
            }
        })
        .def("__repr__", &Scalar::toString)
        .def("is_complex", &Scalar::isComplex)
        .def("__float__", [](const Scalar& s) { return s.to<double>(); })
        .def("__complex__", [](const Scalar& s) { return s.to<std::complex<double>>(); })
        // unboxed to a Python int instead of wrapping through int64.
        .def("__int__", [](const Scalar& s) -> py::object {
            if (s.dtype() == DType::UInt64) {
                const uint64_t v = s.to<uint64_t>();
                return py::reinterpret_steal<py::object>(PyLong_FromUnsignedLongLong(v));
            }
            return py::cast(s.to<int64_t>());
        })
        .def("__bool__", [](const Scalar& s) { return s.to<bool>(); });

    // raises "Integers to negative integer powers are not allowed").
    py::implicitly_convertible<int64_t, Scalar>();
    py::implicitly_convertible<double, Scalar>();
    py::implicitly_convertible<bool, Scalar>();
    py::implicitly_convertible<std::complex<float>, Scalar>();
    py::implicitly_convertible<std::complex<double>, Scalar>();
}
