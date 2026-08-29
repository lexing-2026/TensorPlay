#include <pybind11/pybind11.h>

#include "FileCheck.h"

namespace tensorplay {
namespace python {

void init_filecheck(py::module_& m) {
    py::class_<FileCheck>(m, "FileCheck")
        .def(py::init<>())
        .def("check", &FileCheck::check, py::arg("test_string"),
             py::return_value_policy::reference_internal)
        .def("check_not", &FileCheck::check_not, py::arg("test_string"),
             py::return_value_policy::reference_internal)
        .def("check_same", &FileCheck::check_same, py::arg("test_string"),
             py::return_value_policy::reference_internal)
        .def("check_next", &FileCheck::check_next, py::arg("test_string"),
             py::return_value_policy::reference_internal)
        .def("check_count", &FileCheck::check_count, py::arg("test_string"),
             py::arg("count"), py::arg("exactly") = false,
             py::return_value_policy::reference_internal)
        .def("check_dag", &FileCheck::check_dag, py::arg("test_string"),
             py::return_value_policy::reference_internal)
        .def("check_source_highlighted", &FileCheck::check_source_highlighted,
             py::arg("test_string"),
             py::return_value_policy::reference_internal)
        .def("check_regex", &FileCheck::check_regex, py::arg("test_string"),
             py::return_value_policy::reference_internal)
        .def("run", &FileCheck::run, py::arg("test_string"));
}

} // namespace python
} // namespace tensorplay
