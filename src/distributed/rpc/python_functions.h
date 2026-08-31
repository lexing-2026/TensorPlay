#pragma once

#include "types.h"

#include <pybind11/pybind11.h>

#include <tuple>

namespace tensorplay::distributed::rpc {

namespace py = pybind11;

SerializedPyObj serialize_python_object(py::handle object);
py::object deserialize_python_object(const SerializedPyObj& object);
SerializedPyObj serialize_python_call(
    py::handle callable,
    py::tuple args,
    py::dict kwargs);
std::tuple<py::object, py::tuple, py::dict> deserialize_python_call(
    const SerializedPyObj& object);

}  // namespace tensorplay::distributed::rpc
