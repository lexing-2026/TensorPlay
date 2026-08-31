#include "python_functions.h"

#include "python_rpc_handler.h"

#include <stdexcept>

namespace tensorplay::distributed::rpc {

SerializedPyObj serialize_python_object(py::handle object) {
    return PythonRpcHandler::instance().serialize(object);
}

py::object deserialize_python_object(const SerializedPyObj& object) {
    return PythonRpcHandler::instance().deserialize(object);
}

SerializedPyObj serialize_python_call(
    py::handle callable,
    py::tuple args,
    py::dict kwargs) {
    return serialize_python_object(py::make_tuple(callable, args, kwargs));
}

std::tuple<py::object, py::tuple, py::dict> deserialize_python_call(
    const SerializedPyObj& object) {
    py::tuple call = deserialize_python_object(object).cast<py::tuple>();
    if (call.size() != 3) {
        throw std::runtime_error("RPC call payload must contain three values");
    }
    return {
        call[0],
        call[1].cast<py::tuple>(),
        call[2].cast<py::dict>(),
    };
}

}  // namespace tensorplay::distributed::rpc
