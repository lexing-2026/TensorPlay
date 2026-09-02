#include "python_rpc_handler.h"

#include <stdexcept>

namespace tensorplay::distributed::rpc {
namespace {

constexpr const char* kInternalModule = "tensorplay.distributed.rpc.internal";

py::object function_attribute(const py::module_& module, const char* name) {
    py::object function = module.attr(name);
    if (!PyCallable_Check(function.ptr())) {
        throw py::type_error(std::string("RPC internal attribute is not callable: ") + name);
    }
    return function;
}

}  // namespace

PythonRpcHandler& PythonRpcHandler::instance() {
    static PythonRpcHandler* handler = new PythonRpcHandler();
    handler->initialize();
    return *handler;
}

void PythonRpcHandler::initialize() {
    py::gil_scoped_acquire gil;
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return;
    }
    const py::module_ module = py::module_::import(kInternalModule);
    run_function_ = function_attribute(module, "_run_function");
    serialize_ = function_attribute(module, "serialize");
    deserialize_ = function_attribute(module, "deserialize");
    handle_exception_ = function_attribute(module, "_handle_exception");
    initialized_ = true;
}

py::object PythonRpcHandler::run_function(py::handle udf) {
    initialize();
    py::gil_scoped_acquire gil;
    return run_function_(udf);
}

SerializedPyObj PythonRpcHandler::serialize(py::handle value) {
    initialize();
    py::gil_scoped_acquire gil;
    py::object encoded = serialize_(value);
    if (!PyTuple_Check(encoded.ptr()) && !PyList_Check(encoded.ptr())) {
        throw std::runtime_error("RPC serializer returned an invalid result");
    }
    const Py_ssize_t result_size = PySequence_Size(encoded.ptr());
    if (result_size < 0) {
        throw py::error_already_set();
    }
    if (result_size != 2) {
        throw std::runtime_error("RPC serializer returned an invalid result");
    }
    py::object payload = py::reinterpret_steal<py::object>(
        PySequence_GetItem(encoded.ptr(), 0));
    py::object tensor_values = py::reinterpret_steal<py::object>(
        PySequence_GetItem(encoded.ptr(), 1));
    if (!payload || !tensor_values) {
        throw py::error_already_set();
    }
    if (!PyList_Check(tensor_values.ptr()) &&
        !PyTuple_Check(tensor_values.ptr())) {
        throw std::runtime_error("RPC serializer returned invalid tensors");
    }
    const Py_ssize_t tensor_count = PySequence_Size(tensor_values.ptr());
    if (tensor_count < 0) {
        throw py::error_already_set();
    }
    std::vector<py::object> tensors;
    tensors.reserve(static_cast<size_t>(tensor_count));
    for (Py_ssize_t index = 0; index < tensor_count; ++index) {
        py::object tensor = py::reinterpret_steal<py::object>(
            PySequence_GetItem(tensor_values.ptr(), index));
        if (!tensor) {
            throw py::error_already_set();
        }
        tensors.emplace_back(std::move(tensor));
    }
    return SerializedPyObj(payload.cast<std::string>(), std::move(tensors));
}

py::object PythonRpcHandler::deserialize(const SerializedPyObj& value) {
    initialize();
    py::gil_scoped_acquire gil;
    py::list tensors;
    for (const auto& tensor : value.tensors_) {
        tensors.append(tensor);
    }
    return deserialize_(py::bytes(value.payload_), tensors);
}

void PythonRpcHandler::handle_exception(py::handle value) {
    initialize();
    py::gil_scoped_acquire gil;
    handle_exception_(value);
}

bool PythonRpcHandler::is_remote_exception(py::handle value) {
    initialize();
    py::gil_scoped_acquire gil;
    if (value.is_none()) {
        return false;
    }
    const py::object type =
        py::reinterpret_borrow<py::object>(value.get_type());
    const py::object module = type.attr("__module__");
    const py::object name = type.attr("__qualname__");
    return module.cast<std::string>() == kInternalModule &&
        name.cast<std::string>() == "RemoteException";
}

void PythonRpcHandler::cleanup() {
    py::gil_scoped_acquire gil;
    std::lock_guard<std::mutex> lock(mutex_);
    run_function_ = py::none();
    serialize_ = py::none();
    deserialize_ = py::none();
    handle_exception_ = py::none();
    initialized_ = false;
}

}  // namespace tensorplay::distributed::rpc
