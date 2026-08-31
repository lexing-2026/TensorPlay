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
    const py::tuple result = serialize_(value).cast<py::tuple>();
    if (result.size() != 2) {
        throw std::runtime_error("RPC serializer returned an invalid result");
    }
    return SerializedPyObj(
        result[0].cast<std::string>(),
        result[1].cast<std::vector<py::object>>());
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
