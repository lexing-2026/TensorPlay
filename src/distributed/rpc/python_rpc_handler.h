#pragma once

#include "types.h"

#include <pybind11/pybind11.h>

#include <mutex>

namespace tensorplay::distributed::rpc {

namespace py = pybind11;

class PythonRpcHandler final {
public:
    static PythonRpcHandler& instance();

    PythonRpcHandler(const PythonRpcHandler&) = delete;
    PythonRpcHandler& operator=(const PythonRpcHandler&) = delete;
    PythonRpcHandler(PythonRpcHandler&&) = delete;
    PythonRpcHandler& operator=(PythonRpcHandler&&) = delete;

    py::object run_function(py::handle udf);
    SerializedPyObj serialize(py::handle value);
    py::object deserialize(const SerializedPyObj& value);
    void handle_exception(py::handle value);
    bool is_remote_exception(py::handle value);
    void cleanup();

private:
    PythonRpcHandler() = default;

    void initialize();

    std::mutex mutex_;
    bool initialized_ = false;
    py::object run_function_;
    py::object serialize_;
    py::object deserialize_;
    py::object handle_exception_;
};

}  // namespace tensorplay::distributed::rpc
