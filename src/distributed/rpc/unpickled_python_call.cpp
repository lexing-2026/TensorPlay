#include "unpickled_python_call.h"

#include <utility>

namespace tensorplay::distributed::rpc {

UnpickledPythonCall::UnpickledPythonCall(SerializedPyObj object) {
    auto call = deserialize_python_call(object);
    callable_ = std::move(std::get<0>(call));
    args_ = std::move(std::get<1>(call));
    kwargs_ = std::move(std::get<2>(call));
}

py::object UnpickledPythonCall::callable() const {
    return callable_;
}

py::tuple UnpickledPythonCall::args() const {
    return args_;
}

py::dict UnpickledPythonCall::kwargs() const {
    return kwargs_;
}

bool UnpickledPythonCall::is_async_execution() const noexcept {
    return async_execution_;
}

void UnpickledPythonCall::set_async_execution(bool value) noexcept {
    async_execution_ = value;
}

}  // namespace tensorplay::distributed::rpc
