#include "request_callback_no_python.h"

#include <stdexcept>

namespace tensorplay::distributed::rpc {

RequestCallbackNoPython::RequestCallbackNoPython(
    std::function<MessagePtr(Message&)> handler)
    : handler_(std::move(handler)) {
    if (!handler_) {
        throw std::invalid_argument("request handler cannot be empty");
    }
}

RpcFuturePtr RequestCallbackNoPython::process_message(Message& request) const {
    auto future = std::make_shared<RpcFuture>();
    try {
        MessagePtr response = handler_(request);
        auto* holder = new MessagePtr(std::move(response));
        future->set_result(py::capsule(
            holder,
            "tensorplay.rpc.Message",
            [](PyObject* capsule) {
                auto* value = static_cast<MessagePtr*>(
                    PyCapsule_GetPointer(capsule, "tensorplay.rpc.Message"));
                delete value;
            }));
    } catch (py::error_already_set& error) {
        py::object exception = py::reinterpret_borrow<py::object>(error.value());
        future->set_exception(std::move(exception));
        error.restore();
        PyErr_Clear();
    } catch (const std::exception& error) {
        py::gil_scoped_acquire gil;
        future->set_exception(py::reinterpret_steal<py::object>(
            PyObject_CallOneArg(PyExc_ValueError, py::str(error.what()).ptr())));
    }
    return future;
}

}  // namespace tensorplay::distributed::rpc
