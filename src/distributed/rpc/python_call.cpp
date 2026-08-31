#include "python_call.h"

#include <utility>

namespace tensorplay::distributed::rpc {

PythonCall::PythonCall(SerializedPyObj object, bool async_execution)
    : object_(std::move(object)), async_execution_(async_execution) {}

MessagePtr PythonCall::to_message_impl() && {
    return std::make_shared<Message>(
        std::vector<uint8_t>(object_.payload_.begin(), object_.payload_.end()),
        std::move(object_.tensors_),
        MessageType::PYTHON_CALL);
}

std::unique_ptr<PythonCall> PythonCall::from_message(const Message& message) {
    return std::make_unique<PythonCall>(
        SerializedPyObj(
            std::string(message.payload().begin(), message.payload().end()),
            std::vector<py::object>(message.tensors().begin(), message.tensors().end())),
        false);
}

const SerializedPyObj& PythonCall::serialized_object() const noexcept {
    return object_;
}

bool PythonCall::is_async_execution() const noexcept {
    return async_execution_;
}

}  // namespace tensorplay::distributed::rpc
