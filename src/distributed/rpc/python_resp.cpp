#include "python_resp.h"

#include <utility>

namespace tensorplay::distributed::rpc {

PythonResp::PythonResp(SerializedPyObj object) : object_(std::move(object)) {}

MessagePtr PythonResp::to_message_impl() && {
    return std::make_shared<Message>(
        std::vector<uint8_t>(object_.payload_.begin(), object_.payload_.end()),
        std::move(object_.tensors_),
        MessageType::PYTHON_RET);
}

std::unique_ptr<PythonResp> PythonResp::from_message(const Message& message) {
    return std::make_unique<PythonResp>(SerializedPyObj(
        std::string(message.payload().begin(), message.payload().end()),
        std::vector<py::object>(message.tensors().begin(), message.tensors().end())));
}

const SerializedPyObj& PythonResp::serialized_object() const noexcept {
    return object_;
}

}  // namespace tensorplay::distributed::rpc
