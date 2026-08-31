#include "python_remote_call.h"

#include <utility>

namespace tensorplay::distributed::rpc {

PythonRemoteCall::PythonRemoteCall(SerializedPyObj object, RRefId rref_id)
    : object_(std::move(object)), rref_id_(rref_id) {}

MessagePtr PythonRemoteCall::to_message_impl() && {
    auto message = std::make_shared<Message>(
        std::vector<uint8_t>(object_.payload_.begin(), object_.payload_.end()),
        std::move(object_.tensors_),
        MessageType::PYTHON_REMOTE_CALL);
    message->set_id(rref_id_.local_id);
    return message;
}

std::unique_ptr<PythonRemoteCall> PythonRemoteCall::from_message(const Message& message) {
    return std::make_unique<PythonRemoteCall>(
        SerializedPyObj(
            std::string(message.payload().begin(), message.payload().end()),
            std::vector<py::object>(message.tensors().begin(), message.tensors().end())),
        RRefId(0, message.id()));
}

const SerializedPyObj& PythonRemoteCall::serialized_object() const noexcept {
    return object_;
}

const RRefId& PythonRemoteCall::rref_id() const noexcept {
    return rref_id_;
}

}  // namespace tensorplay::distributed::rpc
