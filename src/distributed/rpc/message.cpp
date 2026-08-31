#include "message.h"

#include <utility>

namespace tensorplay::distributed::rpc {

Message::Message() = default;

Message::Message(
    std::vector<uint8_t> payload,
    std::vector<py::object> tensors,
    MessageType type,
    int64_t id)
    : payload_(std::move(payload)),
      tensors_(std::move(tensors)),
      type_(type),
      id_(id) {}

Message::Message(Message&& other) noexcept
    : payload_(std::move(other.payload_)),
      tensors_(std::move(other.tensors_)),
      type_(other.type_),
      id_(other.id_) {}

Message& Message::operator=(Message&& other) noexcept {
    if (this != &other) {
        payload_ = std::move(other.payload_);
        tensors_ = std::move(other.tensors_);
        type_ = other.type_;
        id_ = other.id_;
    }
    return *this;
}

Message::~Message() {
    if (!tensors_.empty() && Py_IsInitialized()) {
        py::gil_scoped_acquire gil;
        tensors_.clear();
    }
}

std::vector<uint8_t>& Message::payload() noexcept {
    return payload_;
}

const std::vector<uint8_t>& Message::payload() const noexcept {
    return payload_;
}

std::vector<py::object>& Message::tensors() noexcept {
    return tensors_;
}

const std::vector<py::object>& Message::tensors() const noexcept {
    return tensors_;
}

std::vector<uint8_t> Message::move_payload() && {
    return std::move(payload_);
}

std::vector<py::object> Message::move_tensors() && {
    return std::move(tensors_);
}

void Message::reset(
    std::vector<uint8_t> payload,
    std::vector<py::object> tensors,
    MessageType type,
    int64_t id) {
    payload_ = std::move(payload);
    tensors_ = std::move(tensors);
    type_ = type;
    id_ = id;
}

MessageType Message::type() const noexcept {
    return type_;
}

int64_t Message::id() const noexcept {
    return id_;
}

void Message::set_id(int64_t id) noexcept {
    id_ = id;
}

bool Message::is_request() const noexcept {
    return is_request_type(type_);
}

bool Message::is_response() const noexcept {
    return is_response_type(type_);
}

MessagePtr create_exception_response(const std::exception& error, int64_t id) {
    return create_exception_response(error.what(), id);
}

MessagePtr create_exception_response(const std::string& error, int64_t id) {
    std::vector<uint8_t> payload(error.begin(), error.end());
    return std::make_shared<Message>(
        std::move(payload), std::vector<py::object>(), MessageType::EXCEPTION, id);
}

}  // namespace tensorplay::distributed::rpc
