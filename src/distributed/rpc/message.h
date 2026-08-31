#pragma once

#include "types.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace tensorplay::distributed::rpc {

enum class RPCErrorType : uint8_t {
    UNKNOWN_ERROR = 0,
    TIMEOUT = 1,
    INTENTIONAL_FAILURE = 2,
};

enum class MessageTypeFlags : uint16_t {
    NONE = 0x0000,
    REQUEST_TYPE = 0x0100,
    RESPONSE_TYPE = 0x0200,
};

enum class MessageType : uint16_t {
    PYTHON_CALL = 0x0102,
    PYTHON_RET = 0x0203,
    PYTHON_REMOTE_CALL = 0x0105,
    REMOTE_RET = 0x0206,
    PYTHON_RREF_FETCH_CALL = 0x0108,
    PYTHON_RREF_FETCH_RET = 0x020a,
    RREF_USER_DELETE = 0x010b,
    RREF_FORK_REQUEST = 0x010c,
    RREF_CHILD_ACCEPT = 0x010d,
    RREF_ACK = 0x020e,
    FORWARD_AUTOGRAD_REQ = 0x010f,
    FORWARD_AUTOGRAD_RESP = 0x0210,
    BACKWARD_AUTOGRAD_REQ = 0x0111,
    BACKWARD_AUTOGRAD_RESP = 0x0212,
    CLEANUP_AUTOGRAD_CONTEXT_REQ = 0x0113,
    CLEANUP_AUTOGRAD_CONTEXT_RESP = 0x0214,
    RUN_WITH_PROFILING_REQ = 0x0115,
    RUN_WITH_PROFILING_RESP = 0x0216,
    RREF_BACKWARD_REQ = 0x0117,
    RREF_BACKWARD_RESP = 0x0218,
    PYTHON_GATHER_CALL = 0x0119,
    PYTHON_GATHER_RET = 0x021a,
    EXCEPTION = 0x0237,
    UNKNOWN = 0x003c,
};

constexpr uint16_t message_type_value(MessageType type) noexcept {
    return static_cast<uint16_t>(type);
}

constexpr bool is_request_type(MessageType type) noexcept {
    return (message_type_value(type) &
            static_cast<uint16_t>(MessageTypeFlags::REQUEST_TYPE)) != 0;
}

constexpr bool is_response_type(MessageType type) noexcept {
    return (message_type_value(type) &
            static_cast<uint16_t>(MessageTypeFlags::RESPONSE_TYPE)) != 0;
}

class Message final {
public:
    Message();
    Message(
        std::vector<uint8_t> payload,
        std::vector<py::object> tensors,
        MessageType type,
        int64_t id = -1);
    ~Message();

    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;
    Message(Message&& other) noexcept;
    Message& operator=(Message&& other) noexcept;

    std::vector<uint8_t>& payload() noexcept;
    const std::vector<uint8_t>& payload() const noexcept;
    std::vector<py::object>& tensors() noexcept;
    const std::vector<py::object>& tensors() const noexcept;
    std::vector<uint8_t> move_payload() &&;
    std::vector<py::object> move_tensors() &&;
    void reset(
        std::vector<uint8_t> payload,
        std::vector<py::object> tensors,
        MessageType type,
        int64_t id = -1);
    MessageType type() const noexcept;
    int64_t id() const noexcept;
    void set_id(int64_t id) noexcept;
    bool is_request() const noexcept;
    bool is_response() const noexcept;

private:
    std::vector<uint8_t> payload_;
    std::vector<py::object> tensors_;
    MessageType type_ = MessageType::UNKNOWN;
    int64_t id_ = -1;
};

using MessagePtr = std::shared_ptr<Message>;

MessagePtr create_exception_response(const std::exception& error, int64_t id);
MessagePtr create_exception_response(const std::string& error, int64_t id);

}  // namespace tensorplay::distributed::rpc
