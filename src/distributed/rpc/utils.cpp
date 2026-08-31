#include "utils.h"

#include <array>

namespace tensorplay::distributed::rpc {
namespace {
std::atomic<uint64_t> message_counter{1};
}

uint64_t next_message_id() {
    return message_counter.fetch_add(1, std::memory_order_relaxed);
}

std::string message_type_name(MessageType type) {
    switch (type) {
        case MessageType::PYTHON_CALL: return "PYTHON_CALL";
        case MessageType::PYTHON_RET: return "PYTHON_RET";
        case MessageType::PYTHON_REMOTE_CALL: return "PYTHON_REMOTE_CALL";
        case MessageType::REMOTE_RET: return "REMOTE_RET";
        case MessageType::PYTHON_RREF_FETCH_CALL: return "PYTHON_RREF_FETCH_CALL";
        case MessageType::PYTHON_RREF_FETCH_RET: return "PYTHON_RREF_FETCH_RET";
        case MessageType::RREF_USER_DELETE: return "RREF_USER_DELETE";
        case MessageType::RREF_FORK_REQUEST: return "RREF_FORK_REQUEST";
        case MessageType::RREF_CHILD_ACCEPT: return "RREF_CHILD_ACCEPT";
        case MessageType::RREF_ACK: return "RREF_ACK";
        case MessageType::FORWARD_AUTOGRAD_REQ: return "FORWARD_AUTOGRAD_REQ";
        case MessageType::FORWARD_AUTOGRAD_RESP: return "FORWARD_AUTOGRAD_RESP";
        case MessageType::BACKWARD_AUTOGRAD_REQ: return "BACKWARD_AUTOGRAD_REQ";
        case MessageType::BACKWARD_AUTOGRAD_RESP: return "BACKWARD_AUTOGRAD_RESP";
        case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ: return "CLEANUP_AUTOGRAD_CONTEXT_REQ";
        case MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP: return "CLEANUP_AUTOGRAD_CONTEXT_RESP";
        case MessageType::RUN_WITH_PROFILING_REQ: return "RUN_WITH_PROFILING_REQ";
        case MessageType::RUN_WITH_PROFILING_RESP: return "RUN_WITH_PROFILING_RESP";
        case MessageType::RREF_BACKWARD_REQ: return "RREF_BACKWARD_REQ";
        case MessageType::RREF_BACKWARD_RESP: return "RREF_BACKWARD_RESP";
        case MessageType::PYTHON_GATHER_CALL: return "PYTHON_GATHER_CALL";
        case MessageType::PYTHON_GATHER_RET: return "PYTHON_GATHER_RET";
        case MessageType::EXCEPTION: return "EXCEPTION";
        case MessageType::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

bool is_retryable_error(RPCErrorType type) noexcept {
    return type == RPCErrorType::TIMEOUT;
}

MessagePtr exception_message(const std::string& message, int64_t id) {
    return create_exception_response(message, id);
}

}  // namespace tensorplay::distributed::rpc
