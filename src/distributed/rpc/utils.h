#pragma once

#include "message.h"

#include <atomic>
#include <cstdint>
#include <string>

namespace tensorplay::distributed::rpc {

uint64_t next_message_id();
std::string message_type_name(MessageType type);
bool is_retryable_error(RPCErrorType type) noexcept;
MessagePtr exception_message(const std::string& message, int64_t id = -1);

}  // namespace tensorplay::distributed::rpc
