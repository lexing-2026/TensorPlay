#pragma once

#include "message.h"

#include <memory>

namespace tensorplay::distributed::rpc {

class RpcCommandBase {
public:
    virtual ~RpcCommandBase() = default;
    MessagePtr to_message() && {
        return std::move(*this).to_message_impl();
    }
    virtual MessagePtr to_message_impl() && = 0;
};

}  // namespace tensorplay::distributed::rpc
