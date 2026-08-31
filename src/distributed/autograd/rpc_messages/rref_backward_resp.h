#pragma once

#include "rpc/message.h"

namespace tensorplay::distributed::autograd {

class RRefBackwardResp final {
public:
    rpc::MessagePtr to_message() const;
    static RRefBackwardResp from_message(const rpc::Message& message);
};

}  // namespace tensorplay::distributed::autograd
