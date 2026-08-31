#pragma once

#include "rpc/message.h"

namespace tensorplay::distributed::autograd {

class CleanupAutogradContextResp final {
public:
    rpc::MessagePtr to_message() const;
    static CleanupAutogradContextResp from_message(const rpc::Message& message);
};

}  // namespace tensorplay::distributed::autograd
