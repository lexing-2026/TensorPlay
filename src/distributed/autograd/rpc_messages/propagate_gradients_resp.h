#pragma once

#include "rpc/message.h"

namespace tensorplay::distributed::autograd {

class PropagateGradientsResp final {
public:
    rpc::MessagePtr to_message() const;
    static PropagateGradientsResp from_message(const rpc::Message& message);
};

}  // namespace tensorplay::distributed::autograd
