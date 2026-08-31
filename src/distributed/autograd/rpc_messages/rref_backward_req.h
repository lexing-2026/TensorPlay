#pragma once

#include "rpc/message.h"
#include "rpc/types.h"

#include <cstdint>

namespace tensorplay::distributed::autograd {

class RRefBackwardReq final {
public:
    RRefBackwardReq(
        rpc::RRefId rref_id,
        int64_t context_id,
        bool retain_graph);

    rpc::MessagePtr to_message() const;
    static RRefBackwardReq from_message(const rpc::Message& message);

    const rpc::RRefId& rref_id() const noexcept;
    int64_t context_id() const noexcept;
    bool retain_graph() const noexcept;

private:
    rpc::RRefId rref_id_;
    int64_t context_id_ = -1;
    bool retain_graph_ = false;
};

}  // namespace tensorplay::distributed::autograd
