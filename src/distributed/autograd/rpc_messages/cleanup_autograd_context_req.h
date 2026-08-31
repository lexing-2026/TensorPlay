#pragma once

#include "rpc/message.h"

#include <cstdint>

namespace tensorplay::distributed::autograd {

class CleanupAutogradContextReq final {
public:
    explicit CleanupAutogradContextReq(int64_t context_id);

    rpc::MessagePtr to_message() const;
    static CleanupAutogradContextReq from_message(const rpc::Message& message);

    int64_t context_id() const noexcept;

private:
    int64_t context_id_;
};

}  // namespace tensorplay::distributed::autograd
