#pragma once

#include "request_callback.h"

namespace tensorplay::distributed::rpc {

class RequestCallbackNoPython final : public RequestCallback {
public:
    explicit RequestCallbackNoPython(
        std::function<MessagePtr(Message&)> handler);

protected:
    RpcFuturePtr process_message(Message& request) const override;

private:
    std::function<MessagePtr(Message&)> handler_;
};

}  // namespace tensorplay::distributed::rpc
