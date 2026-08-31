#pragma once

#include "request_callback.h"

namespace tensorplay::distributed::rpc {

class RequestCallbackImpl final : public RequestCallback {
public:
    explicit RequestCallbackImpl(RequestHandler handler);

protected:
    RpcFuturePtr process_message(Message& request) const override;

private:
    RequestHandler handler_;
};

}  // namespace tensorplay::distributed::rpc
