#pragma once

#include "future.h"
#include "message.h"

#include <functional>
#include <vector>

namespace tensorplay::distributed::rpc {

class RequestCallback {
public:
    virtual ~RequestCallback() = default;

    RpcFuturePtr operator()(Message& request) const;

protected:
    virtual RpcFuturePtr process_message(Message& request) const = 0;
};

using RequestHandler = std::function<RpcFuturePtr(Message&)>;

}  // namespace tensorplay::distributed::rpc
