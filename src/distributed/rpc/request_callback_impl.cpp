#include "request_callback_impl.h"

#include <stdexcept>

namespace tensorplay::distributed::rpc {

RequestCallbackImpl::RequestCallbackImpl(RequestHandler handler)
    : handler_(std::move(handler)) {
    if (!handler_) {
        throw std::invalid_argument("request handler cannot be empty");
    }
}

RpcFuturePtr RequestCallbackImpl::process_message(Message& request) const {
    return handler_(request);
}

}  // namespace tensorplay::distributed::rpc
