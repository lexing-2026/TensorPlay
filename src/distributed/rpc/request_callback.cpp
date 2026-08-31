#include "request_callback.h"

namespace tensorplay::distributed::rpc {

RpcFuturePtr RequestCallback::operator()(Message& request) const {
    return process_message(request);
}

}  // namespace tensorplay::distributed::rpc
