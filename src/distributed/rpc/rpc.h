#pragma once

#include "rpc_runtime.h"

namespace tensorplay::distributed::rpc {

inline RpcRuntime& rpc_agent() {
    return global_rpc_runtime();
}

}  // namespace tensorplay::distributed::rpc
