#pragma once

#include "future.h"
#include "rref_context.h"

#include <pybind11/pybind11.h>

#include <atomic>
#include <memory>
#include <string>

namespace tensorplay::distributed::rpc {

class RpcRuntime;

class RpcRRef final : public std::enable_shared_from_this<RpcRRef> {
public:
    RpcRRef(
        RpcRuntime* runtime,
        WorkerInfo owner,
        RRefId rref_id,
        ForkId fork_id,
        RpcFuturePtr creation,
        std::shared_ptr<RRefState> local_state,
        std::weak_ptr<std::atomic<bool>> runtime_lifetime);
    ~RpcRRef();

    RpcRRef(const RpcRRef&) = delete;
    RpcRRef& operator=(const RpcRRef&) = delete;

    const WorkerInfo& owner() const noexcept;
    const RRefId& rref_id() const noexcept;
    const ForkId& fork_id() const noexcept;
    bool is_owner() const;
    bool confirmed_by_owner() const noexcept;
    py::object to_here(double timeout_seconds = -1.0) const;
    py::object local_value() const;
    void backward(int64_t context_id = -1, bool retain_graph = false) const;
    std::shared_ptr<RpcRRef> fork() const;
    std::string repr() const;

private:
    RpcRuntime* runtime_ = nullptr;
    WorkerInfo owner_;
    RRefId rref_id_;
    ForkId fork_id_;
    RpcFuturePtr creation_;
    std::shared_ptr<RRefState> local_state_;
    std::weak_ptr<std::atomic<bool>> runtime_lifetime_;
};

}  // namespace tensorplay::distributed::rpc
