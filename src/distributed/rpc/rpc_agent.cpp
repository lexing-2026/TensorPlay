#include "rpc_agent.h"

#include <sstream>

namespace tensorplay::distributed::rpc {

namespace {
std::atomic<RpcAgent*> current_agent{nullptr};
}

RpcAgent::RpcAgent(WorkerInfo worker, double timeout_seconds)
    : worker_info_(std::move(worker)),
      timeout_ms_(static_cast<int64_t>(timeout_seconds * 1000.0)) {}

RpcAgent::~RpcAgent() = default;

const WorkerInfo& RpcAgent::worker_info() const noexcept {
    return worker_info_;
}

RpcAgent* RpcAgent::current_rpc_agent() noexcept {
    return current_agent.load(std::memory_order_acquire);
}

void RpcAgent::set_current_rpc_agent(RpcAgent* agent) noexcept {
    current_agent.store(agent, std::memory_order_release);
}

std::chrono::milliseconds RpcAgent::rpc_timeout() const noexcept {
    return std::chrono::milliseconds(timeout_ms_.load());
}

void RpcAgent::set_rpc_timeout(std::chrono::milliseconds timeout) noexcept {
    timeout_ms_.store(timeout.count());
}

std::unordered_map<std::string, std::string> RpcAgent::get_debug_info() const {
    return {
        {"worker_name", worker_info_.name},
        {"worker_id", std::to_string(worker_info_.id)},
        {"rpc_timeout_ms", std::to_string(rpc_timeout().count())},
    };
}

}  // namespace tensorplay::distributed::rpc
