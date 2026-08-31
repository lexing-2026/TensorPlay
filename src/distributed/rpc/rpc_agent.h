#pragma once

#include "future.h"
#include "message.h"
#include "tensorpipe_utils.h"
#include "types.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorplay::distributed::rpc {

class RpcAgent {
public:
    explicit RpcAgent(WorkerInfo worker, double timeout_seconds);
    virtual ~RpcAgent();

    RpcAgent(const RpcAgent&) = delete;
    RpcAgent& operator=(const RpcAgent&) = delete;

    const WorkerInfo& worker_info() const noexcept;
    static RpcAgent* current_rpc_agent() noexcept;
    static void set_current_rpc_agent(RpcAgent* agent) noexcept;
    std::chrono::milliseconds rpc_timeout() const noexcept;
    void set_rpc_timeout(std::chrono::milliseconds timeout) noexcept;

    virtual RpcFuturePtr send(
        const WorkerInfo& to,
        MessagePtr message,
        double timeout_seconds = -1.0,
        const DeviceMap& device_map = {}) = 0;
    virtual RpcFuturePtr send_with_retries(
        const WorkerInfo& to,
        MessagePtr message,
        RpcRetryOptions options = {}) = 0;
    virtual const WorkerInfo& get_worker_info(const std::string& name) const = 0;
    virtual const WorkerInfo& get_worker_info(worker_id_t id) const = 0;
    virtual std::vector<WorkerInfo> get_worker_infos() const = 0;
    virtual std::unordered_map<std::string, std::string> get_metrics() const = 0;
    virtual std::unordered_map<std::string, std::string> get_debug_info() const;
    virtual void start() = 0;
    virtual void shutdown() = 0;
    virtual void join(bool shutdown = false, double timeout_seconds = 0.0) = 0;
    virtual void sync(double timeout_seconds = -1.0) = 0;

protected:
    WorkerInfo worker_info_;
    std::atomic<int64_t> timeout_ms_;
};

}  // namespace tensorplay::distributed::rpc
