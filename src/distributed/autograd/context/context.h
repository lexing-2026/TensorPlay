#pragma once

#include "Autograd.h"
#include "Node.h"
#include "rpc/future.h"
#include "rpc/types.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tensorplay::distributed::rpc {
class RpcAgent;
}

namespace tensorplay::distributed::autograd {

class SendRpcBackward;
class RecvRpcBackward;

class DistAutogradContext final {
public:
    explicit DistAutogradContext(
        int64_t context_id,
        rpc::RpcAgent* agent = nullptr);

    int64_t context_id() const noexcept;

    void add_send(
        int64_t message_id,
        const std::shared_ptr<SendRpcBackward>& function);
    void add_recv(
        int64_t message_id,
        const std::shared_ptr<RecvRpcBackward>& function);
    std::shared_ptr<SendRpcBackward> send_function(int64_t message_id) const;

    std::unordered_map<
        int64_t,
        std::shared_ptr<SendRpcBackward>> send_functions() const;
    std::unordered_map<
        int64_t,
        std::shared_ptr<RecvRpcBackward>> recv_functions() const;

    void add_known_worker(rpc::worker_id_t worker_id);
    std::unordered_set<rpc::worker_id_t> known_workers() const;

    void accumulate_grad(const tensorplay::Tensor& variable, const tensorplay::Tensor& grad);
    std::vector<std::pair<tensorplay::Tensor, tensorplay::Tensor>> gradients() const;

    void add_outstanding_rpc(const rpc::RpcFuturePtr& future);
    void wait_outstanding_rpcs();
    void clear_outstanding_rpcs();

    rpc::RpcAgent* agent() const noexcept;
    void set_agent(rpc::RpcAgent* agent) noexcept;

    bool retain_graph() const noexcept;
    void set_retain_graph(bool value) noexcept;

private:
    int64_t context_id_;
    rpc::RpcAgent* agent_ = nullptr;
    bool retain_graph_ = false;
    std::unordered_set<rpc::worker_id_t> known_workers_;
    std::unordered_map<int64_t, std::shared_ptr<SendRpcBackward>> send_functions_;
    std::unordered_map<int64_t, std::shared_ptr<RecvRpcBackward>> recv_functions_;
    std::unordered_map<
        const void*,
        std::pair<tensorplay::Tensor, tensorplay::Tensor>> accumulated_grads_;
    std::vector<rpc::RpcFuturePtr> outstanding_rpcs_;
    mutable std::mutex mutex_;
};

using ContextPtr = std::shared_ptr<DistAutogradContext>;

class ThreadLocalDistAutogradContext final {
public:
    explicit ThreadLocalDistAutogradContext(ContextPtr context);
    ~ThreadLocalDistAutogradContext();

    ThreadLocalDistAutogradContext(const ThreadLocalDistAutogradContext&) = delete;
    ThreadLocalDistAutogradContext& operator=(const ThreadLocalDistAutogradContext&) = delete;

    static ContextPtr current();

private:
    ContextPtr previous_;
};

}  // namespace tensorplay::distributed::autograd
