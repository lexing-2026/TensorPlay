#pragma once

#include "context.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace tensorplay::distributed::rpc {
class RpcAgent;
}

namespace tensorplay::distributed::autograd {

class DistAutogradContainer final {
public:
    static DistAutogradContainer& init(
        rpc::worker_id_t worker_id,
        rpc::RpcAgent* agent = nullptr);
    static DistAutogradContainer& instance();
    static bool is_initialized_global() noexcept;

    ContextPtr new_context();
    ContextPtr get_or_create(int64_t context_id);
    ContextPtr retrieve(int64_t context_id) const;
    void release(int64_t context_id);
    void release_if_present(int64_t context_id);
    void validate(int64_t context_id) const;
    bool is_initialized() const noexcept;

    ContextPtr current() const;
    static bool has_current() noexcept;
    static int64_t current_context_id() noexcept;
    static void set_current_context_id(int64_t context_id);
    static void force_current_context_id(int64_t context_id) noexcept;
    void clear_current_context() const noexcept;

    int64_t new_message_id();
    int64_t max_id() const noexcept;
    rpc::worker_id_t worker_id() const noexcept;
    size_t size() const;

    void set_agent(rpc::RpcAgent* agent);
    rpc::RpcAgent* agent() const noexcept;

private:
    explicit DistAutogradContainer(uint32_t shards);

    static DistAutogradContainer& instance_internal();
    static uint32_t compute_shards();

    mutable std::mutex mutex_;
    std::unordered_map<int64_t, ContextPtr> contexts_;
    std::atomic<int64_t> next_context_id_{0};
    std::atomic<int64_t> next_message_id_{0};
    rpc::worker_id_t worker_id_ = 0;
    int64_t max_id_ = 0;
    rpc::RpcAgent* agent_ = nullptr;
    bool initialized_ = false;
};

class ContextGuard final {
public:
    explicit ContextGuard(int64_t context_id);
    ~ContextGuard();

    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;

private:
    int64_t previous_;
};

}  // namespace tensorplay::distributed::autograd
