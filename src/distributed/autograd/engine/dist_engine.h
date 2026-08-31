#pragma once

#include "context/context.h"
#include "functions/sendrpc_backward.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace tensorplay::distributed::autograd {

class DistEngine final {
public:
    static DistEngine& getInstance();

    void execute(
        int64_t context_id,
        const tensorplay::tpx::variable_list& roots,
        bool retain_graph);

    void execute_send_function(
        const ContextPtr& context,
        const std::shared_ptr<SendRpcBackward>& function,
        bool retain_graph);

    size_t num_backward_passes() const;
    std::unordered_map<std::string, int64_t> get_debug_info() const;

    DistEngine(const DistEngine&) = delete;
    DistEngine& operator=(const DistEngine&) = delete;

private:
    DistEngine() = default;

    friend class BackwardPassCleanupGuard;

    void begin(int64_t context_id);
    void end(int64_t context_id) noexcept;

    mutable std::mutex mutex_;
    std::unordered_map<int64_t, size_t> active_contexts_;
};

class BackwardPassCleanupGuard final {
public:
    explicit BackwardPassCleanupGuard(int64_t context_id)
        : context_id_(context_id) {}

    ~BackwardPassCleanupGuard() {
        DistEngine::getInstance().end(context_id_);
    }

    BackwardPassCleanupGuard(const BackwardPassCleanupGuard&) = delete;
    BackwardPassCleanupGuard& operator=(const BackwardPassCleanupGuard&) = delete;

private:
    int64_t context_id_;
};

}  // namespace tensorplay::distributed::autograd
