#pragma once

#include "rpc_runtime.h"
#include "tensorpipe_backend.h"

#include <optional>

namespace tensorplay::distributed::rpc {

struct TensorPipeRpcBackendOptions final : RpcBackendOptions {
    TensorPipeRpcBackendOptions() = default;

    TensorPipeRpcBackendOptions(
        int worker_threads,
        std::optional<std::vector<std::string>> transport_names,
        std::optional<std::vector<std::string>> channel_names,
        double timeout_seconds,
        std::string method)
        : RpcBackendOptions{
              timeout_seconds,
              std::move(method),
              worker_threads,
              std::move(transport_names),
              std::move(channel_names),
              {},
              {}} {
        validate();
    }
};

struct NetworkSourceInfo final {
    worker_id_t source_rank = 0;
    std::vector<uint8_t> address;
};

struct AggregatedNetworkData final {
    uint64_t calls = 0;
    uint64_t sent_bytes = 0;
    uint64_t received_bytes = 0;
    uint64_t errors = 0;
};

class TensorPipeAgent : public RpcRuntime {
public:
    TensorPipeAgent(
        std::shared_ptr<tensorplay::distributed::Store> store,
        std::string name,
        worker_id_t rank,
        worker_id_t world_size,
        TensorPipeRpcBackendOptions options);

    const TensorPipeRpcBackendOptions& backend_options() const noexcept;
    void set_device_map(
        const std::string& worker,
        std::unordered_map<std::string, std::string> device_map);
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
    device_maps() const;

private:
    TensorPipeRpcBackendOptions options_;
};

}  // namespace tensorplay::distributed::rpc
