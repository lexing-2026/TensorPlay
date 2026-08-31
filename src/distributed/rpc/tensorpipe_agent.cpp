#include "tensorpipe_agent.h"

namespace tensorplay::distributed::rpc {

TensorPipeAgent::TensorPipeAgent(
    std::shared_ptr<tensorplay::distributed::Store> store,
    std::string name,
    worker_id_t rank,
    worker_id_t world_size,
    TensorPipeRpcBackendOptions options)
    : options_(std::move(options)) {
    configure_backend(options_.transports, options_.channels);
    init(
        name,
        rank,
        world_size,
        options_.num_worker_threads,
        options_.rpc_timeout_seconds,
        options_.init_method,
        std::move(store));
    for (const auto& entry : options_.device_maps) {
        RpcRuntime::set_device_map(entry.first, entry.second);
    }
}

const TensorPipeRpcBackendOptions& TensorPipeAgent::backend_options() const noexcept {
    return options_;
}

void TensorPipeAgent::set_device_map(
    const std::string& worker,
    std::unordered_map<std::string, std::string> device_map) {
    options_.set_device_map(worker, device_map);
    RpcRuntime::set_device_map(worker, std::move(device_map));
}

std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
TensorPipeAgent::device_maps() const {
    return options_.device_maps;
}

}  // namespace tensorplay::distributed::rpc
