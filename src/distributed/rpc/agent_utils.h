#pragma once

#include "types.h"

#include "store/Store.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace tensorplay::distributed::rpc {

std::string environment_value(const char* primary, const char* secondary = nullptr);
void validate_worker_name(const std::string& name);
std::vector<WorkerInfo> collect_worker_infos(
    const std::shared_ptr<tensorplay::distributed::Store>& store,
    const std::string& current_name,
    worker_id_t current_id,
    worker_id_t world_size,
    const std::chrono::milliseconds& timeout);
std::chrono::milliseconds timeout_to_duration(double seconds);

}  // namespace tensorplay::distributed::rpc
