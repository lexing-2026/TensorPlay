#include "agent_utils.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace tensorplay::distributed::rpc {

std::string environment_value(const char* primary, const char* secondary) {
    const char* value = std::getenv(primary);
    if (value != nullptr && *value != '\0') {
        return value;
    }
    if (secondary != nullptr) {
        value = std::getenv(secondary);
        if (value != nullptr && *value != '\0') {
            return value;
        }
    }
    return {};
}

void validate_worker_name(const std::string& name) {
    const bool valid_size = !name.empty() && name.size() < 128;
    const bool valid_char = std::all_of(
        name.begin(), name.end(), [](unsigned char value) {
            return std::isalnum(value) || value == '-' || value == '_' ||
                value == ':';
        });
    if (!valid_size || !valid_char) {
        throw std::invalid_argument(
            "worker name must contain only letters, numbers, '-', '_', or ':' "
            "and be shorter than 128 characters");
    }
}

namespace {

std::vector<uint8_t> name_bytes(const std::string& name) {
    return std::vector<uint8_t>(name.begin(), name.end());
}

}  // namespace

std::vector<WorkerInfo> collect_worker_infos(
    const std::shared_ptr<tensorplay::distributed::Store>& store,
    const std::string& current_name,
    worker_id_t current_id,
    worker_id_t world_size,
    const std::chrono::milliseconds& timeout) {
    if (world_size <= 0 || current_id < 0 || current_id >= world_size) {
        throw std::invalid_argument("invalid worker group dimensions");
    }
    validate_worker_name(current_name);
    if (!store) {
        return {WorkerInfo{current_name, current_id}};
    }
    const std::string prefix = "names/";
    store->set(
        prefix + std::to_string(current_id), name_bytes(current_name));
    std::vector<std::string> keys;
    keys.reserve(static_cast<size_t>(world_size));
    for (int64_t id = 0; id < static_cast<int64_t>(world_size); ++id) {
        keys.push_back(prefix + std::to_string(id));
    }
    if (!store->wait(keys, timeout)) {
        throw std::runtime_error("RPC worker name exchange timed out");
    }
    std::vector<WorkerInfo> workers;
    workers.reserve(static_cast<size_t>(world_size));
    for (int64_t id = 0; id < static_cast<int64_t>(world_size); ++id) {
        const auto bytes = store->get(prefix + std::to_string(id));
        const std::string worker_name(bytes.begin(), bytes.end());
        validate_worker_name(worker_name);
        for (const auto& worker : workers) {
            if (worker.name == worker_name) {
                throw std::invalid_argument(
                    "RPC worker names must be unique");
            }
        }
        workers.push_back({worker_name, static_cast<worker_id_t>(id)});
    }
    return workers;
}

std::chrono::milliseconds timeout_to_duration(double seconds) {
    if (!std::isfinite(seconds)) {
        throw std::invalid_argument(
            "timeout must be finite");
    }
    if (seconds < 0.0) {
        return std::chrono::milliseconds(-1);
    }
    if (seconds >
        static_cast<double>(std::numeric_limits<int64_t>::max()) / 1000.0) {
        throw std::overflow_error("timeout exceeds the duration range");
    }
    const auto milliseconds = static_cast<int64_t>(seconds * 1000.0);
    return std::chrono::milliseconds(std::max<int64_t>(0, milliseconds));
}

}  // namespace tensorplay::distributed::rpc
