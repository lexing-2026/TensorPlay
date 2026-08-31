#pragma once

#include <pybind11/pybind11.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorplay::distributed::rpc {

namespace py = pybind11;

using worker_id_t = int16_t;
using local_id_t = int64_t;

struct GloballyUniqueId final {
    static constexpr int kLocalIdBits = 48;

    GloballyUniqueId() = default;
    GloballyUniqueId(worker_id_t created_on, local_id_t local_id);

    bool operator==(const GloballyUniqueId& other) const noexcept;
    bool operator!=(const GloballyUniqueId& other) const noexcept;

    py::tuple to_python() const;
    static GloballyUniqueId from_python(py::handle value);
    std::string to_string() const;

    worker_id_t created_on = 0;
    local_id_t local_id = 0;

    struct Hash {
        size_t operator()(const GloballyUniqueId& value) const noexcept {
            const auto created = static_cast<uint64_t>(
                static_cast<uint16_t>(value.created_on));
            const auto local = static_cast<uint64_t>(value.local_id);
            return static_cast<size_t>((created << kLocalIdBits) ^ local);
        }
    };
};

using RRefId = GloballyUniqueId;
using ForkId = GloballyUniqueId;
using ProfilingId = GloballyUniqueId;

std::ostream& operator<<(std::ostream& stream, const GloballyUniqueId& value);

struct WorkerInfo final {
    std::string name;
    worker_id_t id = 0;

    bool operator==(const WorkerInfo& other) const noexcept {
        return id == other.id && name == other.name;
    }
};

std::ostream& operator<<(std::ostream& stream, const WorkerInfo& value);

struct RpcRetryOptions final {
    int max_retries = 5;
    int64_t retry_duration_ms = 1000;
    double retry_backoff = 1.5;
};

struct RpcBackendOptions {
    double rpc_timeout_seconds = 60.0;
    std::string init_method = "env://";
    int num_worker_threads = 16;
    std::optional<std::vector<std::string>> transports;
    std::optional<std::vector<std::string>> channels;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
        device_maps;
    std::vector<std::string> devices;

    void validate() const;
    void set_device_map(
        const std::string& worker,
        std::unordered_map<std::string, std::string> device_map);
};

struct SerializedPyObj final {
    SerializedPyObj() = default;
    SerializedPyObj(std::string payload, std::vector<py::object> tensors)
        : payload_(std::move(payload)), tensors_(std::move(tensors)) {}

    std::vector<py::object> to_python_values() &&;
    static SerializedPyObj from_python_values(py::iterable values);

    std::string payload_;
    std::vector<py::object> tensors_;
};

}  // namespace tensorplay::distributed::rpc
