#include "types.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tensorplay::distributed::rpc {
GloballyUniqueId::GloballyUniqueId(worker_id_t created_on_value, local_id_t local_id_value)
    : created_on(created_on_value), local_id(local_id_value) {
    if (local_id_value < 0) {
        throw std::invalid_argument("global id local id must be non-negative");
    }
}

bool GloballyUniqueId::operator==(const GloballyUniqueId& other) const noexcept {
    return created_on == other.created_on && local_id == other.local_id;
}

bool GloballyUniqueId::operator!=(const GloballyUniqueId& other) const noexcept {
    return !(*this == other);
}

py::tuple GloballyUniqueId::to_python() const {
    return py::make_tuple(static_cast<int64_t>(created_on), local_id);
}

GloballyUniqueId GloballyUniqueId::from_python(py::handle value) {
    if (!PyTuple_Check(value.ptr()) || PyTuple_GET_SIZE(value.ptr()) != 2) {
        throw py::type_error("global id must be a pair of integers");
    }
    const auto created = py::cast<int64_t>(PyTuple_GET_ITEM(value.ptr(), 0));
    const auto local = py::cast<int64_t>(PyTuple_GET_ITEM(value.ptr(), 1));
    if (created < std::numeric_limits<worker_id_t>::min() ||
        created > std::numeric_limits<worker_id_t>::max()) {
        throw py::value_error("global id worker id is out of range");
    }
    if (local < 0) {
        throw py::value_error("global id local id must be non-negative");
    }
    return GloballyUniqueId(static_cast<worker_id_t>(created), local);
}

std::string GloballyUniqueId::to_string() const {
    std::ostringstream stream;
    stream << "GloballyUniqueId(created_on=" << created_on
           << ", local_id=" << local_id << ')';
    return stream.str();
}

std::ostream& operator<<(std::ostream& stream, const GloballyUniqueId& value) {
    return stream << value.to_string();
}

std::ostream& operator<<(std::ostream& stream, const WorkerInfo& value) {
    return stream << "WorkerInfo(name='" << value.name << "', id=" << value.id
                  << ')';
}

void RpcBackendOptions::validate() const {
    if (!std::isfinite(rpc_timeout_seconds) || rpc_timeout_seconds < 0) {
        throw std::invalid_argument("rpc timeout must be non-negative");
    }
    if (num_worker_threads <= 0) {
        throw std::invalid_argument("num worker threads must be positive");
    }
}

void RpcBackendOptions::set_device_map(
    const std::string& worker,
    std::unordered_map<std::string, std::string> device_map) {
    auto& target = device_maps[worker];
    for (auto& entry : device_map) {
        if (entry.first.empty() || entry.second.empty()) {
            throw std::invalid_argument("device map contains an empty device");
        }
        auto existing = target.find(entry.first);
        if (existing != target.end() && existing->second != entry.second) {
            throw std::invalid_argument("source device already has a mapping");
        }
        for (const auto& current : target) {
            if (current.first != entry.first && current.second == entry.second) {
                throw std::invalid_argument("device map target is already mapped");
            }
        }
        target[entry.first] = std::move(entry.second);
    }
}

std::vector<py::object> SerializedPyObj::to_python_values() && {
    std::vector<py::object> values;
    values.reserve(tensors_.size() + 1);
    for (auto& tensor : tensors_) {
        values.emplace_back(std::move(tensor));
    }
    values.emplace_back(py::bytes(payload_));
    return values;
}

SerializedPyObj SerializedPyObj::from_python_values(py::iterable values) {
    std::vector<py::object> items;
    for (py::handle value : values) {
        items.emplace_back(py::reinterpret_borrow<py::object>(value));
    }
    if (items.empty() || !PyBytes_Check(items.back().ptr())) {
        throw py::type_error("serialized object values must end with bytes");
    }
    std::string payload = items.back().cast<std::string>();
    items.pop_back();
    return SerializedPyObj(std::move(payload), std::move(items));
}

}  // namespace tensorplay::distributed::rpc
