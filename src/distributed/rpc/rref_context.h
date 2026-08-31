#pragma once

#include "types.h"

#include <pybind11/pybind11.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace tensorplay::distributed::rpc {

namespace py = pybind11;

struct RRefState final {
    mutable std::mutex mutex;
    std::condition_variable condition;
    bool ready = false;
    bool has_exception = false;
    size_t references = 1;
    py::object value = py::none();
    py::object error = py::none();

    ~RRefState();
};

class RRefContext final {
public:
    RRefContext() = default;
    ~RRefContext();

    RRefContext(const RRefContext&) = delete;
    RRefContext& operator=(const RRefContext&) = delete;

    std::shared_ptr<RRefState> create(const RRefId& id);
    std::shared_ptr<RRefState> find(const RRefId& id) const;
    void set_value(const RRefId& id, py::object value);
    void set_exception(const RRefId& id, py::object error);
    py::object wait(const RRefId& id, double timeout_seconds) const;
    void retain(const RRefId& id);
    bool release(const RRefId& id);
    void clear();
    size_t size() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<RRefId, std::shared_ptr<RRefState>, GloballyUniqueId::Hash>
        owners_;
};

}  // namespace tensorplay::distributed::rpc
