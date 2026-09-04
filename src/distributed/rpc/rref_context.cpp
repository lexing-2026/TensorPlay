#include "rref_context.h"

#include <chrono>
#include <limits>
#include <stdexcept>

namespace tensorplay::distributed::rpc {

RRefState::~RRefState() {
    if (!Py_IsInitialized()) {
        return;
    }
    py::gil_scoped_acquire gil;
    value = py::none();
    error = py::none();
}

RRefContext::~RRefContext() {
    clear();
}

std::shared_ptr<RRefState> RRefContext::create(const RRefId& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto state = std::make_shared<RRefState>();
    auto existing = owners_.find(id);
    if (existing != owners_.end()) {
        return existing->second;
    }
    owners_.emplace(id, state);
    return state;
}

std::shared_ptr<RRefState> RRefContext::create(
    const RRefId& id,
    const ForkId& fork_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto existing = owners_.find(id);
    if (existing != owners_.end()) {
        return existing->second;
    }
    auto state = std::make_shared<RRefState>();
    state->forks.emplace(fork_id);
    owners_.emplace(id, state);
    return state;
}

std::shared_ptr<RRefState> RRefContext::find(const RRefId& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iterator = owners_.find(id);
    return iterator == owners_.end() ? nullptr : iterator->second;
}

void RRefContext::set_value(const RRefId& id, py::object value) {
    auto state = find(id);
    if (!state) {
        throw std::runtime_error("RRef owner entry does not exist");
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->ready) {
            throw std::runtime_error("RRef owner entry can only be completed once");
        }
        state->value = std::move(value);
        state->error = py::none();
        state->has_exception = false;
        state->ready = true;
    }
    state->condition.notify_all();
    std::lock_guard<std::mutex> map_lock(mutex_);
    auto iterator = owners_.find(id);
    if (iterator != owners_.end() && iterator->second == state) {
        std::lock_guard<std::mutex> state_lock(state->mutex);
        if (state->ready && state->references == 0) {
            owners_.erase(iterator);
        }
    }
}

void RRefContext::set_exception(const RRefId& id, py::object error) {
    if (!PyExceptionInstance_Check(error.ptr())) {
        throw py::type_error("RRef owner error must be an exception instance");
    }
    auto state = find(id);
    if (!state) {
        throw std::runtime_error("RRef owner entry does not exist");
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->ready) {
            throw std::runtime_error("RRef owner entry can only be completed once");
        }
        state->error = std::move(error);
        state->value = py::none();
        state->has_exception = true;
        state->ready = true;
    }
    state->condition.notify_all();
    std::lock_guard<std::mutex> map_lock(mutex_);
    auto iterator = owners_.find(id);
    if (iterator != owners_.end() && iterator->second == state) {
        std::lock_guard<std::mutex> state_lock(state->mutex);
        if (state->ready && state->references == 0) {
            owners_.erase(iterator);
        }
    }
}

py::object RRefContext::wait(const RRefId& id, double timeout_seconds) const {
    auto state = find(id);
    if (!state) {
        throw std::runtime_error("RRef owner entry does not exist");
    }
    bool ready = false;
    {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(state->mutex);
        if (timeout_seconds < 0.0) {
            state->condition.wait(lock, [&state]() { return state->ready; });
            ready = true;
        } else {
            ready = state->condition.wait_for(
                lock,
                std::chrono::duration<double>(timeout_seconds),
                [&state]() { return state->ready; });
        }
    }
    if (!ready) {
        PyErr_SetString(PyExc_TimeoutError, "RRef value wait timed out");
        throw py::error_already_set();
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->has_exception) {
        PyErr_SetObject(
            PyExceptionInstance_Class(state->error.ptr()), state->error.ptr());
        throw py::error_already_set();
    }
    return state->value;
}

void RRefContext::retain(const RRefId& id) {
    auto state = find(id);
    if (!state) {
        throw std::runtime_error("RRef owner entry does not exist");
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->references == std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("RRef owner reference count overflow");
    }
    ++state->references;
}

bool RRefContext::retain(const RRefId& id, const ForkId& fork_id) {
    auto state = find(id);
    if (!state) {
        throw std::runtime_error("RRef owner entry does not exist");
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->forks.emplace(fork_id).second) {
        return false;
    }
    if (state->references == std::numeric_limits<size_t>::max()) {
        state->forks.erase(fork_id);
        throw std::overflow_error("RRef owner reference count overflow");
    }
    ++state->references;
    return true;
}

bool RRefContext::release(const RRefId& id) {
    std::lock_guard<std::mutex> map_lock(mutex_);
    auto iterator = owners_.find(id);
    if (iterator == owners_.end()) {
        return false;
    }
    auto state = iterator->second;
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (state->references > 1) {
        --state->references;
        return false;
    }
    if (!state->ready) {
        state->references = 0;
        return false;
    }
    owners_.erase(iterator);
    return true;
}

bool RRefContext::release(const RRefId& id, const ForkId& fork_id) {
    std::lock_guard<std::mutex> map_lock(mutex_);
    auto iterator = owners_.find(id);
    if (iterator == owners_.end()) {
        return false;
    }
    auto state = iterator->second;
    std::lock_guard<std::mutex> state_lock(state->mutex);
    if (!state->forks.erase(fork_id)) {
        return false;
    }
    if (state->references > 0) {
        --state->references;
    }
    if (state->references != 0 || !state->ready) {
        return false;
    }
    owners_.erase(iterator);
    return true;
}

void RRefContext::clear() {
    std::unordered_map<RRefId, std::shared_ptr<RRefState>, GloballyUniqueId::Hash>
        owners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        owners.swap(owners_);
    }
    owners.clear();
}

size_t RRefContext::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return owners_.size();
}

}  // namespace tensorplay::distributed::rpc
