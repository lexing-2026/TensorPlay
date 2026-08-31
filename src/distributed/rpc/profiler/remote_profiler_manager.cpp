#include "remote_profiler_manager.h"

namespace tensorplay::distributed::rpc::profiler {

void RemoteProfilerManager::start(bool record_call_stack) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = true;
    record_call_stack_ = record_call_stack;
}

void RemoteProfilerManager::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = false;
}

bool RemoteProfilerManager::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

void RemoteProfilerManager::record(Event event) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enabled_) {
        events_.push_back(std::move(event));
    }
}

std::vector<Event> RemoteProfilerManager::events() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_;
}

void RemoteProfilerManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
}

RemoteProfilerManager& global_profiler() {
    static RemoteProfilerManager profiler;
    return profiler;
}

}  // namespace tensorplay::distributed::rpc::profiler
