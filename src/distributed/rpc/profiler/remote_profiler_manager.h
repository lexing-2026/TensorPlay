#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorplay::distributed::rpc::profiler {

struct Event final {
    std::string name;
    std::string source;
    std::string destination;
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    bool error = false;
};

class RemoteProfilerManager final {
public:
    void start(bool record_call_stack);
    void stop();
    bool enabled() const;
    void record(Event event);
    std::vector<Event> events() const;
    void clear();

private:
    mutable std::mutex mutex_;
    bool enabled_ = false;
    bool record_call_stack_ = false;
    std::vector<Event> events_;
};

RemoteProfilerManager& global_profiler();

}  // namespace tensorplay::distributed::rpc::profiler
