#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tensorplay::distributed::rpc::metrics {

class RpcMetricsHandler final {
public:
    void record_call(uint64_t sent_bytes, uint64_t received_bytes);
    void record_error();
    void record_gil_wait(uint64_t microseconds);
    std::unordered_map<std::string, std::string> snapshot() const;
    void clear();

private:
    std::atomic<uint64_t> calls_{0};
    std::atomic<uint64_t> sent_bytes_{0};
    std::atomic<uint64_t> received_bytes_{0};
    std::atomic<uint64_t> errors_{0};
    std::atomic<uint64_t> gil_wait_microseconds_{0};
};

}  // namespace tensorplay::distributed::rpc::metrics
