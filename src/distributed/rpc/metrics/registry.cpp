#include "RpcMetricsHandler.h"

namespace tensorplay::distributed::rpc::metrics {

void RpcMetricsHandler::record_call(uint64_t sent_bytes, uint64_t received_bytes) {
    calls_.fetch_add(1, std::memory_order_relaxed);
    sent_bytes_.fetch_add(sent_bytes, std::memory_order_relaxed);
    received_bytes_.fetch_add(received_bytes, std::memory_order_relaxed);
}

void RpcMetricsHandler::record_error() {
    errors_.fetch_add(1, std::memory_order_relaxed);
}

void RpcMetricsHandler::record_gil_wait(uint64_t microseconds) {
    gil_wait_microseconds_.fetch_add(microseconds, std::memory_order_relaxed);
}

std::unordered_map<std::string, std::string> RpcMetricsHandler::snapshot() const {
    return {
        {"calls", std::to_string(calls_.load(std::memory_order_relaxed))},
        {"sent_bytes", std::to_string(sent_bytes_.load(std::memory_order_relaxed))},
        {"received_bytes", std::to_string(received_bytes_.load(std::memory_order_relaxed))},
        {"errors", std::to_string(errors_.load(std::memory_order_relaxed))},
        {"gil_wait_microseconds",
         std::to_string(gil_wait_microseconds_.load(std::memory_order_relaxed))},
    };
}

void RpcMetricsHandler::clear() {
    calls_.store(0, std::memory_order_relaxed);
    sent_bytes_.store(0, std::memory_order_relaxed);
    received_bytes_.store(0, std::memory_order_relaxed);
    errors_.store(0, std::memory_order_relaxed);
    gil_wait_microseconds_.store(0, std::memory_order_relaxed);
}

}  // namespace tensorplay::distributed::rpc::metrics
