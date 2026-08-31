#pragma once

#include "remote_profiler_manager.h"

namespace tensorplay::distributed::rpc::profiler {

void init_server_profiler();
void shutdown_server_profiler();
void record_server_event(
    const std::string& name,
    const std::string& source,
    const std::string& destination,
    uint64_t start_ns,
    uint64_t end_ns,
    bool error);

}  // namespace tensorplay::distributed::rpc::profiler
