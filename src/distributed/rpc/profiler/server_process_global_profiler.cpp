#include "server_process_global_profiler.h"

namespace tensorplay::distributed::rpc::profiler {

void init_server_profiler() {
    global_profiler().start(false);
}

void shutdown_server_profiler() {
    global_profiler().stop();
}

void record_server_event(
    const std::string& name,
    const std::string& source,
    const std::string& destination,
    uint64_t start_ns,
    uint64_t end_ns,
    bool error) {
    global_profiler().record({name, source, destination, start_ns, end_ns, error});
}

}  // namespace tensorplay::distributed::rpc::profiler
