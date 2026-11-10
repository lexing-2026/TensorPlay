#include "CpuFallback.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

namespace tensorplay {
namespace cpu_fallback {
namespace detail {

// Generated tables (HostFallbackRegistration_<i>.cpp).
void register_host_fallbacks_0(DispatchKey key, int& count);
void register_host_fallbacks_1(DispatchKey key, int& count);
void register_host_fallbacks_2(DispatchKey key, int& count);
void register_host_fallbacks_3(DispatchKey key, int& count);
void register_host_fallbacks_4(DispatchKey key, int& count);
void register_host_fallbacks_5(DispatchKey key, int& count);
void register_host_fallbacks_6(DispatchKey key, int& count);
void register_host_fallbacks_7(DispatchKey key, int& count);

std::atomic<bool> host_call_counting{false};

namespace {

std::mutex& counts_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, int64_t>& counts() {
    static auto* table = new std::unordered_map<std::string, int64_t>;
    return *table;
}

} // namespace

void count_host_call(const char* name) {
    std::lock_guard<std::mutex> lock(counts_mutex());
    ++counts()[name];
}

} // namespace detail

void set_host_call_counting(bool enabled) {
    detail::host_call_counting.store(enabled, std::memory_order_relaxed);
}

std::vector<std::pair<std::string, int64_t>> host_call_counts() {
    std::lock_guard<std::mutex> lock(detail::counts_mutex());
    std::vector<std::pair<std::string, int64_t>> out(detail::counts().begin(),
                                                     detail::counts().end());
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    return out;
}

void reset_host_call_counts() {
    std::lock_guard<std::mutex> lock(detail::counts_mutex());
    detail::counts().clear();
}

int register_host_fallbacks(DispatchKey key) {
    TP_CHECK(is_backend_key(key) && key != DispatchKey::CPU,
             "register_host_fallbacks: ", toString(key), " is not a device backend key");
    int count = 0;
    detail::register_host_fallbacks_0(key, count);
    detail::register_host_fallbacks_1(key, count);
    detail::register_host_fallbacks_2(key, count);
    detail::register_host_fallbacks_3(key, count);
    detail::register_host_fallbacks_4(key, count);
    detail::register_host_fallbacks_5(key, count);
    detail::register_host_fallbacks_6(key, count);
    detail::register_host_fallbacks_7(key, count);
    return count;
}

} // namespace cpu_fallback
} // namespace tensorplay
