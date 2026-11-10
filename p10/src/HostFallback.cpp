#include "CpuFallback.h"

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

} // namespace detail

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
