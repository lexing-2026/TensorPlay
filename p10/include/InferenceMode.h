#pragma once

#include "GradMode.h"
#include "LocalDispatchKeySet.h"
#include "Macros.h"

namespace tensorplay {

// Thread-local inference-mode switch used by tensor creation and dispatch.
// Like GradMode, the TLS slot lives in the library so the exported surface
// stays free of thread-storage objects.
class P10_API InferenceMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);

private:
    InferenceMode() = delete;
};

// Whether the autograd layer is excluded from dispatch on this thread.  A
// Python dispatch mode handler runs below autograd: operators it re-enters
// execute without recording history, whatever the grad-mode flag says.
inline bool autograd_dispatch_excluded() {
    const auto excluded = impl::tls_local_dispatch_key_set().excluded;
    return excluded.has(DispatchKey::AutogradCPU) ||
           excluded.has(DispatchKey::AutogradCUDA) ||
           excluded.has(DispatchKey::AutogradVulkan) ||
           excluded.has(DispatchKey::AutogradSparse);
}

// RAII helper for C++ call sites.
struct P10_API InferenceModeGuard {
    bool prev_;
    bool prev_grad_;
    explicit InferenceModeGuard(bool enabled = true)
        : prev_(InferenceMode::is_enabled()),
          prev_grad_(GradMode::is_enabled()) {
        InferenceMode::set_enabled(enabled);
        GradMode::set_enabled(!enabled);
    }
    ~InferenceModeGuard() {
        InferenceMode::set_enabled(prev_);
        GradMode::set_enabled(prev_grad_);
    }
    InferenceModeGuard(const InferenceModeGuard&) = delete;
    InferenceModeGuard& operator=(const InferenceModeGuard&) = delete;
};

} // namespace tensorplay
