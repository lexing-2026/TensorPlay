#pragma once

#include "GradMode.h"
#include "Macros.h"

namespace tensorplay {

// Thread-local inference-mode switch used by tensor creation and dispatch.
class P10_API InferenceMode {
public:
    static bool is_enabled() { return enabled_; }
    static void set_enabled(bool enabled) { enabled_ = enabled; }

private:
    // True thread-local storage, same rationale as GradMode: inference mode
    // in one thread must not leak into engine workers or other user threads.
    static thread_local bool enabled_;
};

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
