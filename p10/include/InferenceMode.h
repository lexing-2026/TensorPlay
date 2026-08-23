#pragma once

#include "Macros.h"

namespace tensorplay {

// Thread-local inference-mode switch, mirroring c10::InferenceMode. Lives at
// the p10 layer so generated dispatch code can consult it without depending
// on tpx; tpx re-exports it as tensorplay::tpx::InferenceMode.
//
// While enabled, ops skip autograd recording entirely (outputs of any op get
// requires_grad=False even for inputs that require grad) and in-place
// operations do not bump the version counter. The full torch semantics --
// inference tensors without version counters, rejected later use in
// autograd -- are not implemented yet; this covers the recording/versioning
// behavior, which is what inference_mode() gates in practice.
class P10_API InferenceMode {
public:
    static bool is_enabled() { return enabled_; }
    static void set_enabled(bool enabled) { enabled_ = enabled; }

private:
    // True thread-local storage, same rationale as GradMode: inference mode
    // in one thread must not leak into engine workers or other user threads.
    static thread_local bool enabled_;
};

// RAII helper mirroring c10::InferenceModeGuard for C++ call sites.
struct P10_API InferenceModeGuard {
    bool prev_;
    explicit InferenceModeGuard(bool enabled = true)
        : prev_(InferenceMode::is_enabled()) {
        InferenceMode::set_enabled(enabled);
    }
    ~InferenceModeGuard() { InferenceMode::set_enabled(prev_); }
    InferenceModeGuard(const InferenceModeGuard&) = delete;
    InferenceModeGuard& operator=(const InferenceModeGuard&) = delete;
};

} // namespace tensorplay
