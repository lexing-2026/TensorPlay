#pragma once

#include "Macros.h"

namespace tensorplay {

// Thread-local autograd recording switch. Lives at
// the p10 layer so dispatch code can consult it without depending on tpx;
// tpx re-exports it as tensorplay::tpx::GradMode.
class P10_API GradMode {
public:
    static bool is_enabled() { return enabled_; }
    static void set_enabled(bool enabled) { enabled_ = enabled; }

private:
    // True thread-local storage: no_grad() in one thread must not leak into
    static thread_local bool enabled_;
};

} // namespace tensorplay
