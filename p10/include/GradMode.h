#pragma once

#include "Macros.h"

namespace tensorplay {

// Thread-local autograd recording switch. Lives at the p10 layer so
// dispatch code can consult it without depending on tpx; tpx re-exports
// it as tensorplay::tpx::GradMode. The TLS slot itself lives in the
// library: thread-storage objects cannot carry a dll interface on
// Windows, so the header only exposes the accessors.
class P10_API GradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);

private:
    GradMode() = delete;
};

} // namespace tensorplay
