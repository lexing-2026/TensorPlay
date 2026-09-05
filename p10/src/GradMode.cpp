#include "GradMode.h"

namespace tensorplay {

namespace {
// The slot stays translation-unit local to the library: thread-storage
// objects cannot carry a dll interface on Windows, so callers go through
// the exported accessors instead of touching the variable directly.
thread_local bool tp_grad_mode_enabled = true;
} // namespace

bool GradMode::is_enabled() { return tp_grad_mode_enabled; }
void GradMode::set_enabled(bool enabled) { tp_grad_mode_enabled = enabled; }

} // namespace tensorplay
