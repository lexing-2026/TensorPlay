#pragma once
#include <string>
#include "Macros.h"

namespace tensorplay {

// Capture the current stack trace
P10_API std::string get_stacktrace();

// Print where this process died, and why, on the two signals that mean it
// cannot continue.  For a process that generates code in a worker: a worker
// that dies takes its answer with it, and the only thing that survives is
// whatever it printed.
P10_API void init_crash_handler();

// The same capture as above, with no switch on it, for a caller that cannot
// afford to be told no: a process that is dying has no later to defer to.
P10_API std::string capture_stacktrace_unconditional();

}
