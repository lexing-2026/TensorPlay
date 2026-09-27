#pragma once
#include <string>
#include <vector>
#include "Macros.h"

namespace tensorplay {

// One frame of a captured native stack: where it was, what was running
// there, and which module it came from.  The line number is zero when the
// runtime could not say, which is the normal case for a frame that has
// already returned.
struct StackFrame {
    std::string filename;
    std::string function;
    int lineno{0};
};

// Capture the current stack trace
P10_API std::string get_stacktrace();

// The same capture as get_stacktrace, but as frames rather than as text, so
// that a caller can do something with each one -- record it, compare it, send
// it somewhere -- instead of only printing it.
P10_API std::vector<StackFrame> get_stack_frames();

// Print where this process died, and why, on the two signals that mean it
// cannot continue.  For a process that generates code in a worker: a worker
// that dies takes its answer with it, and the only thing that survives is
// whatever it printed.
P10_API void init_crash_handler();

// The same capture as above, with no switch on it, for a caller that cannot
// afford to be told no: a process that is dying has no later to defer to.
P10_API std::string capture_stacktrace_unconditional();

}
