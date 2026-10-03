// A thread-local marker for execution inside a lowered graph.
//
// A compiled region runs as one unit under a layout plan: it chose the
// memory order of every buffer it hands an operator, and it reads each
// result with the strides the operator reports for operands laid out that
// way.  An operator called from inside one must therefore not pick another
// order for its result on its own -- the plan already asked for the order
// that pays.  Eager execution makes no such promise, so there an operator
// may repack its operands when that is faster.
//
// The depth counter supports nesting (a region entering another region).
// It lives in one library and is reached through these functions: a
// thread-local defined in the header would be a separate variable in every
// shared library that includes it, and the extension that marks a region
// would count into a copy the kernels never read.

#pragma once

#include <cstdint>

#include "Macros.h"

namespace tensorplay::impl {

P10_API void enter_lowered_graph();
P10_API void exit_lowered_graph();
P10_API bool in_lowered_graph();

struct LoweredGraphScope {
    LoweredGraphScope() { enter_lowered_graph(); }
    ~LoweredGraphScope() { exit_lowered_graph(); }
    LoweredGraphScope(const LoweredGraphScope&) = delete;
    LoweredGraphScope& operator=(const LoweredGraphScope&) = delete;
};

}  // namespace tensorplay::impl
