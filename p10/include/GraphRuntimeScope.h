// A thread-local marker for execution inside a lowered graph.
//
// A compiled region runs as one unit under a layout plan: the operators it
// calls may hand each other buffers in a non-default memory order and rely
// on the plan keeping that order dense across the whole region.  Eager
// execution makes no such promise -- each call sees whatever layout the
// caller before it happened to leave, and an operator that returns a
// non-default order can force every consumer to copy it back.
//
// The depth counter supports nesting (a region entering another region).

#pragma once

#include <cstdint>

namespace tensorplay::impl {

inline thread_local int64_t lowered_graph_depth = 0;

inline bool in_lowered_graph() { return lowered_graph_depth > 0; }

struct LoweredGraphScope {
    LoweredGraphScope() { ++lowered_graph_depth; }
    ~LoweredGraphScope() { --lowered_graph_depth; }
};

}  // namespace tensorplay::impl
