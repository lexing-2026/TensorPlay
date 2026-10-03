#include "GraphRuntimeScope.h"

namespace tensorplay::impl {
namespace {

thread_local int64_t lowered_graph_depth = 0;

} // namespace

void enter_lowered_graph() { ++lowered_graph_depth; }

void exit_lowered_graph() { --lowered_graph_depth; }

bool in_lowered_graph() { return lowered_graph_depth > 0; }

} // namespace tensorplay::impl
