#pragma once

#include "context/container.h"

namespace tensorplay::distributed::autograd {

void backward(
    int64_t context_id,
    const tensorplay::tpx::variable_list& roots,
    bool retain_graph = false);

}  // namespace tensorplay::distributed::autograd
