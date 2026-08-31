#include "autograd.h"

#include "engine/dist_engine.h"

namespace tensorplay::distributed::autograd {

void backward(
    int64_t context_id,
    const tensorplay::tpx::variable_list& roots,
    bool retain_graph) {
    DistEngine::getInstance().execute(context_id, roots, retain_graph);
}

}  // namespace tensorplay::distributed::autograd
