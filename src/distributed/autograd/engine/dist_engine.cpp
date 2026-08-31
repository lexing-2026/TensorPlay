#include "dist_engine.h"

#include "AccumulateGrad.h"
#include "Autograd.h"
#include "Engine.h"
#include "ManualNodes.h"
#include "context/container.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tensorplay::distributed::autograd {

namespace {

class GradModeGuard final {
public:
    explicit GradModeGuard(bool enabled)
        : previous_(tensorplay::GradMode::is_enabled()) {
        tensorplay::GradMode::set_enabled(enabled);
    }

    ~GradModeGuard() {
        tensorplay::GradMode::set_enabled(previous_);
    }

    GradModeGuard(const GradModeGuard&) = delete;
    GradModeGuard& operator=(const GradModeGuard&) = delete;

private:
    bool previous_;
};

struct LeafState final {
    tensorplay::Tensor variable;
    tensorplay::Tensor before;
};

tensorplay::Tensor copy_gradient(const tensorplay::Tensor& gradient) {
    if (!gradient.defined()) {
        return {};
    }
    return tensorplay::tpx::ops::clone(gradient.detach());
}

void collect_leaves(
    const std::shared_ptr<tensorplay::tpx::Node>& function,
    std::vector<tensorplay::Tensor>& leaves,
    std::unordered_set<tensorplay::tpx::Node*>& seen,
    std::unordered_set<const void*>& seen_variables) {
    if (!function || !seen.insert(function.get()).second) {
        return;
    }
    if (auto* accumulate =
            dynamic_cast<tensorplay::tpx::AccumulateGrad*>(function.get())) {
        const auto variable = accumulate->value_;
        if (variable.defined() &&
            seen_variables.insert(variable.unsafeGetTensorImpl().get()).second) {
            leaves.push_back(variable);
        }
        return;
    }
    for (const auto& edge : function->next_edges()) {
        collect_leaves(edge.function, leaves, seen, seen_variables);
    }
}

std::vector<LeafState> snapshot_leaves(
    const std::vector<std::shared_ptr<tensorplay::tpx::Node>>& roots) {
    std::vector<tensorplay::Tensor> leaves;
    std::unordered_set<tensorplay::tpx::Node*> seen;
    std::unordered_set<const void*> seen_variables;
    for (const auto& root : roots) {
        collect_leaves(root, leaves, seen, seen_variables);
    }
    std::vector<LeafState> result;
    result.reserve(leaves.size());
    for (auto& variable : leaves) {
        result.push_back({variable, copy_gradient(variable.grad())});
    }
    return result;
}

void accumulate_deltas(
    const ContextPtr& context,
    const std::vector<LeafState>& leaves) {
    for (const auto& leaf : leaves) {
        const auto after = leaf.variable.grad();
        if (!after.defined()) {
            continue;
        }
        tensorplay::Tensor delta = after.detach();
        if (leaf.before.defined()) {
            delta = delta - leaf.before;
        }
        if (delta.defined()) {
            context->accumulate_grad(leaf.variable, std::move(delta));
        }
    }
}

tensorplay::Tensor make_root_gradient(const tensorplay::Tensor& root) {
    tensorplay::Tensor gradient(
        std::vector<int64_t>{}, root.dtype(), root.device());
    gradient.fill_(1.0);
    return gradient;
}

}  // namespace

DistEngine& DistEngine::getInstance() {
    static DistEngine* engine = new DistEngine();
    return *engine;
}

void DistEngine::begin(int64_t context_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++active_contexts_[context_id];
}

void DistEngine::end(int64_t context_id) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iterator = active_contexts_.find(context_id);
    if (iterator == active_contexts_.end()) {
        return;
    }
    if (iterator->second <= 1) {
        active_contexts_.erase(iterator);
    } else {
        --iterator->second;
    }
}

void DistEngine::execute(
    int64_t context_id,
    const tensorplay::tpx::variable_list& roots,
    bool retain_graph) {
    auto& container = DistAutogradContainer::instance();
    const auto context = container.retrieve(context_id);
    if (roots.empty()) {
        throw std::invalid_argument(
            "distributed autograd requires at least one root");
    }

    tensorplay::tpx::edge_list root_edges;
    tensorplay::tpx::variable_list gradients;
    std::vector<std::shared_ptr<tensorplay::tpx::Node>> graph_roots;
    root_edges.reserve(roots.size());
    gradients.reserve(roots.size());
    graph_roots.reserve(roots.size());
    for (const auto& root : roots) {
        if (!root.requires_grad()) {
            throw std::invalid_argument(
                "distributed autograd root does not require gradients");
        }
        if (root.numel() != 1) {
            throw std::invalid_argument(
                "distributed autograd roots must be scalar tensors");
        }
        const auto function = tensorplay::tpx::impl::grad_fn(root);
        if (!function) {
            throw std::invalid_argument(
                "distributed autograd root has no gradient function");
        }
        root_edges.emplace_back(
            function, tensorplay::tpx::impl::output_nr(root));
        gradients.push_back(make_root_gradient(root));
        graph_roots.push_back(std::move(function));
    }

    const auto leaves = snapshot_leaves(graph_roots);
    context->set_retain_graph(retain_graph);
    begin(context_id);
    BackwardPassCleanupGuard cleanup(context_id);
    ContextGuard context_guard(context_id);
    GradModeGuard grad_mode(false);
    tensorplay::tpx::Engine::get_default_engine().execute(
        root_edges,
        gradients,
        retain_graph,
        false,
        true,
        {});
    accumulate_deltas(context, leaves);
    context->wait_outstanding_rpcs();
    context->clear_outstanding_rpcs();
}

void DistEngine::execute_send_function(
    const ContextPtr& context,
    const std::shared_ptr<SendRpcBackward>& function,
    bool retain_graph) {
    if (!context || !function) {
        throw std::invalid_argument(
            "distributed autograd send function is invalid");
    }
    const auto gradients = function->grads();
    if (gradients.empty()) {
        throw std::invalid_argument(
            "distributed autograd send function has no gradients");
    }
    for (const auto& gradient : gradients) {
        if (!gradient.defined()) {
            throw std::invalid_argument(
                "distributed autograd send gradient is undefined");
        }
    }

    const auto leaves = snapshot_leaves({function});
    context->set_retain_graph(retain_graph);
    begin(context->context_id());
    BackwardPassCleanupGuard cleanup(context->context_id());
    ContextGuard context_guard(context->context_id());
    GradModeGuard grad_mode(false);

    tensorplay::tpx::edge_list root_edges;
    root_edges.emplace_back(function, 0);
    tensorplay::tpx::Engine::get_default_engine().execute(
        root_edges,
        {tensorplay::Tensor()},
        retain_graph,
        false,
        true,
        {});
    accumulate_deltas(context, leaves);
    context->wait_outstanding_rpcs();
    context->clear_outstanding_rpcs();
}

size_t DistEngine::num_backward_passes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t result = 0;
    for (const auto& entry : active_contexts_) {
        if (entry.second != 0) {
            ++result;
        }
    }
    return result;
}

std::unordered_map<std::string, int64_t> DistEngine::get_debug_info() const {
    return {
        {"num_current_backward_passes",
         static_cast<int64_t>(num_backward_passes())},
        {"num_autograd_contexts",
         static_cast<int64_t>(DistAutogradContainer::instance().size())},
    };
}

}  // namespace tensorplay::distributed::autograd
