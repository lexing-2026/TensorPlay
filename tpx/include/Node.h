#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>
#include <typeinfo>
#include <string>
#include "Macros.h"
#include "Edge.h"
#include "Tensor.h"
#include "AnomalyMode.h"

namespace tensorplay {
namespace tpx {

using Tensor = tensorplay::Tensor;
using variable_list = std::vector<Tensor>;
using edge_list = std::vector<Edge>;

// torch InputMetadata analog for BACKWARD input slots.  For custom-function
// nodes the gradients arriving from consumers correspond to the node's
// forward OUTPUTS, so their zero-fill metadata is captured at output-attach
// time (PyNode::attach_outputs) rather than derived from next_edges.
struct OutputSlotMeta {
    std::vector<int64_t> shape;
    DType dtype{DType::Undefined};
    int64_t device_index = -1;
    bool valid = false;
};

// Thread-local monotonically increasing sequence number, assigned at Node
// construction. Mirrors at::sequence_number::get_and_increment().
inline uint64_t get_and_increment_sequence_nr() {
    static thread_local uint64_t counter = 0;
    return counter++;
}

#if defined(__GNUG__) && !defined(TP_NO_CXA_DEMANGLE)
// Demangles a typeid name and keeps only the class component (torch prints
// backward nodes as e.g. "MulBackward0", never namespace-qualified).
inline std::string demangle_node_name(const char* mangled);
#endif

class TENSORPLAY_API Node : public std::enable_shared_from_this<Node> {
public:
    // Hooks mirror torch::autograd::Node: pre-hooks may rewrite the incoming
    // gradients before apply(), post-hooks may rewrite the outputs after.
    using PreHookFn = std::function<variable_list(variable_list&&)>;
    using PostHookFn = std::function<variable_list(const variable_list&, variable_list&&)>;

    Node() : sequence_nr_(get_and_increment_sequence_nr()) { init_anomaly_metadata(); }
    explicit Node(uint64_t sequence_nr) : sequence_nr_(sequence_nr) { init_anomaly_metadata(); }

    // torch parity (ctx.set_materialize_grads): when false the engine hands
    // undefined input gradients through as-is instead of zero-filling them
    // from the edge's recorded InputMetadata.
    bool materialize_grads() const { return materialize_grads_; }
    void set_materialize_grads(bool v) { materialize_grads_ = v; }

    // Torch parity (ADInplaceOrView "view functions"): set by the generated
    // view wrappers / tpx::as_strided so in-place ops can reject mutations
    // of views of leaf variables (check_inplace, VariableTypeUtils.h).
    bool is_view_fn() const { return is_view_fn_; }
    void set_view_fn(bool v) { is_view_fn_ = v; }

    // Backward-input (output-slot) metadata, filled by custom-function nodes
    // at attach time; empty for generated derivative nodes.
    const std::vector<OutputSlotMeta>& output_metas() const { return output_metas_; }
    std::vector<OutputSlotMeta>& output_metas() { return output_metas_; }

    virtual variable_list apply(variable_list&& inputs) = 0;

    // Mirrors at::Node::name(): demangled class name for introspection and
    // anomaly-mode error messages.
    virtual std::string name() const {
#if defined(__GNUG__) && !defined(TP_NO_CXA_DEMANGLE)
        return demangle_node_name(typeid(*this).name());
#else
        return typeid(*this).name();
#endif
    }

    void add_next_edge(Edge edge) {
        update_topological_nr(edge);
        next_edges_.push_back(std::move(edge));
    }

    void add_next_edge_list(std::vector<Edge> edges) {
        for (const Edge& edge : edges) {
            update_topological_nr(edge);
        }
        next_edges_.insert(next_edges_.end(), std::make_move_iterator(edges.begin()), std::make_move_iterator(edges.end()));
    }

    const std::vector<Edge>& next_edges() const { return next_edges_; }

    // Number of gradient inputs this node expects, indexed by input_nr.
    virtual size_t num_inputs() const { return next_edges_.size(); }

    // Virtual so generated/hand-written nodes can also free the forward
    // tensors they saved (SavedVariable::reset_data) when the graph is
    // released, mirroring torch::autograd::Node::release_variables.
    virtual void release_variables() {
        next_edges_.clear();
    }

    uint64_t sequence_nr() const { return sequence_nr_; }
    void set_sequence_nr(uint64_t nr) { sequence_nr_ = nr; }

    uint64_t topological_nr() const { return topological_nr_; }

    // Hook registration (thread-safe enough for graph-construction time; the
    // engine reads them during backward while the graph is immutable).
    void add_pre_hook(PreHookFn hook) { pre_hooks_.push_back(std::move(hook)); }
    void add_post_hook(PostHookFn hook) { post_hooks_.push_back(std::move(hook)); }
    const std::vector<PreHookFn>& pre_hooks() const { return pre_hooks_; }
    const std::vector<PostHookFn>& post_hooks() const { return post_hooks_; }

    // Debug metadata for anomaly mode, created lazily on first access.
    AnomalyMetadata* anomaly_metadata() {
        if (!anomaly_metadata_) {
            anomaly_metadata_ = std::make_unique<AnomalyMetadata>();
        }
        return anomaly_metadata_.get();
    }

private:
    // Mirrors torch::autograd::Node's constructor: when anomaly mode is on,
    // capture where (in which stack) this node was created and record the
    // node being evaluated (if any) as its parent.
    void init_anomaly_metadata() {
        if (AnomalyMode::is_enabled()) {
            anomaly_metadata()->store_stack();
            if (auto parent = get_current_evaluating_node()) {
                anomaly_metadata()->assign_parent(parent);
            }
        }
    }

    void update_topological_nr(const Edge& edge) {
        if (!edge.is_valid()) return;
        auto topo_nr = edge.function->topological_nr();
        if (topological_nr_ <= topo_nr) {
            topological_nr_ = topo_nr + 1;
        }
    }

protected:
    std::vector<Edge> next_edges_;
    uint64_t sequence_nr_ = 0;
    bool materialize_grads_ = true;
    bool is_view_fn_ = false;
    std::vector<OutputSlotMeta> output_metas_;
    uint64_t topological_nr_ = 0;

private:
    std::vector<PreHookFn> pre_hooks_;
    std::vector<PostHookFn> post_hooks_;
    std::unique_ptr<AnomalyMetadata> anomaly_metadata_ = nullptr;
};

#if defined(__GNUG__) && !defined(TP_NO_CXA_DEMANGLE)
#include <cxxabi.h>

inline std::string demangle_node_name(const char* mangled) {
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status != 0 || !demangled) return mangled;
    std::string full(demangled);
    free(demangled);
    // Keep only the unqualified class name.
    auto pos = full.rfind("::");
    return pos == std::string::npos ? full : full.substr(pos + 2);
}
#endif

} // namespace tpx
} // namespace tensorplay