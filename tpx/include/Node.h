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

namespace tensorplay {
namespace tpx {

using Tensor = tensorplay::Tensor;
using variable_list = std::vector<Tensor>;
using edge_list = std::vector<Edge>;

// Thread-local monotonically increasing sequence number, assigned at Node
// construction. Mirrors at::sequence_number::get_and_increment().
inline uint64_t get_and_increment_sequence_nr() {
    static thread_local uint64_t counter = 0;
    return counter++;
}

class TENSORPLAY_API Node : public std::enable_shared_from_this<Node> {
public:
    // Hooks mirror torch::autograd::Node: pre-hooks may rewrite the incoming
    // gradients before apply(), post-hooks may rewrite the outputs after.
    using PreHookFn = std::function<variable_list(variable_list&&)>;
    using PostHookFn = std::function<variable_list(const variable_list&, variable_list&&)>;

    Node() : sequence_nr_(get_and_increment_sequence_nr()) {}
    explicit Node(uint64_t sequence_nr) : sequence_nr_(sequence_nr) {}
    virtual ~Node() = default;

    virtual variable_list apply(variable_list&& inputs) = 0;

    // Mirrors at::Node::name(): RTTI-based class name for introspection.
    virtual std::string name() const { return typeid(*this).name(); }

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

    void release_variables() {
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

private:
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
    uint64_t topological_nr_ = 0;

private:
    std::vector<PreHookFn> pre_hooks_;
    std::vector<PostHookFn> post_hooks_;
};

} // namespace tpx
} // namespace tensorplay