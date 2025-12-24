#pragma once
#include <vector>
#include <memory>
#include "Macros.h"
#include "Edge.h"
#include "TPXTensor.h"

namespace tensorplay {
namespace tpx {

using variable_list = std::vector<Tensor>;

class TENSORPLAY_API Node : public std::enable_shared_from_this<Node> {
public:
    virtual ~Node() = default;
    
    virtual variable_list apply(variable_list&& inputs) = 0;
    
    void add_next_edge(Edge edge) {
        next_edges_.push_back(std::move(edge));
    }
    
    void add_next_edge_list(std::vector<Edge> edges) {
        next_edges_.insert(next_edges_.end(), std::make_move_iterator(edges.begin()), std::make_move_iterator(edges.end()));
    }
    
    const std::vector<Edge>& next_edges() const { return next_edges_; }
    
    void release_variables() {
        next_edges_.clear();
    }
    
    uint64_t sequence_nr() const { return sequence_nr_; }
    void set_sequence_nr(uint64_t nr) { sequence_nr_ = nr; }

protected:
    std::vector<Edge> next_edges_;
    uint64_t sequence_nr_ = 0;
};

} // namespace tpx
} // namespace tensorplay
