#pragma once
#include <memory>
#include <cstdint>
#include <vector>
#include "Macros.h"
#include "TPXTensor.h"
#include "AutogradMetaInterface.h"
#include "Edge.h"

namespace tensorplay {
namespace tpx {

class Node; // Forward declaration


// Helper to collect next edges for autograd
TENSORPLAY_API std::vector<Edge> collect_next_edges(const Tensor& t);
TENSORPLAY_API std::vector<Edge> collect_next_edges(const std::optional<Tensor>& t);

inline void collect_next_edges_helper(std::vector<Edge>& edges, const Tensor& t) {
    auto next = collect_next_edges(t);
    edges.insert(edges.end(), next.begin(), next.end());
}

inline void collect_next_edges_helper(std::vector<Edge>& edges, const std::optional<Tensor>& t) {
    auto next = collect_next_edges(t);
    edges.insert(edges.end(), next.begin(), next.end());
}

template<typename... Args>
std::vector<Edge> collect_next_edges(const Args&... args) {
    std::vector<Edge> edges;
    (collect_next_edges_helper(edges, args), ...);
    return edges;
}

TENSORPLAY_API void backward(const Tensor& tensor, const Tensor& gradient = {}, bool retain_graph = false, bool create_graph = false);
TENSORPLAY_API void backward(const std::vector<Tensor>& tensors, const std::vector<Tensor>& gradients = {}, bool retain_graph = false, bool create_graph = false);

// Computes and returns the sum of gradients of outputs w.r.t. the inputs.
// If allow_unused is False, specifying inputs that were not used to compute outputs will raise an error.
TENSORPLAY_API std::vector<Tensor> grad(
    const std::vector<Tensor>& outputs, 
    const std::vector<Tensor>& inputs, 
    const std::vector<Tensor>& grad_outputs = {}, 
    bool retain_graph = false, 
    bool create_graph = false, 
    bool allow_unused = false);


class TENSORPLAY_API GradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);
};

class TENSORPLAY_API AutogradMeta : public AutogradMetaInterface {
private:
    bool requires_grad_;
    bool retain_grad_;
    std::shared_ptr<Tensor> grad_;
    std::shared_ptr<Node> grad_fn_;
    std::shared_ptr<Node> grad_accumulator_;
    uint32_t output_nr_;
    
public:
    explicit AutogradMeta(bool requires_grad = false);
    
    bool requires_grad() const override;
    void set_requires_grad(bool requires_grad) override;
    Tensor grad() const override;
    void set_grad(const Tensor& grad) override;
    void accum_grad(const Tensor& grad) override;
    
    bool retain_grad() const override;
    void set_retain_grad(bool retain_grad) override;
    
    void set_grad_fn(std::shared_ptr<Node> grad_fn) override;
    std::shared_ptr<Node> grad_fn() const override;
    
    void set_grad_accumulator(std::shared_ptr<Node> grad_accumulator) override;
    std::shared_ptr<Node> grad_accumulator() const override;
    
    uint32_t output_nr() const override;
    void set_output_nr(uint32_t output_nr) override;
};

} // namespace tpx
} // namespace tensorplay
