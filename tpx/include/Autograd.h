#pragma once
#include <memory>
#include <cstdint>
#include <vector>
#include "Macros.h"
#include "Tensor.h"
#include "Node.h"
#include "Edge.h"
#include "AutogradMeta.h"
#include "GradMode.h"
#include "InferenceMode.h"

namespace tensorplay {
namespace tpx {

// The single tensor type exposed by tpx is the p10 Tensor; tpx attaches
// Tensor were merged into one type).
using Tensor = tensorplay::Tensor;

// Free-function accessors over the AutogradMeta extension point. These mirror
namespace impl {

TENSORPLAY_API AutogradMeta* get_autograd_meta(const Tensor& t);
TENSORPLAY_API AutogradMeta* get_or_create_autograd_meta(const Tensor& t);

inline bool requires_grad(const Tensor& t) {
    if (auto* meta = get_autograd_meta(t)) return meta->requires_grad();
    return false;
}

TENSORPLAY_API void set_requires_grad(const Tensor& t, bool requires_grad);

inline Tensor grad(const Tensor& t) { return t.grad(); }

// Gradient metadata is mutable even when the value tensor is passed as a
inline void set_grad(const Tensor& t, const Tensor& grad) {
    if (auto* meta = get_or_create_autograd_meta(t)) meta->set_grad(grad);
}

inline void retain_grad(const Tensor& t) {
    if (auto* meta = get_or_create_autograd_meta(t)) meta->set_retains_grad(true);
}

inline std::shared_ptr<Node> grad_fn(const Tensor& t) {
    if (auto* meta = get_autograd_meta(t)) return meta->grad_fn();
    return nullptr;
}

inline void set_grad_fn(const Tensor& t, std::shared_ptr<Node> grad_fn, uint32_t output_nr = 0) {
    if (auto* meta = get_or_create_autograd_meta(t)) {
        meta->set_grad_fn(std::move(grad_fn));
        meta->set_output_nr(output_nr);
    }
}

inline uint32_t output_nr(const Tensor& t) {
    if (auto* meta = get_autograd_meta(t)) return meta->output_nr();
    return 0;
}

inline std::shared_ptr<Node> grad_accumulator(const Tensor& t) {
    if (auto* meta = get_autograd_meta(t)) return meta->grad_accumulator();
    return nullptr;
}

inline void set_grad_accumulator(const Tensor& t, std::shared_ptr<Node> grad_accumulator) {
    if (auto* meta = get_or_create_autograd_meta(t)) meta->set_grad_accumulator(std::move(grad_accumulator));
}

inline bool is_leaf(const Tensor& t) {
    return grad_fn(t) == nullptr;
}

// true when `t` is a differentiable view whose base chain ends at a leaf
// (its grad_fn chain walks through view nodes down to an AccumulateGrad).
TENSORPLAY_API bool is_view_of_leaf(const Tensor& t);

} // namespace impl

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

// Autograd-aware view/manipulation free functions (formerly tpx::Tensor methods).
// select/slice moved to the generated tpx::ops surface (yaml + derivatives).
TENSORPLAY_API Tensor as_strided(const Tensor& self, const std::vector<int64_t>& size,
                                 const std::vector<int64_t>& stride,
                                 std::optional<int64_t> storage_offset = std::nullopt);
// the generated slice op reuses its backward.
TENSORPLAY_API Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length);
// expand() likewise moved to the generated tpx::ops surface.

// ToCopyBackward node whose backward casts the gradient back to the source
TENSORPLAY_API Tensor to(const Tensor& self, DType dtype, bool non_blocking = false, bool copy = false);
TENSORPLAY_API Tensor to(const Tensor& self, Device device, bool non_blocking = false, bool copy = false);
TENSORPLAY_API Tensor to(const Tensor& self, Device device, DType dtype, bool non_blocking = false, bool copy = false);

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


// GradMode lives in the p10 layer so dispatch code can consult it; it is
// re-exported here for source compatibility.
using GradMode = tensorplay::GradMode;

// InferenceMode likewise lives at the p10 layer; it is re-exported for the
// generated tpx wrappers and the Python bindings.
using InferenceMode = tensorplay::InferenceMode;

} // namespace tpx
} // namespace tensorplay
