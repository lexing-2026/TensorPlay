#include "Autograd.h"
#include "TensorImpl.h"
#include "AccumulateGrad.h"
#include "Engine.h"
#include "InputBuffer.h"
#include "ManualNodes.h" // For SelectBackward/SliceBackward/AsStridedBackward

namespace tensorplay {
namespace tpx {

namespace impl {

AutogradMeta* get_autograd_meta(const Tensor& t) {
    if (auto* impl = t.unsafeGetTensorImpl().get()) {
        return static_cast<AutogradMeta*>(impl->autograd_meta());
    }
    return nullptr;
}

AutogradMeta* get_or_create_autograd_meta(const Tensor& t) {
    auto impl = t.unsafeGetTensorImpl();
    if (!impl) return nullptr;
    if (auto* meta = impl->autograd_meta()) {
        return static_cast<AutogradMeta*>(meta);
    }
    auto meta = std::make_shared<AutogradMeta>();
    auto* raw = meta.get();
    impl->set_autograd_meta(std::move(meta));
    return raw;
}

void set_requires_grad(const Tensor& t, bool requires_grad) {
    if (auto* meta = get_or_create_autograd_meta(t)) {
        meta->set_requires_grad(requires_grad);
    }
}

} // namespace impl

void AutogradMeta::accum_grad(const tensorplay::Tensor& grad) {
    if (!grad_.defined()) {
        grad_ = grad;
    } else if (!GradMode::is_enabled() && can_accumulate_inplace(grad_)) {
        // In-place accumulation when safe: avoids an allocation per backward
        grad_ += grad;
    } else {
        // Accumulate gradient
        grad_ = grad_ + grad;
    }
}

std::vector<Edge> collect_next_edges(const Tensor& t) {
    std::vector<Edge> edges;
    if (impl::requires_grad(t)) {
        auto fn = impl::grad_fn(t);
        if (fn) {
            edges.emplace_back(fn, impl::output_nr(t));
        } else {
            // Leaf
            auto* meta = impl::get_autograd_meta(t);
            if (meta) {
                // Hold a strong ref locally: the graph edge becomes its owner,
                // while the tensor only keeps a weak cache reference (mirrors
                // c10's weak grad_accumulator_).
                std::shared_ptr<Node> acc = meta->grad_accumulator();
                if (!acc) {
                    acc = std::make_shared<AccumulateGrad>(t);
                    meta->set_grad_accumulator(acc);
                }
                edges.emplace_back(std::move(acc), 0);
            } else {
                edges.emplace_back();
            }
        }
    } else {
        edges.emplace_back();
    }
    return edges;
}

std::vector<Edge> collect_next_edges(const std::optional<Tensor>& t) {
    if (t.has_value()) {
        return collect_next_edges(*t);
    }
    return {Edge()};
}

void backward(const std::vector<Tensor>& tensors, const std::vector<Tensor>& gradients, bool retain_graph, bool create_graph) {
    if (!gradients.empty() && tensors.size() != gradients.size()) {
        TP_THROW(RuntimeError, "Mismatch in tensors and gradients size");
    }

    std::vector<Edge> roots;
    std::vector<Tensor> inputs;
    roots.reserve(tensors.size());
    inputs.reserve(tensors.size());

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        if (!tensor.requires_grad()) {
            TP_THROW(RuntimeError, "Tensor does not require grad and does not have a grad_fn");
        }

        // Prepare gradient
        Tensor grad;
        if (i < gradients.size() && gradients[i].defined()) {
            grad = gradients[i];
        } else {
            if (tensor.numel() != 1) {
                TP_THROW(RuntimeError, "grad can be implicitly created only for scalar outputs");
            }
            // Create scalar tensor on the same device and fill with 1.0
            std::vector<int64_t> shape = {};
            grad = Tensor(shape, tensor.dtype(), tensor.device());
            grad.fill_(1.0);
        }
        inputs.push_back(grad);

        // Prepare root
        if (auto fn = impl::grad_fn(tensor)) {
            roots.emplace_back(fn, impl::output_nr(tensor));
        } else if (tensor.requires_grad()) {
            // Leaf node
            auto* meta = impl::get_autograd_meta(tensor);
            if (meta) {
                std::shared_ptr<Node> acc = meta->grad_accumulator();
                if (!acc) {
                    acc = std::make_shared<AccumulateGrad>(tensor);
                    meta->set_grad_accumulator(acc);
                }
                roots.emplace_back(std::move(acc), 0);
            }
        }
    }

    Engine::get_default_engine().execute(roots, inputs, retain_graph, create_graph,
                                         /*accumulate_grad=*/true, /*outputs=*/{});
}

std::vector<Tensor> grad(
    const std::vector<Tensor>& outputs,
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& grad_outputs,
    bool retain_graph,
    bool create_graph,
    bool allow_unused) {

    if (outputs.empty()) {
        TP_THROW(RuntimeError, "grad requires at least one output tensor");
    }
    if (inputs.empty()) {
        TP_THROW(RuntimeError, "grad requires at least one input tensor");
    }

    // 1. Prepare roots
    std::vector<Edge> roots;
    roots.reserve(outputs.size());
    std::vector<Tensor> root_grads;
    root_grads.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs[i];
        if (!output.requires_grad()) {
            TP_THROW(RuntimeError, "element " + std::to_string(i) + " of tensors does not require grad and does not have a grad_fn");
        }

        // Prepare grad
        Tensor gradient;
        if (i < grad_outputs.size() && grad_outputs[i].defined()) {
            gradient = grad_outputs[i];
        } else {
            if (output.numel() != 1) {
                TP_THROW(RuntimeError, "grad can be implicitly created only for scalar outputs");
            }
            std::vector<float> data = {1.0f};
            gradient = Tensor::tensor(data, output.dtype(), output.device()).reshape({});
        }
        root_grads.push_back(gradient);

        // Prepare edge
        if (auto fn = impl::grad_fn(output)) {
            roots.emplace_back(fn, impl::output_nr(output));
        } else {
            // Leaf
            auto* meta = impl::get_autograd_meta(output);
            if (meta) {
                std::shared_ptr<Node> acc = meta->grad_accumulator();
                if (!acc) {
                    acc = std::make_shared<AccumulateGrad>(output);
                    meta->set_grad_accumulator(acc);
                }
                roots.emplace_back(std::move(acc), 0);
            }
        }
    }

    // 2. Build the output edges (the tensors w.r.t. which we differentiate).
    // The engine captures the input gradients of these edges' functions.
    std::vector<Edge> output_edges;
    output_edges.reserve(inputs.size());

    for (const auto& input : inputs) {
        if (!input.requires_grad()) {
            TP_THROW(RuntimeError, "One of the differentiated Tensors does not require grad");
        }

        Edge edge;
        if (auto fn = impl::grad_fn(input)) {
            edge = Edge(fn, impl::output_nr(input));
        } else {
            // Leaf
            auto* meta = impl::get_autograd_meta(input);
            if (meta) {
                std::shared_ptr<Node> acc = meta->grad_accumulator();
                if (!acc) {
                    acc = std::make_shared<AccumulateGrad>(input);
                    meta->set_grad_accumulator(acc);
                }
                edge = Edge(std::move(acc), 0);
            }
        }

        if (!edge.is_valid()) {
            TP_THROW(RuntimeError, "Could not determine gradient edge for input");
        }
        output_edges.push_back(std::move(edge));
    }

    // 3. Execute
    auto captured = Engine::get_default_engine().execute(roots, root_grads, retain_graph,
                                                         create_graph, /*accumulate_grad=*/false,
                                                         output_edges);

    // 4. Collect results
    std::vector<Tensor> results;
    results.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor res;
        if (i < captured.size() && captured[i].defined()) {
            res = captured[i];
        } else {
            if (!allow_unused) {
                TP_THROW(RuntimeError, "One of the differentiated Tensors was not used in the graph");
            }
        }
        results.push_back(std::move(res));
    }

    return results;
}

void backward(const Tensor& tensor, const Tensor& gradient, bool retain_graph, bool create_graph) {
    std::vector<Tensor> tensors = {tensor};
    std::vector<Tensor> gradients;
    if (gradient.defined()) {
        gradients.push_back(gradient);
    }
    backward(tensors, gradients, retain_graph, create_graph);
}

Tensor as_strided(const Tensor& self, const std::vector<int64_t>& size,
                  const std::vector<int64_t>& stride,
                  std::optional<int64_t> storage_offset) {
    bool requires_grad = self.requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (requires_grad) {
        grad_fn = std::make_shared<AsStridedBackward>(self.shape(), size, stride, storage_offset, self.dtype(), self.device());
        grad_fn->add_next_edge_list(collect_next_edges(self));
    }

    Tensor result = self.as_strided(size, stride, storage_offset);
    if (requires_grad && result.defined()) {
        impl::set_grad_fn(result, grad_fn);
    }
    return result;
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
    bool requires_grad = self.requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (requires_grad) {
        grad_fn = std::make_shared<SelectBackward>(self.shape(), dim, index, self.dtype(), self.device());
        grad_fn->add_next_edge_list(collect_next_edges(self));
    }

    Tensor result = self.select(dim, index);
    if (requires_grad && result.defined()) {
        impl::set_grad_fn(result, grad_fn);
    }
    return result;
}

Tensor slice(const Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
    bool requires_grad = self.requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (requires_grad) {
        grad_fn = std::make_shared<SliceBackward>(self.shape(), dim, start, end, step, self.dtype(), self.device());
        grad_fn->add_next_edge_list(collect_next_edges(self));
    }

    Tensor result = self.slice(dim, start, end, step);
    if (requires_grad && result.defined()) {
        impl::set_grad_fn(result, grad_fn);
    }
    return result;
}

namespace {

// Torch formats shapes in error messages as "[2, 4]".
std::string expand_shape_string(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

} // namespace

Tensor expand(const Tensor& self, const std::vector<int64_t>& size) {
    std::vector<int64_t> target_shape = size;
    std::vector<int64_t> self_shape = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> self_strides = self.strides();

    int64_t ndim = target_shape.size();
    int64_t self_ndim = self_shape.size();

    if (ndim < self_ndim) {
        // Torch wording (its legacy type repr replaced by the dtype name).
        throw std::runtime_error("expand(tensorplay." + std::string(toString(self.dtype())) +
                                 "{" + expand_shape_string(self_shape) + "}, size=[" +
                                 expand_shape_string(target_shape).substr(1) +
                                 "): the number of sizes provided (" + std::to_string(ndim) +
                                 ") must be greater or equal to the number of dimensions in the tensor (" +
                                 std::to_string(self_ndim) + ")");
    }

    std::vector<int64_t> new_strides(ndim);

    // Match dimensions from back
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t target_dim = target_shape[ndim - 1 - i];
        int64_t self_dim_idx = self_ndim - 1 - i;

        if (self_dim_idx >= 0) {
            int64_t self_dim = self_shape[self_dim_idx];
            int64_t self_stride = self_strides[self_dim_idx];

            if (target_dim == -1) {
                target_dim = self_dim;
                target_shape[ndim - 1 - i] = target_dim;
            }

            if (self_dim == 1 && target_dim > 1) {
                new_strides[ndim - 1 - i] = 0;
            } else if (self_dim == target_dim) {
                new_strides[ndim - 1 - i] = self_stride;
            } else {
                // Torch wording, including the double spaces after periods.
                throw std::runtime_error("The expanded size of the tensor (" +
                                         std::to_string(target_dim) + ") must match the existing size (" +
                                         std::to_string(self_dim) + ") at non-singleton dimension " +
                                         std::to_string(ndim - 1 - i) + ".  Target sizes: [" +
                                         expand_shape_string(target_shape).substr(1) + ".  Tensor sizes: [" +
                                         expand_shape_string(self_shape).substr(1));
            }
        } else {
            // New dimension at front
            if (target_dim == -1) throw std::runtime_error("expand: cannot infer size for new dimension");
            new_strides[ndim - 1 - i] = 0; // Broadcast
        }
    }

    return as_strided(self, target_shape, new_strides);
}

} // namespace tpx
} // namespace tensorplay