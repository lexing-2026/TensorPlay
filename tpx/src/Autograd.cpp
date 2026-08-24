#include "Autograd.h"
#include "TensorImpl.h"
#include "AccumulateGrad.h"
#include "Engine.h"
#include "InputBuffer.h"
#include "ManualNodes.h" // For AsStridedBackward
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "tensorplay/ops/AutogradNodesGenerated.h"

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

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    // torch's narrow is a slice of `length` elements starting at `start`;
    // routing through the generated slice op carries the gradient via
    // SliceBackward.
    if (length < 0) {
        TP_THROW(RuntimeError, "narrow(): length cannot be negative but got ", length);
    }
    return tensorplay::tpx::ops::slice(self, dim, start, start + length, 1);
}

// expand() moved to the generated dispatcher surface (native_functions.yaml);
// the derivative formulas in derivatives.yaml now resolve against
// tensorplay::tpx::ops::expand, which carries autograd routing.

// Mirrors torch's ToCopyBackward / _to_copy_backward: the gradient is cast
// back to the source tensor's dtype and device.
struct ToCopyBackward : public Node {
    DType dtype_;
    Device device_;

    ToCopyBackward(DType dtype, Device device) : dtype_(dtype), device_(device) {}

    size_t num_inputs() const override { return 1; }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        return {inputs[0].to(device_, dtype_)};
    }
};

Tensor to(const Tensor& self, DType dtype, bool non_blocking, bool copy) {
    bool requires_grad = self.requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (requires_grad && (self.dtype() != dtype)) {
        grad_fn = std::make_shared<ToCopyBackward>(self.dtype(), self.device());
        grad_fn->add_next_edge_list(collect_next_edges(self));
    }
    Tensor result = self.to(dtype, non_blocking, copy);
    if (requires_grad && result.defined() && grad_fn) {
        impl::set_grad_fn(result, grad_fn);
    }
    return result;
}

Tensor to(const Tensor& self, Device device, bool non_blocking, bool copy) {
    bool requires_grad = self.requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (requires_grad && !(self.device() == device)) {
        grad_fn = std::make_shared<ToCopyBackward>(self.dtype(), self.device());
        grad_fn->add_next_edge_list(collect_next_edges(self));
    }
    Tensor result = self.to(device, non_blocking, copy);
    if (requires_grad && result.defined() && grad_fn) {
        impl::set_grad_fn(result, grad_fn);
    }
    return result;
}

Tensor to(const Tensor& self, Device device, DType dtype, bool non_blocking, bool copy) {
    bool requires_grad = self.requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (requires_grad && ((self.dtype() != dtype) || !(self.device() == device))) {
        grad_fn = std::make_shared<ToCopyBackward>(self.dtype(), self.device());
        grad_fn->add_next_edge_list(collect_next_edges(self));
    }
    Tensor result = self.to(device, dtype, non_blocking, copy);
    if (requires_grad && result.defined() && grad_fn) {
        impl::set_grad_fn(result, grad_fn);
    }
    return result;
}

} // namespace tpx
} // namespace tensorplay