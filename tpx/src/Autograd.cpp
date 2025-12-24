#include "Autograd.h"
#include "TPXTensor.h"
#include "TensorImpl.h"
#include "AccumulateGrad.h"
#include "Engine.h"

namespace tensorplay {
namespace tpx {

namespace {
    thread_local bool grad_mode_enabled = true;
}

bool GradMode::is_enabled() {
    return grad_mode_enabled;
}

void GradMode::set_enabled(bool enabled) {
    grad_mode_enabled = enabled;
}

struct AutogradMetaFactoryImpl : public AutogradMetaFactory {
    std::shared_ptr<AutogradMetaInterface> make() const override {
        return std::make_shared<AutogradMeta>();
    }
};

static AutogradMetaFactoryImpl global_autograd_meta_factory_impl;

// Register factory on startup
struct AutogradMetaFactoryRegister {
    AutogradMetaFactoryRegister() {
        SetAutogradMetaFactory(&global_autograd_meta_factory_impl);
    }
};

static AutogradMetaFactoryRegister global_autograd_meta_factory_register;

AutogradMeta::AutogradMeta(bool requires_grad) 
    : requires_grad_(requires_grad), retain_grad_(false), output_nr_(0) {}

bool AutogradMeta::requires_grad() const {
    return requires_grad_ || grad_fn_ != nullptr;
}

void AutogradMeta::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
}

Tensor AutogradMeta::grad() const {
    if (grad_) {
        return *grad_;
    }
    return Tensor(); 
}

void AutogradMeta::set_grad(const Tensor& grad) {
    if (grad.defined()) {
        grad_ = std::make_shared<Tensor>(grad);
    } else {
        grad_ = nullptr;
    }
}

void AutogradMeta::accum_grad(const Tensor& grad) {
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(grad);
    } else {
        // Accumulate gradient
        *grad_ = grad_->add(grad);
    }
}

void AutogradMeta::set_grad_fn(std::shared_ptr<Node> grad_fn) {
    grad_fn_ = std::move(grad_fn);
}

std::shared_ptr<Node> AutogradMeta::grad_fn() const {
    return grad_fn_;
}

void AutogradMeta::set_grad_accumulator(std::shared_ptr<Node> grad_accumulator) {
    grad_accumulator_ = std::move(grad_accumulator);
}

std::shared_ptr<Node> AutogradMeta::grad_accumulator() const {
    return grad_accumulator_;
}

uint32_t AutogradMeta::output_nr() const {
    return output_nr_;
}

void AutogradMeta::set_output_nr(uint32_t output_nr) {
    output_nr_ = output_nr;
}

bool AutogradMeta::retain_grad() const {
    return retain_grad_;
}

void AutogradMeta::set_retain_grad(bool retain_grad) {
    retain_grad_ = retain_grad;
}

std::vector<Edge> collect_next_edges(const Tensor& t) {
    std::vector<Edge> edges;
    if (t.requires_grad()) {
        auto fn = t.grad_fn();
        if (fn) {
             uint32_t output_nr = 0;
             if (t.autograd_meta()) output_nr = t.autograd_meta()->output_nr();
             edges.emplace_back(fn, output_nr);
        } else {
             // Leaf
             auto meta = t.autograd_meta();
             if (meta) {
                 if (!meta->grad_accumulator()) {
                     meta->set_grad_accumulator(std::make_shared<AccumulateGrad>(t));
                 }
                 edges.emplace_back(meta->grad_accumulator(), 0);
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
    
    // DEBUG THROW


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
        if (auto fn = tensor.grad_fn()) {
            uint32_t output_nr = 0;
            if (tensor.autograd_meta()) {
                output_nr = tensor.autograd_meta()->output_nr();
            }
            roots.emplace_back(fn, output_nr);
        } else if (tensor.requires_grad()) {
            // Leaf node
            auto meta = tensor.autograd_meta();
            if (meta) {
                if (!meta->grad_accumulator()) {
                    meta->set_grad_accumulator(std::make_shared<AccumulateGrad>(tensor));
                }
                roots.emplace_back(meta->grad_accumulator(), 0);
            }
        }
    }

    Engine::get_default_engine().test_link();
    Engine::get_default_engine().execute_graph(roots, inputs, retain_graph, create_graph);
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
        if (auto fn = output.grad_fn()) {
            uint32_t output_nr = 0;
            if (output.autograd_meta()) {
                output_nr = output.autograd_meta()->output_nr();
            }
            roots.emplace_back(fn, output_nr);
        } else {
            // Leaf
            auto meta = output.autograd_meta();
            if (meta) {
                if (!meta->grad_accumulator()) {
                    meta->set_grad_accumulator(std::make_shared<AccumulateGrad>(output));
                }
                roots.emplace_back(meta->grad_accumulator(), 0);
            }
        }
    }
    
    // 2. Identify capture nodes
    std::unordered_map<Node*, std::vector<Tensor>> capture_grads;
    
    struct InputInfo {
        Node* node;
        uint32_t output_nr; // Index into the captured vector
    };
    std::vector<InputInfo> input_infos;
    input_infos.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        if (!input.requires_grad()) {
            TP_THROW(RuntimeError, "One of the differentiated Tensors does not require grad");
        }
        
        Node* node = nullptr;
        uint32_t output_nr = 0;
        
        if (auto fn = input.grad_fn()) {
            // Non-leaf
            node = fn.get();
            if (input.autograd_meta()) {
                output_nr = input.autograd_meta()->output_nr();
            }
        } else {
            // Leaf
            auto meta = input.autograd_meta();
            if (meta) {
                if (!meta->grad_accumulator()) {
                    meta->set_grad_accumulator(std::make_shared<AccumulateGrad>(input));
                }
                node = meta->grad_accumulator().get();
                // AccumulateGrad always has 1 input (index 0)
                output_nr = 0;
            }
        }
        
        if (!node) {
             TP_THROW(RuntimeError, "Could not determine gradient node for input");
        }
        
        input_infos.push_back({node, output_nr});
        capture_grads[node] = {}; // Initialize entry
    }
    
    // 3. Execute
    Engine::get_default_engine().test_link(); // Ensure engine instance
    Engine::get_default_engine().execute_graph(roots, root_grads, retain_graph, create_graph, &capture_grads);
    
    // 4. Collect results
    std::vector<Tensor> results;
    results.reserve(inputs.size());
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        Node* node = input_infos[i].node;
        uint32_t output_nr = input_infos[i].output_nr;
        
        auto it = capture_grads.find(node);
        bool found = false;
        Tensor res;
        
        if (it != capture_grads.end()) {
            const auto& grads = it->second;
            if (output_nr < grads.size() && grads[output_nr].defined()) {
                res = grads[output_nr];
                found = true;
            }
        }
        
        if (!found) {
            if (!allow_unused) {
                TP_THROW(RuntimeError, "One of the differentiated Tensors was not used in the graph");
            }
        }
        results.push_back(res);
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


} // namespace tpx
} // namespace tensorplay
