#pragma once
#include "Node.h"
#include "Autograd.h"
#include <tuple>
#include <utility>

namespace tensorplay {
namespace tpx {

// Dummy root node used when executing a graph with multiple roots: it holds
// the root edges and feeds the root gradients as its outputs.
struct GraphRoot : public Node {
    GraphRoot(edge_list functions, variable_list inputs)
        : functions_(std::move(functions)), inputs_(std::move(inputs)) {
        add_next_edge_list(functions_);
    }

    variable_list apply(variable_list&& inputs) override {
        return std::move(inputs_);
    }

    edge_list functions_;
    variable_list inputs_;
};

struct SelectBackward : public Node {
    std::vector<int64_t> input_shape_;
    int64_t dim_;
    int64_t index_;
    DType dtype_;
    Device device_;

    SelectBackward(Size shape, int64_t dim, int64_t index, DType dtype, Device device)
        : input_shape_(static_cast<std::vector<int64_t>>(shape)), dim_(dim), index_(index), dtype_(dtype), device_(device) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        Tensor grad = inputs[0];
        
        Tensor grad_input = Tensor::zeros(input_shape_, dtype_, device_);
        grad_input.select(dim_, index_).copy_(grad);

        return {grad_input};
    }
};

struct SliceBackward : public Node {
    std::vector<int64_t> input_shape_;
    int64_t dim_;
    int64_t start_;
    int64_t end_;
    int64_t step_;
    DType dtype_;
    Device device_;

    SliceBackward(Size shape, int64_t dim, int64_t start, int64_t end, int64_t step, DType dtype, Device device)
        : input_shape_(static_cast<std::vector<int64_t>>(shape)), dim_(dim), start_(start), end_(end), step_(step), dtype_(dtype), device_(device) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        Tensor grad = inputs[0];
        
        Tensor grad_input = Tensor::zeros(input_shape_, dtype_, device_);
        grad_input.slice(dim_, start_, end_, step_).copy_(grad);

        return {grad_input};
    }
};

struct AsStridedBackward : public Node {
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> view_size_;
    std::vector<int64_t> view_stride_;
    std::optional<int64_t> storage_offset_;
    DType dtype_;
    Device device_;

    AsStridedBackward(Size input_shape, std::vector<int64_t> view_size, std::vector<int64_t> view_stride, std::optional<int64_t> storage_offset, DType dtype, Device device)
        : input_shape_(static_cast<std::vector<int64_t>>(input_shape)), 
          view_size_(std::move(view_size)), 
          view_stride_(std::move(view_stride)), 
          storage_offset_(storage_offset), 
          dtype_(dtype), 
          device_(device) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        Tensor grad = inputs[0];
        
        Tensor grad_input = Tensor::zeros(input_shape_, dtype_, device_);

        // Create view of grad_input and accumulate gradient
        // We use p10 methods directly to avoid autograd overhead here
        grad_input.as_strided(view_size_, view_stride_, storage_offset_).add_(grad);

        return {grad_input};
    }
};

// SDPA returns three gradients from one fused reference/backward kernel.  A
// hand-written node keeps the tuple alive once per backward invocation instead
// of evaluating the expensive native backward three times (one per input),
// which is particularly important for Llama training.
struct ScaledDotProductAttentionBackward : public Node {
    Tensor query_;
    Tensor key_;
    Tensor value_;
    bool is_causal_;
    int64_t impl_;

    ScaledDotProductAttentionBackward(Tensor query, Tensor key, Tensor value,
                                      bool is_causal, int64_t impl)
        : query_(std::move(query)), key_(std::move(key)), value_(std::move(value)),
          is_causal_(is_causal), impl_(impl) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) {
            return {Tensor(), Tensor(), Tensor()};
        }
        auto grads = Tensor::scaled_dot_product_attention_backward(
            inputs[0], query_, key_, value_, is_causal_, impl_);
        return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads)};
    }
};

// mean(dtype=...) may accumulate in a wider dtype, but its derivative must be
// represented in the input dtype (the same contract as torch).  Keep this
// cast in the manual node so a float32 reduction of an fp16/bf16 tensor does
// not leak a float32 gradient into the leaf or into the SDPA backward node.
struct MeanBackward : public Node {
    Tensor self_;

    explicit MeanBackward(Tensor self) : self_(std::move(self)) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        Tensor grad = inputs[0].expand(self_.shape());
        if (grad.dtype() != self_.dtype()) grad = grad.to(self_.dtype());
        return {grad / Scalar(static_cast<float>(self_.numel()))};
    }
};

struct CatBackward : public Node {
    std::vector<Tensor> tensors_;
    int64_t dim_;

    CatBackward(std::vector<Tensor> tensors, int64_t dim)
        : tensors_(std::move(tensors)), dim_(dim) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) {
            return variable_list(tensors_.size(), Tensor());
        }
        const Tensor& grad = inputs[0];
        int64_t dim = dim_ < 0 ? dim_ + grad.dim() : dim_;
        int64_t offset = 0;
        variable_list grads;
        grads.reserve(tensors_.size());
        for (const auto& tensor : tensors_) {
            const int64_t size = tensor.size(dim);
            grads.push_back(grad.slice(dim, offset, offset + size));
            offset += size;
        }
        return grads;
    }
};

struct StackBackward : public Node {
    std::vector<Tensor> tensors_;
    int64_t dim_;

    StackBackward(std::vector<Tensor> tensors, int64_t dim)
        : tensors_(std::move(tensors)), dim_(dim) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) {
            return variable_list(tensors_.size(), Tensor());
        }
        const Tensor& grad = inputs[0];
        int64_t dim = dim_ < 0 ? dim_ + grad.dim() : dim_;
        variable_list grads;
        grads.reserve(tensors_.size());
        for (size_t i = 0; i < tensors_.size(); ++i) {
            grads.push_back(grad.select(dim, static_cast<int64_t>(i)));
        }
        return grads;
    }
};

}
}
