#pragma once
#include "Node.h"
#include "Autograd.h"
#include "SavedVariable.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <tuple>
#include <utility>
#include <algorithm>

namespace tensorplay {
namespace tpx {

// Backward components of the native F.linear (aten Linear.cpp linear_backward
// reduced to its 2-D-weight contract).  Like MatmulBackward, every step
// composes dispatched recordable primitives so create_graph sees a graph
// through the op (double-backward).
//
// Shapes: out = in_flat @ W^T (+ bias), where in_flat is `input` viewed as
// {prod_leading, K} (1-D input behaves as a single row).

inline Tensor linear_backward_input(const Tensor& grad, const Tensor& input,
                                    const Tensor& weight) {
    if (!grad.defined()) return Tensor();
    const bool vector_input = input.dim() <= 1;
    Tensor g = grad.dim() == 1 ? ops::reshape(grad, {1, grad.size(0)}) : grad;
    // product of leading dims of input == rows of g either way.
    Tensor gxw = ops::matmul(g, weight);
    if (vector_input) {
        std::vector<int64_t> in_sizes(
            static_cast<std::vector<int64_t>>(input.shape()));
        return ops::reshape(gxw, in_sizes);
    }
    auto target = static_cast<std::vector<int64_t>>(input.shape());
    if (gxw.dim() != static_cast<int64_t>(target.size()))
        return ops::reshape(gxw, target);
    return gxw;
}

inline Tensor linear_backward_weight(const Tensor& grad, const Tensor& input,
                                     const Tensor& weight) {
    if (!grad.defined()) return Tensor();
    const int64_t k = weight.size(1);
    const int64_t n = weight.size(0);
    // Flatten both grad and input to 2-D so the transpose+matmul contracts
    // over the batch dimension cleanly: dW = g_flat.T @ x_flat -> {N, K}.
    Tensor x = input.dim() >= 2
        ? ops::reshape(input, {-1, k})
        : ops::reshape(input, {1, input.size(0)});
    Tensor g = grad.dim() == 1 ? ops::reshape(grad, {1, n})
                                : ops::reshape(grad, {-1, n});
    return ops::matmul(ops::transpose(g, -2, -1), x);
}

inline Tensor linear_backward_bias(const Tensor& grad) {
    if (!grad.defined()) return Tensor();
    if (grad.dim() == 1) {
        return ops::sum(grad);
    }
    std::vector<int64_t> dims;
    for (int64_t d = 0; d < grad.dim() - 1; ++d) dims.push_back(d);
    return ops::sum(grad, dims);
}

// Port of at::is_expandable_to (aten/src/ATen/ExpandUtils.h): can `shape`
// be expanded to `desired`?  Dims are aligned at the trailing side; a dim
// may differ from the target only when the *source* dim is 1.
inline bool is_expandable_to(const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& desired) {
    const size_t ndim = shape.size();
    const size_t target_dim = desired.size();
    if (ndim > target_dim) return false;
    for (size_t i = 0; i < ndim; ++i) {
        const auto size = shape[ndim - i - 1];
        const auto target = desired[target_dim - i - 1];
        if (size != target && size != 1) return false;
    }
    return true;
}

// Port of at::_sum_to / at::sum_to (aten/src/ATen/ExpandUtils.h): reduce
// `tensor` down to `shape` with a single batched keepdim sum over the
// leading extra dims and any dim whose target size is 1, then view back.
inline Tensor sum_to(Tensor tensor, const std::vector<int64_t>& shape) {
    if (shape.empty()) return ops::sum(tensor);

    const auto sizes = static_cast<std::vector<int64_t>>(tensor.shape());
    std::vector<int64_t> reduce_dims;
    const int64_t leading_dims =
        static_cast<int64_t>(sizes.size() - shape.size());
    for (int64_t i = 0; i < leading_dims; ++i) reduce_dims.push_back(i);
    for (int64_t i = leading_dims; i < static_cast<int64_t>(sizes.size()); ++i) {
        if (shape[i - leading_dims] == 1 && sizes[i] != 1)
            reduce_dims.push_back(i);
    }

    if (!reduce_dims.empty())
        tensor = tensor.sum(reduce_dims, /*keepdim=*/true);

    return leading_dims > 0 ? tensor.view(shape) : tensor;
}

// ATen native op sum_to_size (aten/src/ATen/native/TensorShape.cpp):
// expandability check + sum_to.
inline Tensor sum_to_size(const Tensor& self, const std::vector<int64_t>& size) {
    TP_CHECK(
        is_expandable_to(size, static_cast<std::vector<int64_t>>(self.shape())),
        "size ", Size(size).toString(), " is not expandable to size ",
        self.shape().toString(), ".");

    return sum_to(self, size);
}

// Port of autograd::maybe_multiply
// (torch/csrc/autograd/FunctionsManual.cpp:138): derivatives expressed as
// `expr * alpha` elide the pointwise multiply entirely when the scalar is 1,
// so beta=alpha=1 backwards (every F.linear/addmm training step) no longer
// pay a full-tensor mul + allocation per gradient slot.
inline Tensor maybe_multiply(const Tensor& t, const Scalar& s) {
    if (s.toDouble() == 1.0) return t;
    return t.mul(s);
}

// Faithful port of autograd::repeat_backward
// (torch/csrc/autograd/FunctionsManual.cpp): guard zero repeats, sum away
// unsqueezed leading dims, then one reshape to interleaved (repeat, size)
// pairs — only where repeat != 1 — followed by a single batched sum.
inline Tensor repeat_backward(Tensor grad, const std::vector<int64_t>& repeats,
                              const std::vector<int64_t>& input_shape) {
    if (std::find(repeats.begin(), repeats.end(), 0) != repeats.end()) {
        return ops::zeros(input_shape, grad.dtype(), grad.device());
    }
    const int64_t input_dims = static_cast<int64_t>(input_shape.size());
    const int64_t num_unsqueezed = grad.dim() - input_dims;
    for (int64_t i = 0; i < num_unsqueezed; ++i) {
        grad = grad.sum(std::vector<int64_t>{0}, /*keepdim=*/false);
    }

    std::vector<int64_t> grad_size;
    std::vector<int64_t> sum_dims;
    for (int64_t dim = 0; dim < input_dims; ++dim) {
        const auto repeat = repeats[dim + num_unsqueezed];
        // Reshape gradient (repeat > 1); dims repeated once pass through.
        if (repeat != 1) {
            grad_size.push_back(repeat);
            sum_dims.push_back(static_cast<int64_t>(grad_size.size() - 1));
        }
        grad_size.push_back(input_shape[dim]);
    }
    // One-time reshape & batched sum; empty sum_dims means no repeats beyond
    // unsqueezing and grad already has input_shape.
    if (!sum_dims.empty()) {
        grad = grad.reshape(grad_size);
        grad = grad.sum(sum_dims);
    }
    return grad;
}

// Faithful port of autograd::unsqueeze_to
// (torch/csrc/autograd/FunctionsManual.cpp): squeeze backward re-inserts
// exactly the size-1 dims that the forward removed; dims listed but not
// squeezed (size != 1) pass through untouched.  Ascending sequential
// unsqueeze keeps later insertion indices valid.
inline Tensor unsqueeze_to(const Tensor& grad, const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& self_sizes) {
    const int64_t ndim = static_cast<int64_t>(self_sizes.size());
    std::vector<bool> mask(self_sizes.size(), false);
    for (auto d : dims) {
        if (d < 0) d += ndim;
        if (d >= 0 && d < ndim) mask[static_cast<size_t>(d)] = true;
    }
    Tensor result = grad;
    for (int64_t d = 0; d < ndim; ++d) {
        if (mask[static_cast<size_t>(d)] && self_sizes[static_cast<size_t>(d)] == 1) {
            result = ops::unsqueeze(result, d);
        }
    }
    return result;
}

// Derivative of max/min(dim, keepdim): route the incoming gradient to the
// winning positions.  Recordable composition of unsqueeze/eq/mul -- the
// upstream namesake native (FunctionsManual.cpp) does the same with
// dispatched at:: ops, so create_graph sees through max/min(dim).
inline Tensor value_selecting_reduction_backward(const Tensor& grad, int64_t dim,
                                                 const Tensor& indices,
                                                 const Tensor& self, bool keepdim) {
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0, "value_selecting_reduction_backward expects a non-scalar input");
    const int64_t d = dim < 0 ? dim + nd : dim;
    Tensor g = grad;
    Tensor idx = indices;
    if (!keepdim) {
        g = ops::unsqueeze(g, d);
        idx = ops::unsqueeze(idx, d);
    }
    // Position iota shaped as ones everywhere except the reduced dim, so the
    // equality broadcast marks exactly the winning slot per output element.
    std::vector<int64_t> iota_sizes(static_cast<size_t>(nd), 1);
    iota_sizes[static_cast<size_t>(d)] = self.size(d);
    Tensor pos = ops::arange(Scalar(self.size(d)), DType::Int64, self.device());
    pos = pos.reshape(iota_sizes);
    Tensor mask = ops::eq(idx, pos);
    if (mask.dtype() != g.dtype()) mask = mask.to(g.dtype());
    return ops::mul(g, mask);
}

// Derivative of mean(dim, keepdim): re-insert the reduced dims, scale by the
// kept-element count, then EXPAND back to self's shape.  The expansion must
// be a recorded op (ExpandBackward -> sum_to_size): a bare broadcast would
// leave singleton dims in the gradient shape, silently corrupting
// second-order results.
inline Tensor broadcast_mean_backward(const Tensor& grad, const Tensor& self,
                                      const std::vector<int64_t>& dims, bool keepdim) {
    Tensor g = grad;
    if (!keepdim) {
        std::vector<int64_t> sorted;
        sorted.reserve(dims.size());
        for (auto d : dims) {
            const int64_t dd = d < 0 ? d + static_cast<int64_t>(self.dim()) : d;
            TP_CHECK(dd >= 0 && dd < self.dim(), "Dimension out of range");
            sorted.push_back(dd);
        }
        std::sort(sorted.begin(), sorted.end());
        for (auto d : sorted) g = ops::unsqueeze(g, d);
    }
    const double count =
        static_cast<double>(self.numel()) / static_cast<double>(g.numel());
    Tensor scaled = ops::div(g, Scalar(count));
    return ops::expand(scaled,
                       static_cast<std::vector<int64_t>>(self.shape()));
}

// block_diag backward: scatter each output-block gradient back to its input.
// Upstream derives this from CopySlices on the zeros+slice+copy_ composite;
// our copy_ does not yet record view mutation, so the layout is explicit.
// Block extents follow the forward promotion: 0-D -> 1x1, 1-D -> (1, n).
struct BlockDiagBackward : public Node {
    std::vector<SavedVariable> tensors_;

    explicit BlockDiagBackward(std::vector<Tensor> tensors) {
        tensors_.reserve(tensors.size());
        for (auto& t : tensors) tensors_.emplace_back(std::move(t));
    }

    size_t num_inputs() const override { return 1; }

    variable_list apply(variable_list&& inputs) override {
        const Tensor& grad = inputs.empty() ? Tensor() : inputs[0];
        variable_list grads;
        grads.reserve(tensors_.size());
        int64_t off0 = 0, off1 = 0;
        for (auto& sv : tensors_) {
            Tensor t = sv.unpack();
            const int64_t h = (t.dim() == 0) ? 1 : (t.dim() == 1 ? 1 : t.size(0));
            const int64_t w = (t.dim() == 0) ? 1 : (t.dim() == 1 ? t.size(0) : t.size(1));
            Tensor g;
            if (grad.defined()) {
                g = grad.slice(0, off0, off0 + h)
                         .slice(1, off1, off1 + w);
                if (t.dim() == 1) g = g.squeeze(0);
                else if (t.dim() == 0) g = g.reshape({});
            } else {
                g = Tensor();
            }
            grads.push_back(g);
            off0 += h;
            off1 += w;
        }
        return grads;
    }
};

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

// NOTE(history): a hand-written ScaledDotProductAttentionBackward used to live
// here but was never instantiated -- the generated autograd node (from
// derivatives.yaml) is authoritative and avoids the double bookkeeping.
// mean(dtype=...) may accumulate in a wider dtype, but its derivative must be
// represented in the input dtype (the same contract as torch).  Keep this
// cast in the manual node so a float32 reduction of an fp16/bf16 tensor does
// not leak a float32 gradient into the leaf or into the SDPA backward node.
struct MeanBackward : public Node {
    SavedVariable self_;

    explicit MeanBackward(Tensor self) : self_(std::move(self)) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        const Tensor self = self_.unpack();
        Tensor grad = inputs[0].expand(self.shape());
        if (grad.dtype() != self.dtype()) grad = grad.to(self.dtype());
        return {grad / Scalar(static_cast<float>(self.numel()))};
    }

    void release_variables() override {
        Node::release_variables();
        self_.reset_data();
    }
};

// matmul backward over every dim combination (dot / vec@mat / mat@vec /
// batched with broadcasting).  Upstream likewise keeps a hand-written native
// (matmul_backward, LinearAlgebra.cpp) because derivative formulas cannot
// branch on dim(); like upstream, every step composes dispatched recordable
// primitives, so create_graph sees a graph through `@` (double-backward).
struct MatmulBackward : public Node {
    SavedVariable self_;
    SavedVariable other_;

    explicit MatmulBackward(Tensor self, Tensor other)
        : self_(std::move(self)), other_(std::move(other)) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor(), Tensor()};
        const Tensor grad = inputs[0];
        const Tensor self = self_.unpack();
        const Tensor other = other_.unpack();
        const bool self_vector = self.dim() == 1;
        const bool other_vector = other.dim() == 1;

        if (isComplexType(self.dtype())) {
            // The complex adjoint is the conjugate transpose; the recordable
            // conj building blocks (select/slice derivatives) are not wired
            // yet, so delegate to the retained native helper ops -- values
            // match upstream exactly, but the complex branch does not record
            // (same depth as before this node existed).
            return {ops::matmul_backward_self(grad, self, other),
                    ops::matmul_backward_other(grad, self, other)};
        }

        // Normalize vectors into matrix space (same convention as the fused
        // matmul_backward kernels and upstream LinearAlgebra.cpp).
        Tensor self_m = self_vector ? ops::unsqueeze(self, 0) : self;
        Tensor other_m = other_vector ? ops::unsqueeze(other, -1) : other;
        Tensor grad_m = grad;
        if (self_vector && other_vector) {
            grad_m = ops::unsqueeze(ops::unsqueeze(grad, 0), 0);
        } else if (self_vector) {
            grad_m = ops::unsqueeze(grad, -2);
        } else if (other_vector) {
            grad_m = ops::unsqueeze(grad, -1);
        }

        auto adjoint = [](const Tensor& t) {
            return t.dim() == 2 ? ops::t(t) : ops::transpose(t, -2, -1);
        };
        // Broadcast-accumulate `g` down to `target` (port of the kernels'
        // sum_to_shape_cpu, expressed with a recordable batched keepdim sum).
        auto reduce_to = [](const Tensor& g, const Tensor& target) {
            const auto src = static_cast<std::vector<int64_t>>(g.shape());
            const auto dst = static_cast<std::vector<int64_t>>(target.shape());
            std::vector<int64_t> dims;
            const int64_t leading =
                static_cast<int64_t>(src.size() - dst.size());
            for (int64_t i = 0; i < leading; ++i) dims.push_back(i);
            for (int64_t i = 0; i < static_cast<int64_t>(dst.size()); ++i) {
                if (dst[i] == 1 && src[leading + i] != 1)
                    dims.push_back(leading + i);
            }
            if (dims.empty()) return g;
            Tensor out = ops::sum(g, dims, /*keepdim=*/true);
            if (out.dim() != static_cast<int64_t>(dst.size()))
                out = ops::reshape(out, dst);
            return out;
        };

        Tensor grad_self = ops::matmul(grad_m, adjoint(other_m));
        grad_self = reduce_to(grad_self, self_m);
        if (self_vector) grad_self = ops::squeeze(grad_self, 0);
        if (grad_self.dtype() != self.dtype())
            grad_self = grad_self.to(self.dtype());

        Tensor grad_other = ops::matmul(adjoint(self_m), grad_m);
        grad_other = reduce_to(grad_other, other_m);
        if (other_vector) grad_other = ops::squeeze(grad_other, -1);
        if (grad_other.dtype() != other.dtype())
            grad_other = grad_other.to(other.dtype());

        return {grad_self, grad_other};
    }

    void release_variables() override {
        Node::release_variables();
        self_.reset_data();
        other_.reset_data();
    }
};

struct CatBackward : public Node {
    std::vector<SavedVariable> tensors_;
    int64_t dim_;

    CatBackward(std::vector<Tensor> tensors, int64_t dim) : dim_(dim) {
        tensors_.reserve(tensors.size());
        for (auto& t : tensors) tensors_.emplace_back(std::move(t));
    }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) {
            return variable_list(tensors_.size(), Tensor());
        }
        const Tensor& grad = inputs[0];
        int64_t dim = dim_ < 0 ? dim_ + grad.dim() : dim_;
        int64_t offset = 0;
        variable_list grads;
        grads.reserve(tensors_.size());
        for (const auto& saved : tensors_) {
            const Tensor tensor = saved.unpack();
            const int64_t size = tensor.size(dim);
            grads.push_back(grad.slice(dim, offset, offset + size));
            offset += size;
        }
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        for (auto& saved : tensors_) saved.reset_data();
    }
};

// ===========================================================================
// Multi-output view ops: unbind / split / split.sizes / chunk
// (torch FunctionsManual.cpp unbind_backward / split_with_sizes_backward).
// One shared node serves all forward outputs; each output carries the same
// grad_fn with its own output_nr, which indexes the node's input slots.
// Unused output slots are zero-filled: the engine materializes them from
// output_metas_, and apply() additionally substitutes zeros defensively.
// ===========================================================================

namespace detail {

inline void record_output_slots(Node* node,
                                std::vector<std::vector<int64_t>>& shapes,
                                const std::vector<Tensor>& outputs,
                                DType& dtype, Device& device) {
    shapes.reserve(outputs.size());
    node->output_metas().reserve(outputs.size());
    for (const auto& t : outputs) {
        shapes.push_back(static_cast<std::vector<int64_t>>(t.shape()));
        OutputSlotMeta m;
        m.shape = shapes.back();
        m.dtype = t.dtype();
        m.device_index = t.device().index();
        m.valid = true;
        node->output_metas().push_back(std::move(m));
    }
    if (!outputs.empty()) {
        dtype = outputs[0].dtype();
        device = outputs[0].device();
    }
}

inline std::vector<Tensor> materialize_grads(
        const variable_list& inputs,
        const std::vector<std::vector<int64_t>>& shapes,
        DType dtype, const Device& device) {
    std::vector<Tensor> grads;
    grads.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        if (i < inputs.size() && inputs[i].defined()) {
            grads.push_back(inputs[i]);
        } else {
            grads.push_back(ops::zeros(shapes[i], dtype, device));
        }
    }
    return grads;
}

} // namespace detail

struct UnbindBackward : public Node {
    int64_t dim_;
    std::vector<std::vector<int64_t>> shapes_;
    DType dtype_ = DType::Float32;
    Device device_{DeviceType::CPU};

    UnbindBackward(int64_t dim, const std::vector<Tensor>& outputs) : dim_(dim) {
        detail::record_output_slots(this, shapes_, outputs, dtype_, device_);
    }

    size_t num_inputs() const override { return shapes_.size(); }

    variable_list apply(variable_list&& inputs) override {
        // torch unbind_backward: stack(grads, dim) with undefined grads
        // replaced by zeros.
        std::vector<Tensor> grads = detail::materialize_grads(inputs, shapes_, dtype_, device_);
        if (grads.empty()) return {Tensor()};
        return {ops::stack(grads, dim_)};
    }
};

// Shared by split, split.sizes and chunk: torch records SplitBackward0 for
// all three (chunk is CompositeImplicitAutograd through split).
struct SplitBackward : public Node {
    int64_t dim_;
    std::vector<std::vector<int64_t>> shapes_;
    DType dtype_ = DType::Float32;
    Device device_{DeviceType::CPU};

    SplitBackward(int64_t dim, const std::vector<Tensor>& outputs) : dim_(dim) {
        detail::record_output_slots(this, shapes_, outputs, dtype_, device_);
    }

    size_t num_inputs() const override { return shapes_.size(); }

    variable_list apply(variable_list&& inputs) override {
        // torch split_with_sizes_backward: cat(grads, dim) with undefined
        // grads replaced by zeros sized to each split.
        std::vector<Tensor> grads = detail::materialize_grads(inputs, shapes_, dtype_, device_);
        if (grads.empty()) return {Tensor()};
        return {ops::cat(grads, dim_)};
    }
};

// torch differentiates roll as grad.roll(-shifts, dims) (derivatives.yaml:
// "self: grad.roll_symint(fmap(reverse_list_symint(shifts), [](c10::SymInt i)
// {return -i;}), reverse_list(dims))").  TensorPlay's formula DSL cannot map
// over int64 lists, so the element-wise negation happens here and the node
// simply re-rolls the gradient with the negated shifts.
struct RollBackward : public Node {
    std::vector<int64_t> shifts_;
    std::vector<int64_t> dims_;

    RollBackward(std::vector<int64_t> shifts, std::vector<int64_t> dims)
        : shifts_(std::move(shifts)), dims_(std::move(dims)) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        const Tensor& grad = inputs[0];

        std::vector<int64_t> neg_shifts(shifts_.size());
        for (size_t i = 0; i < shifts_.size(); ++i) neg_shifts[i] = -shifts_[i];
        return {tensorplay::tpx::ops::roll(grad, neg_shifts, dims_)};
    }
};

struct StackBackward : public Node {
    std::vector<SavedVariable> tensors_;
    int64_t dim_;

    StackBackward(std::vector<Tensor> tensors, int64_t dim) : dim_(dim) {
        tensors_.reserve(tensors.size());
        for (auto& t : tensors) tensors_.emplace_back(std::move(t));
    }

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

    void release_variables() override {
        Node::release_variables();
        for (auto& saved : tensors_) saved.reset_data();
    }
};

// ===========================================================================
// Manual backward helpers -- port of torch/csrc/autograd/FunctionsManual.cpp
// for the elementwise formulas that carry torch's complex convention: every
// pointwise Jacobian J is applied as grad * J.conj(), so the stored leaf
// gradient matches torch's conjugated-gradient representation.
// conj over real dtypes is an alias (see conj_cpu), so the real training
// path stays copy-free.
// ===========================================================================

// FunctionsManual.cpp handle_r_to_c: if the forward output dtype is real but
// the formula produced a complex gradient, keep only the real part.
inline Tensor handle_r_to_c(DType self_st, Tensor gradient_result) {
    if (!isComplexType(self_st) && isComplexType(gradient_result.dtype())) {
        return ops::real(gradient_result);
    }
    return gradient_result;
}

// Scalar flavor of the mul backward's `other.conj()`: conjugate a complex
// python/C++ scalar in place of a tensor op.
inline Scalar scalar_conj_if_complex(const Scalar& s) {
    if (!s.isComplex()) return s;
    const std::complex<double> c = s.to<std::complex<double>>();
    return Scalar(std::complex<double>(c.real(), -c.imag()));
}

// FunctionsManual.cpp mul_tensor_backward: grad * other.conj()
template <typename T>
inline Tensor mul_tensor_backward(const Tensor& grad, const T& other,
                                  DType self_st) {
    Tensor scaled;
    if constexpr (std::is_same_v<T, Scalar>) {
        scaled = grad * scalar_conj_if_complex(other);
    } else {
        scaled = grad * ops::conj(other);
    }
    return handle_r_to_c(self_st, std::move(scaled));
}

// FunctionsManual.cpp div_tensor_self_backward: grad / other.conj()
template <typename T>
inline Tensor div_tensor_self_backward(const Tensor& grad, const T& other,
                                       DType self_st) {
    Tensor scaled;
    if constexpr (std::is_same_v<T, Scalar>) {
        scaled = grad / scalar_conj_if_complex(other);
    } else {
        scaled = grad / ops::conj(other);
    }
    return handle_r_to_c(self_st, std::move(scaled));
}

// div_tensor_other_backward: -grad * conj((self / other) / other)
// (FunctionsManual.cpp conjugates the whole quotient, self included).
inline Tensor div_tensor_other_backward(const Tensor& grad,
                                        const Tensor& self,
                                        const Tensor& other) {
    return handle_r_to_c(
        other.dtype(), -(grad * ops::conj(self / other / other)));
}

// FunctionsManual.cpp pow_backward (scalar exponent): zero exponent short
// circuits; otherwise exponent * self^(exponent-1) under conj.
inline Tensor pow_backward(const Tensor& grad, const Tensor& self,
                           const Scalar& exponent) {
    if (exponent.isIntegral() && exponent.to<int64_t>() == 0) {
        return ops::zeros_like(self);
    }
    return handle_r_to_c(
        self.dtype(),
        grad * ops::conj(self.pow(Scalar(exponent.to<double>() - 1.0)) *
                         exponent.to<double>()));
}

// pow_backward_self (tensor exponent): d z^b / dz = b * z^(b-1)
inline Tensor pow_backward_self(const Tensor& grad, const Tensor& self,
                                const Tensor& exponent) {
    const Tensor one = ops::ones_like(exponent);
    return handle_r_to_c(
        self.dtype(),
        grad * ops::conj(exponent * self.pow(exponent - one)));
}

// pow_backward_exponent (tensor base): d(a^b)/db = a^b * log(a); zeros where
// base == 0 with non-negative real exponent.
inline Tensor pow_backward_exponent(const Tensor& grad, const Tensor& self,
                                    const Tensor& exponent,
                                    const Tensor& result) {
    const Tensor cond = ops::logical_and(ops::eq(self, Scalar(0)),
                                         ops::ge(exponent, Scalar(0)));
    return ops::where(cond, ops::zeros_like(self),
                      grad * ops::conj(result * self.log()));
}

// FunctionsManual.cpp log1p_backward: grad / (self + 1).conj()
inline Tensor log1p_backward(const Tensor& grad, const Tensor& self) {
    return grad / ops::conj(self + 1);
}

// derivatives.yaml acosh: real path keeps the cheap (x*x-1).rsqrt(); complex
// uses the numerically safer ((x+1).rsqrt() * (x-1).rsqrt()).conj().
inline Tensor acosh_backward(const Tensor& grad, const Tensor& self) {
    if (!isComplexType(self.dtype())) {
        return grad * ops::rsqrt(self * self - 1);
    }
    return grad * ops::conj(ops::rsqrt(self + 1) * ops::rsqrt(self - 1));
}

// FunctionsManual.cpp angle_backward: zero at z == 0, otherwise
// grad * i * z / |z|^2 (already the conjugated Jacobian).
inline Tensor angle_backward(const Tensor& grad, const Tensor& self) {
    if (!isComplexType(self.dtype())) {
        return ops::zeros_like(self);
    }
    const Tensor zero_c = ops::eq(self, Scalar(0));
    const Tensor zero = ops::zeros_like(self);
    const Tensor ii = ops::full({}, Scalar(std::complex<double>(0.0, 1.0)),
                                DType::ComplexDouble, self.device());
    const Tensor zd = self.to(DType::ComplexDouble);
    Tensor out = grad * zd * ii / zd.abs().pow(Scalar(2));
    return ops::where(zero_c, zero, out.to(self.dtype()));
}

// FunctionsManual.cpp prod_backward fast path: exact when no element is zero;
// all-zero gradient when more than one zero exists (the single-zero scatter
// case needs nonzero(), which p10 does not expose yet).
inline Tensor prod_backward_fast(const Tensor& grad, const Tensor& input,
                                 const Tensor& result) {
    if (input.dim() == 0) return grad;
    const Tensor has_zero = ops::any(ops::eq(input, Scalar(0)));
    if (has_zero.item().to<bool>()) {
        return ops::zeros_like(input);
    }
    return grad * ops::conj(result / input);
}

// ===========================================================================
// Reduction / indexing / special-function backward helpers.
// Faithful ports of torch/csrc/autograd/FunctionsManual.cpp (and the ATen
// natives it delegates to), expressed as compositions of dispatched recordable
// primitives so first-order values match torch and create_graph records.
// ===========================================================================

inline std::vector<int64_t> wrap_dims(const std::vector<int64_t>& dims,
                                      int64_t ndim) {
    std::vector<int64_t> out;
    out.reserve(dims.size());
    for (auto d : dims) out.push_back(d < 0 ? d + ndim : d);
    return out;
}

inline std::vector<int64_t> all_dims(int64_t ndim) {
    std::vector<int64_t> d(ndim);
    for (int64_t i = 0; i < ndim; ++i) d[i] = i;
    return d;
}

// FunctionsManual.cpp restore_reduced_dims: re-insert size-1 dims removed by
// a keepdim=False reduction so the gradient broadcasts against the input.
inline Tensor restore_reduced_dims(const Tensor& output,
                                   const std::vector<int64_t>& dims,
                                   bool keepdim) {
    if (keepdim) return output;
    const int64_t total = output.dim() + static_cast<int64_t>(dims.size());
    std::vector<int64_t> target(total, -1);
    for (auto i : dims) {
        if (i < 0) i += total;
        target[i] = 1;
    }
    int64_t j = 0;
    for (int64_t d = 0; d < output.dim(); ++d) {
        while (j < total && target[j] != -1) ++j;
        target[j++] = output.size(d);
    }
    return ops::reshape(output, target);
}

// FunctionsManual.cpp scale_grad_by_count.
inline Tensor scale_grad_by_count(const Tensor& grad, const Tensor& mask,
                                  const std::vector<int64_t>& dims) {
    Tensor mask_f = mask.dtype() == grad.dtype() ? mask : mask.to(grad.dtype());
    return ops::mul(ops::div(grad, ops::sum(mask_f, dims, true)), mask_f);
}

// amax/amin backward (FunctionsManual.cpp: restore dims, mask the argmax
// positions, split the gradient across ties).
inline Tensor amax_amin_backward(const Tensor& grad, const Tensor& self,
                                 const Tensor& result,
                                 std::vector<int64_t> dims, bool keepdim) {
    const int64_t nd = self.dim();
    dims = wrap_dims(dims, nd);
    if (dims.empty()) dims = all_dims(nd);
    const Tensor g = restore_reduced_dims(grad, dims, keepdim);
    const Tensor r = restore_reduced_dims(result, dims, keepdim);
    return scale_grad_by_count(g, ops::eq(r, self), dims);
}

// FunctionsManual.cpp evenly_distribute_backward (device-generic branch):
// full-reduction max/min/median spread the gradient evenly across all
// positions that attained the reduced value (NaN matches NaN).
inline Tensor evenly_distribute_backward(const Tensor& grad,
                                         const Tensor& input,
                                         const Tensor& value) {
    const Tensor both_nan =
        ops::logical_and(ops::isnan(input), ops::isnan(value));
    const Tensor mask = ops::logical_or(ops::eq(input, value), both_nan);
    Tensor mask_f = mask.dtype() == grad.dtype() ? mask : mask.to(grad.dtype());
    return ops::mul(mask_f, ops::div(grad, ops::sum(mask_f)));
}

// ATen value_selecting_reduction_backward (ReduceOps.cpp): route the gradient
// to the winning positions via scatter (O(n); used by topk/sort/mode/kthvalue
// and dim-reductions returning indices).
inline Tensor value_selecting_backward(const Tensor& grad, int64_t dim,
                                       const Tensor& indices,
                                       const Tensor& self, bool keepdim) {
    const int64_t nd = self.dim();
    const int64_t d = dim < 0 ? dim + nd : dim;
    Tensor g = grad, idx = indices;
    if (!keepdim && nd > 0) {
        g = ops::unsqueeze(g, d);
        idx = ops::unsqueeze(idx, d);
    }
    if (g.dtype() != self.dtype()) g = g.to(self.dtype());
    return ops::scatter(ops::zeros_like(self), d, idx, g);
}

// ATen cummaxmin_backward (ReduceOps.cpp): duplicate winning positions are
// accumulated with scatter_add.
inline Tensor cummaxmin_backward(const Tensor& grad, const Tensor& input,
                                 const Tensor& indices, int64_t dim) {
    if (input.numel() == 0) return input;
    const int64_t nd = input.dim();
    const int64_t d = dim < 0 ? dim + nd : dim;
    Tensor g = grad.dtype() == input.dtype() ? grad : grad.to(input.dtype());
    return ops::scatter_add(ops::zeros_like(input), d, indices, g);
}

// FunctionsManual.cpp sum_backward: restore reduced dims, expand back.
inline Tensor sum_backward(const Tensor& grad,
                           const std::vector<int64_t>& sizes,
                           std::vector<int64_t> dims, bool keepdim) {
    if (sizes.empty()) return grad;
    dims = wrap_dims(dims, static_cast<int64_t>(sizes.size()));
    if (dims.empty()) dims = all_dims(static_cast<int64_t>(sizes.size()));
    return ops::expand(restore_reduced_dims(grad, dims, keepdim), sizes);
}

// FunctionsManual.cpp nansum_backward.
inline Tensor nansum_backward(const Tensor& grad, const Tensor& self,
                              const std::vector<int64_t>& dims, bool keepdim) {
    const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    Tensor g = sum_backward(grad, sizes, dims, keepdim);
    if (g.dtype() != self.dtype()) g = g.to(self.dtype());
    return ops::mul(g, ops::logical_not(ops::isnan(self)));
}

// nanmean backward: scale by the per-slice non-NaN count, then mask.
inline Tensor nanmean_backward(const Tensor& grad, const Tensor& self,
                               std::optional<int64_t> dim, bool keepdim) {
    const int64_t nd = self.dim();
    std::vector<int64_t> dims;
    if (dim.has_value()) {
        dims.push_back(*dim < 0 ? *dim + nd : *dim);
    } else {
        dims = all_dims(nd);
    }
    const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    Tensor g = sum_backward(grad, sizes, dims, keepdim);
    if (g.dtype() != self.dtype()) g = g.to(self.dtype());
    const Tensor non_nan = ops::logical_not(ops::isnan(self));
    const Tensor count = ops::sum(non_nan.to(g.dtype()), dims, true);
    return ops::mul(ops::div(g, count), non_nan);
}

// FunctionsManual.cpp norm_backward (both arities).  Real-dtype port; the
// p == 0 norm is a count and has no gradient (undefined, engine treats as 0).
inline Tensor norm_backward(Tensor grad, const Tensor& self, double p,
                            Tensor norm, std::vector<int64_t> dims,
                            bool keepdim) {
    const int64_t ndim = self.dim();
    if (!keepdim && ndim != 0) {
        dims = wrap_dims(dims, ndim);
        if (dims.empty()) dims = all_dims(ndim);
        grad = restore_reduced_dims(grad, dims, keepdim);
        norm = restore_reduced_dims(norm, dims, keepdim);
    }
    if (dims.empty()) dims = all_dims(ndim);
    if (p == 0.0) {
        return Tensor();
    } else if (p == 1.0) {
        return ops::sgn(self) * grad;
    } else if (p == 2.0) {
        return grad * ops::masked_fill(self / norm, ops::eq(norm, Scalar(0)),
                                       Scalar(0));
    } else if (std::isinf(p)) {
        const Tensor self_abs = ops::abs(self);
        const Tensor mask = ops::logical_or(ops::eq(self_abs, norm),
                                            ops::isnan(self_abs));
        Tensor mask_f = mask.to(grad.dtype());
        return ops::sgn(self) *
               ops::mul(ops::div(grad, ops::sum(mask_f, dims, true)), mask_f);
    } else if (p < 1.0) {
        const Tensor self_scaled =
            ops::sgn(self) *
            ops::masked_fill(ops::abs(self).pow(Scalar(p - 1.0)),
                             ops::eq(self, Scalar(0)), Scalar(0));
        return self_scaled * grad * norm.pow(Scalar(1.0 - p));
    } else if (p < 2.0) {
        const Tensor self_scaled = ops::sgn(self) * ops::abs(self).pow(Scalar(p - 1.0));
        Tensor scale_v = ops::masked_fill(grad / norm.pow(Scalar(p - 1.0)),
                                          ops::eq(norm, Scalar(0)), Scalar(0));
        return self_scaled * scale_v;
    } else {
        const Tensor self_scaled = self * ops::abs(self).pow(Scalar(p - 2.0));
        Tensor scale_v = ops::masked_fill(grad / norm.pow(Scalar(p - 1.0)),
                                          ops::eq(norm, Scalar(0)), Scalar(0));
        return self_scaled * scale_v;
    }
}

inline Tensor norm_backward(const Tensor& grad, const Tensor& self, double p,
                            const Tensor& norm) {
    return norm_backward(grad, self, p, norm, {}, true);
}

// Scalar-p flavor (used by dist, whose p is a Scalar in the schema).
inline Tensor norm_backward(const Tensor& grad, const Tensor& self,
                            const Scalar& p, const Tensor& norm) {
    return norm_backward(grad, self, p.to<double>(), norm, {}, true);
}

// FunctionsManual.cpp prod_safe_zeros_backward: exclusive normal/reverse
// cumprod pair -- exact even when the input contains zeros.
inline Tensor prod_safe_zeros_backward(const Tensor& grad, const Tensor& inp,
                                       int64_t dim) {
    if (inp.numel() == 0) return ops::expand_as(grad, inp);
    if (inp.size(dim) == 1) return grad;
    auto ones_size = static_cast<std::vector<int64_t>>(inp.shape());
    ones_size[dim] = 1;
    const Tensor ones = ops::ones(ones_size, grad.dtype(), grad.device());
    const Tensor excl_normal = ops::cumprod(
        ops::cat({ones, ops::narrow(inp, dim, 0, inp.size(dim) - 1)}, dim), dim);
    const Tensor excl_reverse = ops::flip(
        ops::cumprod(
            ops::cat({ops::ones(ones_size, grad.dtype(), grad.device()),
                      ops::flip(ops::narrow(inp, dim, 1, inp.size(dim) - 1),
                                {dim})},
                     dim),
            dim),
        {dim});
    return grad * ops::conj(excl_normal * excl_reverse);
}

// FunctionsManual.cpp prod_backward (full reduction): exact including the
// single-zero case (the safe path handles it; >1 zeros naturally give 0).
inline Tensor prod_backward(const Tensor& grad, const Tensor& input,
                            const Tensor& result) {
    if (input.dim() == 0) return grad;
    const Tensor flat = ops::reshape(input, {-1});
    const int64_t total_zeros =
        ops::sum(ops::eq(flat, Scalar(0))).item().to<int64_t>();
    if (total_zeros == 0) return grad * ops::conj(result / input);
    const auto sizes = static_cast<std::vector<int64_t>>(input.shape());
    return ops::reshape(
        prod_safe_zeros_backward(ops::reshape(grad, {-1}), flat, 0), sizes);
}

// prod_backward over a dim list: move the reduced dims to the back, flatten
// into rows (one per fiber), apply the 1-D algorithm along the row dim, and
// permute back.  Exact with zeros for the same reason as the 1-D case.
inline Tensor prod_backward(Tensor grad, const Tensor& input, Tensor result,
                            std::vector<int64_t> dims, bool keepdim) {
    const int64_t nd = input.dim();
    if (nd == 0) return grad;
    dims = wrap_dims(dims, nd);
    if (dims.empty()) dims = all_dims(nd);
    if (!keepdim) {
        // Unsqueeze the reduced slots, then expand to the input shape so the
        // permute/flatten below lines grad/result up with the input fibers.
        const auto in_shape = static_cast<std::vector<int64_t>>(input.shape());
        grad = ops::expand(restore_reduced_dims(grad, dims, keepdim), in_shape);
        result = ops::expand(restore_reduced_dims(result, dims, keepdim),
                             in_shape);
    }
    std::vector<bool> reduced(nd, false);
    for (auto d : dims) reduced[d] = true;
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < nd; ++i) if (!reduced[i]) perm.push_back(i);
    for (auto d : dims) perm.push_back(d);
    const int64_t keep_cnt = static_cast<int64_t>(perm.size() - dims.size());
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < keep_cnt; ++i) outer *= input.size(perm[i]);
    for (size_t i = keep_cnt; i < perm.size(); ++i) inner *= input.size(perm[i]);
    auto permuted_sizes = static_cast<std::vector<int64_t>>(
        ops::permute(input, perm).shape());
    Tensor inp2d = ops::reshape(ops::permute(input, perm), {outer, inner});
    Tensor g2d = ops::reshape(ops::permute(grad, perm), {outer, inner});
    Tensor r2d = ops::reshape(ops::permute(result, perm), {outer, inner});
    const int64_t total_zeros =
        ops::sum(ops::eq(inp2d, Scalar(0))).item().to<int64_t>();
    Tensor out2d = total_zeros == 0
        ? g2d * ops::conj(r2d / inp2d)
        : prod_safe_zeros_backward(g2d, inp2d, 1);
    Tensor out_perm = ops::reshape(out2d, permuted_sizes);
    std::vector<int64_t> inv(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) inv[perm[i]] = static_cast<int64_t>(i);
    return ops::permute(out_perm, inv);
}

// ATen cumprod_backward (ReduceOps.cpp): O(n) zero-aware composition
// (reversed cumsum of output*grad divided by the input, with the first-zero
// mask gymnastics for slices containing zeros).
inline Tensor reversed_cumsum(const Tensor& w, int64_t dim) {
    return ops::flip(ops::cumsum(ops::flip(w, {dim}), dim), {dim});
}

inline Tensor cumprod_backward(const Tensor& grad, const Tensor& input,
                               int64_t dim, const Tensor& output) {
    if (input.numel() <= 1) return grad;
    const int64_t nd = input.dim();
    const int64_t d = dim < 0 ? dim + nd : dim;
    if (input.size(d) == 1) return grad;
    const Tensor input_conj = ops::conj(input);
    const Tensor output_conj = ops::conj(output);
    const Tensor w = output_conj * grad;
    const Tensor is_zero = ops::eq(input, Scalar(0));
    if (!ops::any(is_zero).item().to<bool>()) {
        return ops::div(reversed_cumsum(w, d), input_conj);
    }
    Tensor grad_input = ops::zeros_like(input);
    const Tensor cumsum_z = ops::cumsum(is_zero, d);

    // k < z1: positions before the first zero.
    const Tensor mask_before = ops::eq(cumsum_z, Scalar(0));
    Tensor grad_before = reversed_cumsum(
        ops::masked_fill(w, ops::logical_not(mask_before), Scalar(0)), d);
    grad_before = ops::div(grad_before, input_conj);
    grad_input = ops::where(mask_before, grad_before, grad_input);

    // k == z1: the first zero itself.
    const Tensor mask1 = ops::eq(cumsum_z, Scalar(1));
    const Tensor first_zero_index = ops::argmax(mask1, d, true);
    const Tensor first_zero_mask = ops::logical_and(mask1, is_zero);
    const Tensor between = ops::logical_and(mask1, ops::logical_not(first_zero_mask));
    Tensor grad_at_fz =
        ops::cumprod(ops::masked_fill(input_conj, ops::logical_not(between),
                                      Scalar(1)), d);
    const Tensor grad_masked =
        ops::masked_fill(grad, ops::ne(cumsum_z, Scalar(1)), Scalar(0));
    const Tensor idx_m1 = ops::where(ops::eq(first_zero_index, Scalar(0)),
                                     ops::zeros_like(first_zero_index),
                                     first_zero_index - 1);
    const Tensor output_before_zero = ops::masked_fill(
        ops::gather(output_conj, d, idx_m1),
        ops::eq(first_zero_index, Scalar(0)), Scalar(1));
    grad_at_fz = ops::mul(ops::sum(grad_at_fz * grad_masked, {d}, true),
                          output_before_zero);
    grad_input = ops::where(first_zero_mask, grad_at_fz, grad_input);
    return grad_input;
}

// FunctionsManual.cpp logcumsumexp_backward (real branch): split positive /
// negative gradient mass, run a reversed logcumsumexp, re-exponentiate.
inline Tensor logcumsumexp_backward(const Tensor& grad, const Tensor& self,
                                    const Tensor& result, int64_t dim) {
    if (grad.dim() == 0 || grad.numel() == 0) return grad;
    const int64_t nd = self.dim();
    const int64_t d = dim < 0 ? dim + nd : dim;
    auto reverse_lse = [&](const Tensor& x) {
        return ops::flip(ops::logcumsumexp(ops::flip(x, {d}), d), {d});
    };
    constexpr double kNegInf = -std::numeric_limits<double>::infinity();
    const Tensor log_abs_grad = ops::log(ops::abs(grad));
    const Tensor log_grad_pos = ops::where(ops::gt(grad, Scalar(0)),
                                           log_abs_grad, Scalar(kNegInf));
    const Tensor log_grad_neg = ops::where(ops::lt(grad, Scalar(0)),
                                           log_abs_grad, Scalar(kNegInf));
    const Tensor out_pos = ops::exp(reverse_lse(log_grad_pos - result) + self);
    const Tensor out_neg = ops::exp(reverse_lse(log_grad_neg - result) + self);
    return out_pos - out_neg;
}

// FunctionsManual.cpp renorm_backward, with linalg_vector_norm expressed as
// the dispatched norm(dim) reduction (same value for strided dense inputs).
inline Tensor renorm_backward(const Tensor& grad, const Tensor& self,
                              const Scalar& p, int64_t dim,
                              const Scalar& maxnorm) {
    const int64_t n = self.dim();
    const int64_t d = dim < 0 ? dim + n : dim;
    std::vector<int64_t> reduce_dims;
    for (int64_t i = 0; i < n; ++i) if (i != d) reduce_dims.push_back(i);
    const double pd = p.to<double>();
    const Tensor norm = ops::norm(self, reduce_dims, pd, true);
    const Tensor grad_output = ops::sum(ops::conj(self) * grad, reduce_dims, true);
    const Tensor nb = norm_backward(grad_output, self, pd, norm, reduce_dims, true);
    const Tensor invnorm = ops::reciprocal(norm + 1e-7);
    const Tensor grad_norm =
        maxnorm.to<double>() * invnorm * (grad - invnorm * nb);
    return ops::where(ops::gt(norm, maxnorm), grad_norm.to(grad.dtype()), grad);
}

// FunctionsManual.cpp sinc_backward.
inline Tensor sinc_backward(const Tensor& grad, const Tensor& self) {
    const double pi = 3.14159265358979323846;
    const Tensor self_pi = self * pi;
    const Tensor self_squared_pi = self * self * pi;
    const Tensor out = grad * ops::conj(
        (self_pi * ops::cos(self_pi) - ops::sin(self_pi)) / self_squared_pi);
    return ops::where(ops::eq(self_squared_pi, Scalar(0)),
                      ops::zeros_like(grad), out);
}

// FunctionsManual.cpp take_backward: flatten + accumulating put.
inline Tensor take_backward(const Tensor& grad, const Tensor& self,
                            const Tensor& indices) {
    const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    const Tensor flat = ops::reshape(self, {-1});
    const Tensor grad_flat = ops::index_add(
        ops::zeros_like(flat), 0, ops::reshape(indices, {-1}),
        ops::reshape(grad, {-1}));
    return ops::reshape(grad_flat, sizes);
}

// ATen masked_select_backward (TensorAdvancedIndexing.cpp).
inline Tensor masked_select_backward(const Tensor& grad, const Tensor& input,
                                     const Tensor& mask) {
    const auto a = static_cast<std::vector<int64_t>>(input.shape());
    const auto b = static_cast<std::vector<int64_t>>(mask.shape());
    const size_t nd = std::max(a.size(), b.size());
    std::vector<int64_t> bshape(nd, 1);
    for (size_t i = 0; i < nd; ++i) {
        const int64_t x = i < nd - a.size() ? 1 : a[i - (nd - a.size())];
        const int64_t y = i < nd - b.size() ? 1 : b[i - (nd - b.size())];
        bshape[i] = std::max(x, y);
    }
    return ops::masked_scatter(
        ops::zeros(bshape, input.dtype(), input.device()), mask, grad);
}

// ATen masked_scatter_backward_symint: the source gradient is the masked
// slice of grad, zero-padded to the source numel and reshaped back.
inline Tensor masked_scatter_backward(const Tensor& grad, const Tensor& mask,
                                      const Tensor& source) {
    const auto sizes = static_cast<std::vector<int64_t>>(source.shape());
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;
    Tensor sel = ops::masked_select(grad, mask);
    if (const int64_t diff = numel - sel.numel(); diff > 0) {
        sel = ops::cat({sel, ops::zeros({diff}, grad.dtype(), grad.device())}, 0);
    }
    return ops::reshape(sel, sizes);
}

// ATen trace_backward_symint: gradient lives on the diagonal.
inline Tensor trace_backward(const Tensor& grad, const Tensor& self) {
    return ops::eye(self.size(0), self.size(1), grad.dtype(), grad.device()) *
           grad;
}

// FunctionsManual.cpp var_backward (dim-list flavor; empty dims == full
// reduction), ported for TP's int correction.
inline Tensor var_backward(Tensor grad, const Tensor& self,
                           std::vector<int64_t> dims, int64_t correction,
                           bool keepdim) {
    const int64_t nd = self.dim();
    dims = wrap_dims(dims, nd);
    if (nd == 0 || dims.empty()) {
        const double dof = static_cast<double>(self.numel()) -
                           static_cast<double>(correction);
        if (dof <= 0) {
            const Tensor mean = ops::mean(self);
            const Tensor nan_t = ops::full_like(
                self, Scalar(std::numeric_limits<double>::quiet_NaN()));
            const Tensor inf_t = ops::full_like(
                self, Scalar(std::numeric_limits<double>::infinity()));
            return grad * ops::where(ops::eq(self, mean), nan_t, inf_t);
        }
        return ops::mul(grad * (self - ops::mean(self)), Scalar(2.0 / dof));
    }
    if (!keepdim && nd > 1) grad = restore_reduced_dims(grad, dims, keepdim);
    int64_t rnumel = 1;
    for (auto d : dims) rnumel *= self.size(d);
    const double dof = static_cast<double>(rnumel) - static_cast<double>(correction);
    return ops::mul(grad * (self - ops::mean(self, dims, true)),
                    Scalar(2.0 / dof));
}

// FunctionsManual.cpp std_backward.
inline Tensor std_backward(const Tensor& result, const Tensor& grad,
                           const Tensor& self,
                           const std::vector<int64_t>& dims,
                           int64_t correction, bool keepdim) {
    const Tensor grad_var = ops::masked_fill(ops::div(grad, result * 2),
                                             ops::eq(result, Scalar(0)),
                                             Scalar(0));
    return var_backward(grad_var, self, dims, correction, keepdim);
}

// FunctionsManual.cpp mean_backward (sizes/dims flavor).
inline Tensor mean_backward(const Tensor& grad,
                            const std::vector<int64_t>& sizes,
                            std::vector<int64_t> dims, bool keepdim) {
    int64_t count = 1;
    const int64_t nd = static_cast<int64_t>(sizes.size());
    dims = wrap_dims(dims, nd);
    if (dims.empty()) {
        for (auto s : sizes) count *= s;
    } else {
        for (auto d : dims) count *= sizes[d];
    }
    return ops::div(sum_backward(grad, sizes, dims, keepdim),
                    Scalar(static_cast<double>(count)));
}

// Emulation of grad.index(indices) (advanced indexing gather) via a linear
// index + index_select; indices address the leading dims, broadcast against
// each other, and may be negative (wrapped).
inline Tensor index_nd(const Tensor& grad, const std::vector<Tensor>& indices) {
    const int64_t nd = grad.dim();
    const int64_t nidx = static_cast<int64_t>(indices.size());
    if (nidx == 0) return grad;
    TP_CHECK(nidx <= nd, "index_nd: more indices than input dims");
    std::vector<int64_t> bshape = {1};
    for (const auto& idx : indices) {
        const auto s = static_cast<std::vector<int64_t>>(idx.shape());
        std::vector<int64_t> out(std::max(bshape.size(), s.size()), 1);
        for (size_t i = 0; i < out.size(); ++i) {
            const int64_t a = i < out.size() - bshape.size()
                                  ? 1 : bshape[i - (out.size() - bshape.size())];
            const int64_t b = i < out.size() - s.size()
                                  ? 1 : s[i - (out.size() - s.size())];
            out[i] = std::max(a, b);
        }
        bshape = std::move(out);
    }
    Tensor linear;
    for (int64_t i = 0; i < nidx; ++i) {
        Tensor idx = indices[i];
        if (idx.dtype() != DType::Int64) idx = idx.to(DType::Int64);
        idx = ops::where(ops::lt(idx, Scalar(0)), idx + grad.size(i), idx);
        idx = ops::expand(idx, bshape);
        int64_t stride = 1;
        for (int64_t j = i + 1; j < nd; ++j) stride *= grad.size(j);
        const Tensor term = idx * stride;
        linear = linear.defined() ? linear + term : term;
    }
    int64_t outer = 1;
    for (int64_t i = 0; i < nidx; ++i) outer *= grad.size(i);
    const Tensor flat = ops::reshape(grad, {outer, -1});
    const Tensor sel = ops::index_select(flat, 0, ops::reshape(linear, {-1}));
    std::vector<int64_t> out_shape = bshape;
    for (int64_t i = nidx; i < nd; ++i) out_shape.push_back(grad.size(i));
    return ops::reshape(sel, out_shape);
}

// ATen index_put backward pair (FunctionsManual.cpp via derivatives.yaml):
// self keeps grad except at overwritten positions (unless accumulate);
// values gather grad at the indexed positions.
inline std::tuple<Tensor, Tensor> index_put_backward(
        const Tensor& grad, bool accumulate,
        const std::vector<Tensor>& indices, const Tensor& values) {
    const Tensor grad_self = accumulate
        ? grad
        : ops::index_put(grad, indices, ops::zeros_like(values), false);
    return {grad_self, index_nd(grad, indices)};
}

// ===========================================================================
// Hand-written backward nodes for ops the formula DSL cannot express:
// list-gradient alignment (index_put), multiple differentiable outputs
// (aminmax, std_mean, var_mean).  apply() outputs are positionally aligned
// with the edges collected at record time (one per tensor argument, list
// elements included), so list slots are padded with undefined grads.
// ===========================================================================

struct IndexPutBackward : public Node {
    std::vector<SavedVariable> indices_;
    SavedVariable values_;
    bool accumulate_;

    IndexPutBackward(std::vector<Tensor> indices, Tensor values, bool accumulate)
        : values_(std::move(values)), accumulate_(accumulate) {
        indices_.reserve(indices.size());
        for (auto& t : indices) indices_.emplace_back(std::move(t));
    }

    variable_list apply(variable_list&& inputs) override {
        variable_list grads;
        grads.reserve(indices_.size() + 2);
        const Tensor grad = inputs.empty() ? Tensor() : inputs[0];
        if (grad.defined()) {
            std::vector<Tensor> idx;
            idx.reserve(indices_.size());
            for (auto& sv : indices_) idx.push_back(sv.unpack());
            auto [gself, gvalues] =
                index_put_backward(grad, accumulate_, idx, values_.unpack());
            grads.push_back(std::move(gself));
            for (size_t i = 0; i < indices_.size(); ++i) grads.push_back(Tensor());
            grads.push_back(std::move(gvalues));
        } else {
            grads.push_back(Tensor());
            for (size_t i = 0; i < indices_.size(); ++i) grads.push_back(Tensor());
            grads.push_back(Tensor());
        }
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        for (auto& sv : indices_) sv.reset_data();
        values_.reset_data();
    }
};

struct AminmaxBackward : public Node {
    SavedVariable self_;
    std::vector<int64_t> dims_;
    bool keepdim_;

    AminmaxBackward(Tensor self, std::vector<int64_t> dims, bool keepdim)
        : self_(std::move(self)), dims_(std::move(dims)), keepdim_(keepdim) {}

    // Two differentiable outputs (min, max): the engine delivers their grads
    // at input slots 0/1, so the InputBuffer must be sized accordingly.
    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty()) return {Tensor()};
        const Tensor grad_min = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor grad_max = inputs.size() > 1 ? inputs[1] : Tensor();
        if (!grad_min.defined() && !grad_max.defined()) return {Tensor()};
        const Tensor self = self_.unpack();
        const int64_t nd = self.dim();
        auto dims = wrap_dims(dims_, nd);
        if (dims.empty()) dims = all_dims(nd);
        // Recompute min/max positions from the saved input: the forward
        // outputs are not saved (torch saves only self as well).
        auto [minv, maxv] = ops::aminmax(self, dims_, keepdim_);
        Tensor result;
        if (grad_min.defined()) {
            const Tensor g = restore_reduced_dims(grad_min, dims, keepdim_);
            const Tensor m = restore_reduced_dims(minv, dims, keepdim_);
            result = scale_grad_by_count(g, ops::eq(self, m), dims);
        }
        if (grad_max.defined()) {
            const Tensor g = restore_reduced_dims(grad_max, dims, keepdim_);
            const Tensor m = restore_reduced_dims(maxv, dims, keepdim_);
            Tensor gmax = scale_grad_by_count(g, ops::eq(self, m), dims);
            result = result.defined() ? result + gmax : std::move(gmax);
        }
        return {result};
    }

    void release_variables() override {
        Node::release_variables();
        self_.reset_data();
    }
};

struct VarMeanBackward : public Node {
    SavedVariable self_;
    std::vector<int64_t> dims_;
    bool unbiased_;
    bool keepdim_;

    VarMeanBackward(Tensor self, std::vector<int64_t> dims, bool unbiased,
                    bool keepdim)
        : self_(std::move(self)), dims_(std::move(dims)), unbiased_(unbiased),
          keepdim_(keepdim) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty()) return {Tensor()};
        const Tensor gvar = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor gmean = inputs.size() > 1 ? inputs[1] : Tensor();
        if (!gvar.defined() && !gmean.defined()) return {Tensor()};
        const Tensor self = self_.unpack();
        const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
        const int64_t correction = unbiased_ ? 1 : 0;
        Tensor gself;
        if (gvar.defined()) {
            gself = var_backward(gvar, self, dims_, correction, keepdim_);
        }
        if (gmean.defined()) {
            Tensor aux = mean_backward(gmean, sizes, dims_, keepdim_);
            gself = gself.defined() ? gself + aux : std::move(aux);
        }
        return {gself};
    }

    void release_variables() override {
        Node::release_variables();
        self_.reset_data();
    }
};

struct StdMeanBackward : public Node {
    SavedVariable self_;
    std::vector<int64_t> dims_;
    bool unbiased_;
    bool keepdim_;

    StdMeanBackward(Tensor self, std::vector<int64_t> dims, bool unbiased,
                    bool keepdim)
        : self_(std::move(self)), dims_(std::move(dims)), unbiased_(unbiased),
          keepdim_(keepdim) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty()) return {Tensor()};
        const Tensor gstd = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor gmean = inputs.size() > 1 ? inputs[1] : Tensor();
        if (!gstd.defined() && !gmean.defined()) return {Tensor()};
        const Tensor self = self_.unpack();
        const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
        const int64_t correction = unbiased_ ? 1 : 0;
        Tensor gself;
        if (gstd.defined()) {
            const Tensor stdv = std::get<0>(
                ops::std_mean(self, dims_, unbiased_, keepdim_));
            gself = std_backward(stdv, gstd, self, dims_, correction, keepdim_);
        }
        if (gmean.defined()) {
            Tensor aux = mean_backward(gmean, sizes, dims_, keepdim_);
            gself = gself.defined() ? gself + aux : std::move(aux);
        }
        return {gself};
    }

    void release_variables() override {
        Node::release_variables();
        self_.reset_data();
    }
};

}
}

#include "RNNBackward.h"
