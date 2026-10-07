#pragma once
#include "Node.h"
#include "Autograd.h"
#include "SavedVariable.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <array>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <algorithm>

namespace tensorplay {
namespace tpx {

// Backward components of native linear (reduced to its 2-D-weight contract).
// Every step composes dispatched
// recordable primitives so create_graph sees a graph through the op
// (double-backward).
//
// Shapes: out = in_flat @ W^T (+ bias), where in_flat is `input` viewed as
// {prod_leading, K} (1-D input behaves as a single row).

inline Tensor linear_backward_input(const Tensor& grad, const Tensor& input,
                                    const Tensor& weight) {
    if (!grad.defined()) return Tensor();
    const bool vector_input = input.dim() <= 1;
    Tensor g = grad.dim() == 1 ? ops::reshape(grad, {1, grad.size(0)}) : grad;
    if (g.dtype() != weight.dtype()) g = g.to(weight.dtype());
    // product of leading dims of input == rows of g either way.
    Tensor gxw = ops::matmul(g, weight);
    if (vector_input) {
        std::vector<int64_t> in_sizes(
            static_cast<std::vector<int64_t>>(input.shape()));
        gxw = ops::reshape(gxw, in_sizes);
    } else {
        auto target = static_cast<std::vector<int64_t>>(input.shape());
        if (gxw.dim() != static_cast<int64_t>(target.size()))
            gxw = ops::reshape(gxw, target);
    }
    return gxw.dtype() == input.dtype() ? gxw : gxw.to(input.dtype());
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
    if (g.dtype() != x.dtype()) g = g.to(x.dtype());
    Tensor result = ops::matmul(ops::transpose(g, -2, -1), x);
    return result.dtype() == weight.dtype() ? result : result.to(weight.dtype());
}

inline Tensor linear_backward_bias(const Tensor& grad) {
    if (!grad.defined()) return Tensor();
    if (grad.dim() == 1) {
        // a single-row (1-D input) forward where the bias broadcast was an
        // identity, so the bias gradient is the grad itself -- not its sum.
        return grad;
    }
    std::vector<int64_t> dims;
    for (int64_t d = 0; d < grad.dim() - 1; ++d) dims.push_back(d);
    return ops::sum(grad, dims);
}

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
        tensor = ops::sum(tensor, reduce_dims, /*keepdim=*/true);

    return leading_dims > 0 ? ops::view(tensor, shape) : tensor;
}

// Bagged-embedding per-sample-weight gradient.
//
// Only the sum reduction accepts per_sample_weights, so in the other modes the
// slot has no consumer: leaving it undefined skips a full [len(indices), D]
// reduction the engine would immediately discard.
inline Tensor embedding_bag_psw_backward(
        const Tensor& grad, const Tensor& weight, const Tensor& indices,
        const Tensor& offsets, const Tensor& offset2bag,
        const std::optional<Tensor>& per_sample_weights, int64_t mode,
        int64_t padding_idx) {
    constexpr int64_t kBagSum = 0;
    if (mode != kBagSum || !per_sample_weights.has_value() ||
        !per_sample_weights->defined()) {
        return Tensor();
    }
    return ops::_embedding_bag_per_sample_weights_backward(
        grad, weight, indices, offsets, offset2bag, mode, padding_idx);
}

// Scalar multiplier helper for autograd formulas.
// `expr * alpha` elide the pointwise multiply entirely when the scalar is 1,
// so beta=alpha=1 backwards (every F.linear/addmm training step) no longer
// pay a full-tensor mul + allocation per gradient slot.
// PReLU gradients.  The forward reads one slope per channel; the elementwise
// backward kernel wants that slope already shaped to broadcast over `self`,
// and returns the weight gradient per input element.  Folding it back onto
// the parameter sums every axis except the channel axis (all axes when the
// slope is shared).
inline Tensor prelu_broadcast_weight(const Tensor& self, const Tensor& weight) {
    std::vector<int64_t> shape(static_cast<size_t>(self.dim()), 1);
    if (self.dim() >= 2) {
        shape[1] = weight.numel();
    } else if (self.dim() == 1) {
        shape[0] = weight.numel();
    }
    return ops::reshape(weight, shape);
}

inline std::tuple<Tensor, Tensor> prelu_backward(const Tensor& grad,
                                                 const Tensor& self,
                                                 const Tensor& weight) {
    if (!grad.defined()) return {Tensor(), Tensor()};
    const Tensor broadcast_weight = prelu_broadcast_weight(self, weight);
    Tensor grad_input;
    Tensor per_element;
    if (GradMode::is_enabled()) {
        // Recompose the fused kernel from differentiable primitives while a
        // higher-order graph is being recorded.
        const Tensor nonnegative = ops::ge(self, Scalar(0));
        grad_input = ops::where(nonnegative, grad,
                                ops::mul(grad, broadcast_weight));
        per_element = ops::where(nonnegative, ops::zeros_like(self),
                                 ops::mul(grad, self));
    } else {
        auto parts = ops::_prelu_kernel_backward(grad, self, broadcast_weight);
        grad_input = std::get<0>(parts);
        per_element = std::get<1>(parts);
    }

    const auto weight_shape = static_cast<std::vector<int64_t>>(weight.shape());
    Tensor grad_weight;
    if (weight.numel() == 1 || self.dim() == 0) {
        grad_weight = ops::sum(per_element);
    } else {
        const int64_t channel_dim = self.dim() >= 2 ? 1 : 0;
        std::vector<int64_t> reduced;
        for (int64_t d = 0; d < per_element.dim(); ++d) {
            if (d != channel_dim) reduced.push_back(d);
        }
        grad_weight = reduced.empty() ? per_element
                                      : ops::sum(per_element, reduced);
    }
    if (static_cast<std::vector<int64_t>>(grad_weight.shape()) != weight_shape) {
        grad_weight = ops::reshape(grad_weight, weight_shape);
    }
    return {grad_input, grad_weight};
}

inline Tensor maybe_multiply(const Tensor& t, const Scalar& s) {
    bool is_one = false;
    if (s.isFloatingPoint()) {
        is_one = s.toDouble() == 1.0;
    } else if (s.isIntegral(true)) {
        is_one = s.to<int64_t>() == 1;
    }
    return is_one ? t : t.mul(s);
}

// Repeat backward helper.
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

// The backward of a slice / select: the gradient placed into zeros of the
// input's shape. Reading only the input's sizes keeps the node from saving
// the input, which a view of it may rewrite in place afterwards.
inline Tensor slice_backward_sizes(const Tensor& grad, const std::vector<int64_t>& self_sizes,
                                   int64_t dim, std::optional<int64_t> start,
                                   std::optional<int64_t> end, int64_t step) {
    const Tensor zeros = ops::zeros(self_sizes, grad.dtype(), grad.device());
    return ops::slice_scatter(zeros, grad, dim, start, end, step);
}

inline Tensor select_backward_sizes(const Tensor& grad, const std::vector<int64_t>& self_sizes,
                                    int64_t dim, int64_t index) {
    const Tensor zeros = ops::zeros(self_sizes, grad.dtype(), grad.device());
    return ops::select_scatter(zeros, grad, dim, index);
}

// Unsqueeze backward helper.
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
// winning positions through recorded unsqueeze/eq/mul operations.
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

// block_diag backward: scatter each output-block gradient back to its input.
// The layout is explicit because view mutation is not recorded by copy_.
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

// as_strided backward.  Both the input and the view are windows onto one
// storage, so the gradient is assembled there: the view's gradient is
// scattered into a flat buffer laid out like the shared storage, then read
// back through the input's own geometry.  A view that touches an element
// more than once sums every contribution to it; an input that touches one
// more than once shares it equally among its occurrences.  Both geometries
// are absolute on the storage, so a strided or offset input reads back the
// right elements.
namespace as_strided_bwd_detail {

// Whether a strided layout may visit an element twice: with the strides
// sorted, each must step past everything the smaller ones can reach.
inline bool maybe_overlapping(const std::vector<int64_t>& sizes,
                              const std::vector<int64_t>& strides) {
    std::vector<size_t> order(sizes.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return strides[a] < strides[b]; });
    int64_t reach = 0;
    for (size_t i : order) {
        if (strides[i] <= reach) return true;
        reach += strides[i] * (sizes[i] - 1);
    }
    return false;
}

// Elements a layout needs from the start of the buffer.
inline int64_t min_storage_size(const std::vector<int64_t>& sizes,
                                const std::vector<int64_t>& strides, int64_t offset) {
    int64_t size = offset + 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == 0) return offset;
        size += (sizes[i] - 1) * strides[i];
    }
    return size;
}

}  // namespace as_strided_bwd_detail

inline Tensor as_strided_backward(Tensor grad, const std::vector<int64_t>& input_sizes,
                                  const std::vector<int64_t>& input_strides,
                                  int64_t input_offset, const std::vector<int64_t>& view_sizes,
                                  const std::vector<int64_t>& view_strides,
                                  int64_t view_offset) {
    using namespace as_strided_bwd_detail;
    const auto empty_grad = [&] {
        return ops::zeros(input_sizes, grad.dtype(), grad.device());
    };
    // The view's geometry with unit dimensions dropped and broadcast
    // (stride-0) dimensions summed out of the gradient.
    std::vector<int64_t> out_sizes, out_strides;
    for (int64_t i = grad.dim() - 1; i >= 0; --i) {
        const int64_t size = view_sizes[static_cast<size_t>(i)];
        const int64_t stride = view_strides[static_cast<size_t>(i)];
        if (size == 0) return empty_grad();
        if (size == 1) {
            grad = ops::squeeze(grad, i);
        } else if (stride == 0) {
            grad = ops::sum(grad, std::vector<int64_t>{i}, false);
        } else {
            out_sizes.insert(out_sizes.begin(), size);
            out_strides.insert(out_strides.begin(), stride);
        }
    }
    std::vector<int64_t> in_sizes, in_strides;
    for (size_t i = 0; i < input_sizes.size(); ++i) {
        if (input_sizes[i] == 0) return empty_grad();
        if (input_sizes[i] != 1) {
            in_sizes.push_back(input_sizes[i]);
            in_strides.push_back(input_strides[i]);
        }
    }
    const bool out_overlap = maybe_overlapping(out_sizes, out_strides);
    const bool in_overlap = maybe_overlapping(in_sizes, in_strides);

    // The buffer starts at whichever window starts first.
    const int64_t shared_offset = std::min(input_offset, view_offset);
    const int64_t in_offset = input_offset - shared_offset;
    const int64_t out_offset = view_offset - shared_offset;
    const int64_t base_size =
        std::max(min_storage_size(in_sizes, in_strides, in_offset),
                 min_storage_size(out_sizes, out_strides, out_offset));
    Tensor storage = ops::zeros({base_size}, grad.dtype(), grad.device());

    // A recorded backward (create_graph) scatters with index_add so the
    // second derivative sees through it; so does an overlapping view, whose
    // repeated elements must accumulate.
    const bool record = GradMode::is_enabled();
    Tensor positions;
    if (out_overlap || in_overlap || record) {
        positions = ops::arange(Scalar(static_cast<int64_t>(0)), Scalar(base_size),
                                Scalar(static_cast<int64_t>(1)), DType::Int64,
                                grad.device());
    }
    if (out_overlap || record) {
        const Tensor out_index = ops::reshape(
            positions.as_strided(out_sizes, out_strides, out_offset), {-1});
        storage = ops::index_add(storage, 0, out_index, ops::reshape(grad, {-1}));
    } else {
        storage.as_strided(out_sizes, out_strides, out_offset).copy_(grad);
    }
    if (in_overlap) {
        const Tensor in_index = ops::reshape(
            positions.as_strided(in_sizes, in_strides, in_offset), {-1});
        const Tensor count = ops::index_add(
            ops::zeros({base_size}, grad.dtype(), grad.device()), 0, in_index,
            ops::expand(ops::ones({1}, grad.dtype(), grad.device()),
                        {in_index.numel()}));
        // Elements the input never reads divide 0 by 0; nothing reads them.
        storage = ops::div(storage, count);
    }
    return ops::as_strided(storage, input_sizes, input_strides, in_offset);
}

struct AsStridedBackward : public Node {
    std::vector<int64_t> input_sizes_;
    std::vector<int64_t> input_strides_;
    int64_t input_offset_;
    std::vector<int64_t> view_sizes_;
    std::vector<int64_t> view_strides_;
    int64_t view_offset_;

    // `storage_offset` is absolute on the storage; without one the view
    // starts where the input does.
    AsStridedBackward(const Tensor& self, std::vector<int64_t> size,
                      std::vector<int64_t> stride, std::optional<int64_t> storage_offset)
        : input_sizes_(static_cast<std::vector<int64_t>>(self.shape())),
          input_strides_(self.strides()),
          input_offset_(static_cast<int64_t>(
              self.unsafeGetTensorImpl()->storage_offset())),
          view_sizes_(std::move(size)),
          view_strides_(std::move(stride)),
          view_offset_(storage_offset.value_or(input_offset_)) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        return {as_strided_backward(inputs[0], input_sizes_, input_strides_, input_offset_,
                                    view_sizes_, view_strides_, view_offset_)};
    }
};

struct CopySlices : public Node {
    Tensor base_;
    std::vector<int64_t> view_size_;
    std::vector<int64_t> view_stride_;
    int64_t view_storage_offset_ = 0;
    std::function<Tensor(const Tensor&)> view_fn_;
    std::shared_ptr<Node> fn_;

    CopySlices(
        Tensor base,
        const Tensor& view,
        std::function<Tensor(const Tensor&)> view_fn,
        std::shared_ptr<Node> fn)
        : base_(std::move(base)),
          view_size_(static_cast<std::vector<int64_t>>(view.shape())),
          view_stride_(view.strides()),
          view_storage_offset_(static_cast<int64_t>(
              view.unsafeGetTensorImpl()->storage_offset())),
          view_fn_(std::move(view_fn)),
          fn_(std::move(fn)) {
        TP_CHECK(fn_ != nullptr, "CopySlices requires an inner backward node");
        add_next_edge_list(collect_next_edges(base_));
        const auto& inner_edges = fn_->next_edges();
        for (size_t i = 1; i < inner_edges.size(); ++i) {
            add_next_edge(inner_edges[i]);
        }
    }

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) {
            return variable_list(next_edges_.size());
        }
        TP_CHECK(fn_ != nullptr, backward_twice_message());

        const Tensor& grad = inputs[0];
        const auto base_size = static_cast<std::vector<int64_t>>(base_.shape());
        const auto base_stride = base_.strides();
        Tensor result = ops::empty_strided(
            base_size, base_stride, base_.dtype(), base_.device(), false);
        result.copy_(grad);

        Tensor grad_slice;
        if (view_fn_) {
            grad_slice = view_fn_(result);
        } else {
            const int64_t base_offset = static_cast<int64_t>(
                base_.unsafeGetTensorImpl()->storage_offset());
            const int64_t relative_offset = view_storage_offset_ - base_offset;
            grad_slice = result.as_strided(
                view_size_, view_stride_, relative_offset);
        }
        Tensor grad_slice_clone = grad_slice.clone();
        variable_list inner = fn_->apply({std::move(grad_slice_clone)});

        variable_list outputs(next_edges_.size());
        if (!inner.empty() && inner[0].defined()) {
            grad_slice.copy_(inner[0]);
            outputs[0] = std::move(result);
        }
        for (size_t i = 1; i < outputs.size() && i < inner.size(); ++i) {
            outputs[i] = std::move(inner[i]);
        }
        return outputs;
    }

    void release_variables() override {
        Node::release_variables();
        fn_.reset();
        base_ = Tensor();
    }
};

// NOTE(history): a hand-written ScaledDotProductAttentionBackward used to live
// here but was never instantiated -- the generated autograd node (from
// derivatives.yaml) is authoritative and avoids the double bookkeeping.
// mean(dtype=...) may accumulate in a wider dtype, but its derivative must be
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
// batched with broadcasting).  The hand-written node branches on dim(); every
// step composes dispatched recordable primitives, so create_graph sees a graph
// through `@` (double-backward).
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

        // A side whose gradient nobody wants costs no product.
        const bool want_self = should_compute_output(0);
        const bool want_other = should_compute_output(1);

        if (isComplexType(self.dtype())) {
            // The complex adjoint is the conjugate transpose.  The complex
            // branch delegates to the retained helper ops because its view
            // operations are not recordable yet.
            return {want_self ? ops::matmul_backward_self(grad, self, other) : Tensor(),
                    want_other ? ops::matmul_backward_other(grad, self, other) : Tensor()};
        }

        // Normalize vectors into matrix space before applying the batched
        // matrix formulas.
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
        // Broadcast-accumulate `g` down to `target` (using the kernels'
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

        Tensor grad_self;
        if (want_self) {
            Tensor grad_for_self = grad_m;
            if (grad_for_self.dtype() != other_m.dtype())
                grad_for_self = grad_for_self.to(other_m.dtype());
            grad_self = ops::matmul(grad_for_self, adjoint(other_m));
            grad_self = reduce_to(grad_self, self_m);
            if (self_vector) grad_self = ops::squeeze(grad_self, 0);
            if (grad_self.dtype() != self.dtype())
                grad_self = grad_self.to(self.dtype());
        }

        Tensor grad_other;
        if (want_other) {
            Tensor grad_for_other = grad_m;
            if (grad_for_other.dtype() != self_m.dtype())
                grad_for_other = grad_for_other.to(self_m.dtype());
            grad_other = ops::matmul(adjoint(self_m), grad_for_other);
            grad_other = reduce_to(grad_other, other_m);
            if (other_vector) grad_other = ops::squeeze(grad_other, -1);
            if (grad_other.dtype() != other.dtype())
                grad_other = grad_other.to(other.dtype());
        }

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

    // cat is single-output, so upstream grads always arrive at slot 0; sizing
    // the input buffer by next_edges would pad phantom slots per cat input.
    size_t num_inputs() const override { return 1; }

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
            grads.push_back(ops::slice(grad, dim, offset, offset + size, 1));
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
        m.device_type = t.device().type();
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
        // replaced by zeros.
        std::vector<Tensor> grads = detail::materialize_grads(inputs, shapes_, dtype_, device_);
        if (grads.empty()) return {Tensor()};
        return {ops::stack(grads, dim_)};
    }
};

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
        // grads replaced by zeros sized to each split.
        std::vector<Tensor> grads = detail::materialize_grads(inputs, shapes_, dtype_, device_);
        if (grads.empty()) return {Tensor()};
        return {ops::cat(grads, dim_)};
    }
};

// Roll backward negates each requested shift and preserves dimension order.
// TensorPlay's formula DSL cannot map over int64 lists, so the element-wise
// negation happens here and the node simply re-rolls the gradient.
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
            grads.push_back(ops::select(grad, dim, static_cast<int64_t>(i)));
        }
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        for (auto& saved : tensors_) saved.reset_data();
    }
};

// ===========================================================================
// Manual backward helpers: each pointwise Jacobian J is applied as grad *
// J.conj(), so the stored leaf
// conj over real dtypes is an alias (see conj_cpu), so the real training
// path stays copy-free.
// ===========================================================================

// Complex-gradient rule: if the forward output dtype is real but
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

// Complex multiplication backward: grad * other.conj()
template <typename T>
inline Tensor mul_tensor_backward(const Tensor& grad, const T& other,
                                  DType self_st) {
    Tensor scaled;
    if constexpr (std::is_same_v<T, Scalar>) {
        scaled = grad * scalar_conj_if_complex(other);
    } else {
        scaled = grad * (isComplexType(other.dtype()) ? ops::conj(other) : other);
    }
    return handle_r_to_c(self_st, std::move(scaled));
}

// Complex division backward: grad / other.conj()
template <typename T>
inline Tensor div_tensor_self_backward(const Tensor& grad, const T& other,
                                       DType self_st) {
    Tensor scaled;
    if constexpr (std::is_same_v<T, Scalar>) {
        scaled = grad / scalar_conj_if_complex(other);
    } else {
        scaled = grad / (isComplexType(other.dtype()) ? ops::conj(other) : other);
    }
    return handle_r_to_c(self_st, std::move(scaled));
}

// div_tensor_other_backward: -grad * conj((self / other) / other)
// The quotient is conjugated as a whole, including self.
inline Tensor div_tensor_other_backward(const Tensor& grad,
                                        const Tensor& self,
                                        const Tensor& other) {
    if (!isComplexType(self.dtype()) && !isComplexType(other.dtype())) {
        return handle_r_to_c(other.dtype(), -(grad * (self / other / other)));
    }
    return handle_r_to_c(
        other.dtype(), -(grad * ops::conj(self / other / other)));
}

// Rounded division is locally constant, so its gradient vanishes; only the
// unrounded form carries the quotient rule through.
template <typename T>
inline Tensor div_tensor_self_backward(const Tensor& grad, const T& other,
                                       DType self_st,
                                       const std::optional<std::string>& rounding_mode) {
    if (rounding_mode.has_value()) return ops::zeros_like(grad);
    return div_tensor_self_backward(grad, other, self_st);
}

inline Tensor div_tensor_other_backward(const Tensor& grad,
                                        const Tensor& self,
                                        const Tensor& other,
                                        const std::optional<std::string>& rounding_mode) {
    if (rounding_mode.has_value()) return ops::zeros_like(grad);
    return div_tensor_other_backward(grad, self, other);
}

// copysign only ever transplants a sign, so the incoming gradient is passed
// through with the sign it acquired.  A zero input has no sign to carry.
inline Tensor copysign_tensor_self_backward(const Tensor& grad,
                                            const Tensor& self,
                                            const Tensor& result) {
    return ops::where(ops::eq(self, Scalar(0)), ops::zeros_like(grad),
                      grad * (result / self));
}

// Scalar-exponent power backward: zero exponent short
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

// A derivative the operator does not provide: asking for it fails instead of
// handing back nothing, so a missing gradient is never mistaken for zero.
[[noreturn]] inline Tensor not_implemented_grad(const char* what) {
    TP_THROW(NotImplementedError, "the derivative for ", what, " is not implemented");
}

// pow_backward_exponent (number base): d(c^b)/db = c^b * log(c).  A zero
// base contributes nothing where the exponent is non-negative, the same
// convention as a zero tensor base.
inline Tensor pow_backward_exponent(const Tensor& grad, const Scalar& base,
                                    const Tensor& exponent,
                                    const Tensor& result) {
    const Tensor log_base =
        ops::log(ops::full({}, base, result.dtype(), result.device()));
    Tensor out = grad * ops::conj(result * log_base);
    if (!base.isComplex() && base.toDouble() == 0.0) {
        out = ops::where(ops::ge(exponent, Scalar(0)), ops::zeros_like(out), out);
    }
    return out;
}

// clamp with tensor bounds.  Inside the bounds the value passes through; a
// value sitting exactly on a bound shares its gradient with that bound, and
// when the bounds are reversed (min > max) the result is max everywhere, so
// max takes the whole gradient and min none.
inline Tensor clamp_tensor_backward_self(const Tensor& grad, const Tensor& self,
                                         const std::optional<Tensor>& min,
                                         const std::optional<Tensor>& max) {
    const bool has_min = min.has_value() && min->defined();
    const bool has_max = max.has_value() && max->defined();
    const Tensor zero = ops::zeros_like(grad);
    if (has_min && has_max) {
        const Tensor min_lt_max = ops::lt(*min, *max);
        const Tensor tie = ops::logical_and(
            ops::logical_or(ops::eq(self, *min), ops::eq(self, *max)), min_lt_max);
        const Tensor inactive =
            ops::logical_or(ops::lt(self, *min), ops::gt(self, *max));
        return ops::where(inactive, zero, ops::where(tie, grad / 2, grad));
    }
    if (has_min) {
        return ops::where(ops::lt(self, *min), zero,
                          ops::where(ops::eq(self, *min), grad / 2, grad));
    }
    if (has_max) {
        return ops::where(ops::gt(self, *max), zero,
                          ops::where(ops::eq(self, *max), grad / 2, grad));
    }
    return grad;
}

inline Tensor clamp_tensor_backward_min(const Tensor& grad, const Tensor& self,
                                        const std::optional<Tensor>& min,
                                        const std::optional<Tensor>& max) {
    const Tensor zero = ops::zeros_like(grad);
    if (max.has_value() && max->defined()) {
        const Tensor min_lt_max = ops::lt(*min, *max);
        const Tensor min_eq_max = ops::eq(*min, *max);
        const Tensor active = ops::logical_or(
            ops::logical_and(min_lt_max, ops::le(self, *min)),
            ops::logical_and(min_eq_max, ops::lt(self, *min)));
        return ops::where(active, ops::where(ops::eq(self, *min), grad / 2, grad), zero);
    }
    return ops::where(ops::gt(self, *min), zero,
                      ops::where(ops::eq(self, *min), grad / 2, grad));
}

inline Tensor clamp_tensor_backward_max(const Tensor& grad, const Tensor& self,
                                        const std::optional<Tensor>& min,
                                        const std::optional<Tensor>& max) {
    const Tensor zero = ops::zeros_like(grad);
    if (min.has_value() && min->defined()) {
        const Tensor min_lt_max = ops::lt(*min, *max);
        const Tensor min_eq_max = ops::eq(*min, *max);
        const Tensor active = ops::logical_or(
            ops::logical_or(ops::lt(*max, *min),
                            ops::logical_and(min_lt_max, ops::ge(self, *max))),
            ops::logical_and(min_eq_max, ops::gt(self, *max)));
        const Tensor split = ops::logical_and(ops::eq(self, *max), min_lt_max);
        return ops::where(active, ops::where(split, grad / 2, grad), zero);
    }
    return ops::where(ops::lt(self, *max), zero,
                      ops::where(ops::eq(self, *max), grad / 2, grad));
}

// log1p backward: grad / (self + 1).conj()
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

// Angle backward: zero at z == 0, otherwise
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

// Product backward fast path: exact when no element is zero;
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
// natives it delegates to), expressed as compositions of dispatched recordable
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

// Restore reduced dimensions by re-inserting size-1 dims removed by
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

// Scale a gradient by the number of contributing elements.
inline Tensor scale_grad_by_count(const Tensor& grad, const Tensor& mask,
                                  const std::vector<int64_t>& dims) {
    Tensor mask_f = mask.dtype() == grad.dtype() ? mask : mask.to(grad.dtype());
    return ops::mul(ops::div(grad, ops::sum(mask_f, dims, true)), mask_f);
}

// amax/amin backward: restore dims, mask the argmax
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

// logsumexp backward: with the reduced axes put back on both the gradient and
// the result, each input's share is exp(self - result) -- its softmax weight
// over those axes.  An empty list reduced every axis.
inline Tensor logsumexp_backward(const Tensor& grad, const Tensor& self,
                                 const Tensor& result, const std::vector<int64_t>& dim,
                                 bool keepdim) {
    if (self.dim() == 0) return ops::mul(grad, ops::exp(ops::sub(self, result)));
    std::vector<int64_t> dims = wrap_dims(dim, self.dim());
    if (dims.empty()) dims = all_dims(self.dim());
    const Tensor g = restore_reduced_dims(grad, dims, keepdim);
    const Tensor r = restore_reduced_dims(result, dims, keepdim);
    return ops::mul(g, ops::exp(ops::sub(self, r)));
}

// Evenly distribute a reduction gradient across contributing elements:
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

// accumulated with scatter_add.
inline Tensor cummaxmin_backward(const Tensor& grad, const Tensor& input,
                                 const Tensor& indices, int64_t dim) {
    if (input.numel() == 0) return input;
    const int64_t nd = input.dim();
    const int64_t d = dim < 0 ? dim + nd : dim;
    Tensor g = grad.dtype() == input.dtype() ? grad : grad.to(input.dtype());
    return ops::scatter_add(ops::zeros_like(input), d, indices, g);
}

// Sum backward: restore reduced dims, expand back.
inline Tensor sum_backward(const Tensor& grad,
                           const std::vector<int64_t>& sizes,
                           std::vector<int64_t> dims, bool keepdim) {
    if (sizes.empty()) return grad;
    dims = wrap_dims(dims, static_cast<int64_t>(sizes.size()));
    if (dims.empty()) dims = all_dims(static_cast<int64_t>(sizes.size()));
    return ops::expand(restore_reduced_dims(grad, dims, keepdim), sizes);
}

// NaN-aware sum backward.
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

// Norm backward (both arities). Real-dtype path; the
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

// Trilinear backward: each operand gradient reruns the contraction with the
// output gradient substituted for one input and that input's expand/sum
// roles swapped.  Built from the same dispatcher-level primitives the
// forward uses, so double backward records through the inner ops.
inline std::tuple<Tensor, Tensor, Tensor> _trilinear_backward(
    const Tensor& grad_out, const Tensor& i1, const Tensor& i2,
    const Tensor& i3, const std::vector<int64_t>& expand1,
    const std::vector<int64_t>& expand2, const std::vector<int64_t>& expand3,
    const std::vector<int64_t>& sumdim, std::array<bool, 3> grad_mask) {
    Tensor grad_i1, grad_i2, grad_i3;
    if (grad_mask[0]) {
        grad_i1 = ops::_trilinear(grad_out, i2, i3, sumdim, expand2, expand3,
                                  expand1, /*unroll_dim=*/1);
    }
    if (grad_mask[1]) {
        grad_i2 = ops::_trilinear(i1, grad_out, i3, expand1, sumdim, expand3,
                                  expand2, /*unroll_dim=*/1);
    }
    if (grad_mask[2]) {
        grad_i3 = ops::_trilinear(i1, i2, grad_out, expand1, expand2, sumdim,
                                  expand3, /*unroll_dim=*/1);
    }
    return {std::move(grad_i1), std::move(grad_i2), std::move(grad_i3)};
}

// Product backward with zeros: exclusive normal/reverse
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

// Product backward (full reduction): exact including the
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

// logcumsumexp backward (real branch): split positive /
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

// renorm backward, with linalg_vector_norm expressed as
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

// sinc backward.
inline Tensor sinc_backward(const Tensor& grad, const Tensor& self) {
    const double pi = 3.14159265358979323846;
    const Tensor self_pi = self * pi;
    const Tensor self_squared_pi = self * self * pi;
    const Tensor out = grad * ops::conj(
        (self_pi * ops::cos(self_pi) - ops::sin(self_pi)) / self_squared_pi);
    return ops::where(ops::eq(self_squared_pi, Scalar(0)),
                      ops::zeros_like(grad), out);
}

// take backward: flatten + accumulating put.
inline Tensor take_backward(const Tensor& grad, const Tensor& self,
                            const Tensor& indices) {
    const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    const Tensor flat = ops::reshape(self, {-1});
    const Tensor grad_flat = ops::index_add(
        ops::zeros_like(flat), 0, ops::reshape(indices, {-1}),
        ops::reshape(grad, {-1}));
    return ops::reshape(grad_flat, sizes);
}

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

inline Tensor trace_backward(const Tensor& grad, const Tensor& self) {
    return ops::eye(self.size(0), self.size(1), grad.dtype(), grad.device()) *
           grad;
}

// var backward (dim-list flavor; empty dims == full reduction), with TP's
// integer correction.
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

// std backward.
inline Tensor std_backward(const Tensor& result, const Tensor& grad,
                           const Tensor& self,
                           const std::vector<int64_t>& dims,
                           int64_t correction, bool keepdim) {
    const Tensor grad_var = ops::masked_fill(ops::div(grad, result * 2),
                                             ops::eq(result, Scalar(0)),
                                             Scalar(0));
    return var_backward(grad_var, self, dims, correction, keepdim);
}

// mean backward (sizes/dims flavor).
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
    // Scaling the reduced gradient before it is expanded divides each value
    // once rather than once per element it is broadcast to.
    return sum_backward(ops::div(grad, Scalar(static_cast<double>(count))), sizes, dims,
                        keepdim);
}

// self keeps grad except at overwritten positions (unless accumulate);
// values gather grad at the indexed positions.
inline std::tuple<Tensor, Tensor> index_put_backward(
        const Tensor& grad, bool accumulate,
        const std::vector<std::optional<Tensor>>& indices,
        const Tensor& values) {
    const Tensor grad_self = accumulate
        ? grad
        : ops::index_put(grad, indices, ops::zeros_like(values), false);
    return {grad_self, ops::index(grad, indices)};
}

// ===========================================================================
// Hand-written backward nodes for ops the formula DSL cannot express:
// list-gradient alignment (index_put), multiple differentiable outputs
// (aminmax, std_mean, var_mean).  apply() outputs are positionally aligned
// with the edges collected at record time (one per tensor argument, list
// elements included), so list slots are padded with undefined grads.
// ===========================================================================

struct IndexBackward : public Node {
    SavedVariable self_;
    std::vector<std::optional<SavedVariable>> indices_;

    IndexBackward(Tensor self, std::vector<std::optional<Tensor>> indices)
        : self_(std::move(self)) {
        indices_.reserve(indices.size());
        for (auto& index : indices) {
            if (index.has_value()) {
                indices_.emplace_back(std::move(*index));
            } else {
                indices_.emplace_back(std::nullopt);
            }
        }
    }

    variable_list apply(variable_list&& inputs) override {
        variable_list grads;
        grads.reserve(1 + indices_.size());
        const Tensor grad = inputs.empty() ? Tensor() : inputs[0];
        if (grad.defined()) {
            const Tensor self = self_.unpack();
            std::vector<std::optional<Tensor>> indices;
            indices.reserve(indices_.size());
            for (auto& index : indices_) {
                indices.emplace_back(index.has_value()
                    ? std::optional<Tensor>(index->unpack())
                    : std::nullopt);
            }
            Tensor zeros = ops::new_zeros(
                self, static_cast<std::vector<int64_t>>(self.shape()));
            grads.push_back(ops::_index_put_impl_(
                zeros, indices, grad, /*accumulate=*/true, /*unsafe=*/true));
        } else {
            grads.push_back(Tensor());
        }
        for (const auto& index : indices_) {
            if (index.has_value()) grads.push_back(Tensor());
        }
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        self_.reset_data();
        for (auto& index : indices_) {
            if (index.has_value()) index->reset_data();
        }
    }
};

// Shared by index_put, index_put_ and _index_put_impl_: outputs line up
// with the edges collected at record time (self, one per present index,
// values).
struct IndexPutBackward : public Node {
    std::vector<std::optional<SavedVariable>> indices_;
    SavedVariable values_;
    bool accumulate_;

    IndexPutBackward(std::vector<std::optional<Tensor>> indices, Tensor values,
                     bool accumulate)
        : values_(std::move(values)), accumulate_(accumulate) {
        indices_.reserve(indices.size());
        for (auto& index : indices) {
            if (index.has_value()) {
                indices_.emplace_back(std::move(*index));
            } else {
                indices_.emplace_back(std::nullopt);
            }
        }
    }

    variable_list apply(variable_list&& inputs) override {
        variable_list grads;
        grads.reserve(indices_.size() + 2);
        const Tensor grad = inputs.empty() ? Tensor() : inputs[0];
        Tensor grad_self;
        Tensor grad_values;
        if (grad.defined()) {
            std::vector<std::optional<Tensor>> indices;
            indices.reserve(indices_.size());
            for (auto& index : indices_) {
                indices.emplace_back(index.has_value()
                    ? std::optional<Tensor>(index->unpack())
                    : std::nullopt);
            }
            std::tie(grad_self, grad_values) = index_put_backward(
                grad, accumulate_, indices, values_.unpack());
        }
        grads.push_back(std::move(grad_self));
        for (const auto& index : indices_) {
            if (index.has_value()) grads.push_back(Tensor());
        }
        grads.push_back(std::move(grad_values));
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        for (auto& index : indices_) {
            if (index.has_value()) index->reset_data();
        }
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
#include "LinalgBackward.h"
#include "SpecialBackward.h"
#include "ActivationBackward.h"
#include "SparseSoftmaxBackward.h"
#include "LossBackward.h"
#include "ShapeKernelBackward.h"
#include "ConvNormBackward.h"
#include "MiscKernelBackward.h"
#include "GridSamplerBackward.h"
#include "SdpaBackward.h"
