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

}
}
