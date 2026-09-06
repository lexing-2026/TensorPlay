// Backend-neutral kernels for the gradient/JVP helper operators that the
// config table declares but no backend implemented.  Every one of them is
// pure tensor algebra over already-dispatched primitives, so a single
// Composite-key registration serves CPU, CUDA and Vulkan alike: the inner
// ops resolve against the caller's own device and no per-backend math is
// duplicated here.

#include "CompositeCommon.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// index_select scatters each selected row back to the position `index` read
// it from; rows no index touched keep the zero they started with, and a
// repeated index accumulates.
Tensor index_select_backward_impl(const Tensor& grad,
                                  const std::vector<int64_t>& self_sizes,
                                  int64_t dim,
                                  const Tensor& index) {
    Tensor grad_input = ops::new_zeros(grad, self_sizes);
    return ops::index_add(grad_input, dim, index, grad);
}

// gather reads self[index] along `dim`, so its backward accumulates the
// gradient at those same coordinates.  sparse_grad asks for the result as a
// COO tensor holding only the touched coordinates.
Tensor gather_backward_impl(const Tensor& grad,
                            const Tensor& self,
                            int64_t dim,
                            const Tensor& index,
                            bool sparse_grad) {
    if (sparse_grad) {
        return ops::_gather_sparse_backward(self, dim, index, grad);
    }
    Tensor grad_input = ops::new_zeros(grad, static_cast<std::vector<int64_t>>(self.shape()));
    return ops::scatter_add(grad_input, dim, index, grad);
}

// masked_scatter consumed the leading `mask.count_nonzero()` elements of the
// source in order, so the source gradient is the masked selection of the
// output gradient, zero-padded back out to the source shape.
Tensor masked_scatter_backward_impl(const Tensor& grad_output,
                                    const Tensor& mask,
                                    const std::vector<int64_t>& sizes) {
    int64_t numel = 1;
    for (int64_t size : sizes) numel *= size;
    Tensor mask_selected = ops::masked_select(grad_output, mask);
    const int64_t diff_nelem = numel - mask_selected.numel();
    if (diff_nelem > 0) {
        // masked_select returns the 1-D run of selected elements; the
        // remaining source positions were never read and get a zero.
        Tensor zeros_fillin =
            ops::zeros({diff_nelem}, grad_output.dtype(), grad_output.device());
        mask_selected = ops::cat({mask_selected, zeros_fillin}, 0);
    }
    return ops::reshape(mask_selected, sizes);
}

// Backward of the reductions that select values rather than combine them
// (max.dim / min.dim / topk / mode): route the gradient to exactly the
// coordinates `indices` names.  Without keepdim the reduced axis was dropped
// from `grad`, so it has to be put back before the scatter.
Tensor value_selecting_reduction_backward_impl(const Tensor& grad,
                                               int64_t dim,
                                               const Tensor& indices,
                                               const std::vector<int64_t>& sizes,
                                               bool keepdim) {
    Tensor grad_out = grad;
    Tensor indices_ = indices;
    if (!keepdim && !sizes.empty()) {
        const int64_t wrapped = wrap_dim(dim, static_cast<int64_t>(sizes.size()));
        grad_out = ops::unsqueeze(grad, wrapped);
        indices_ = ops::unsqueeze(indices, wrapped);
    }
    Tensor grad_input = ops::zeros(sizes, grad.dtype(), grad.device());
    return ops::scatter(grad_input, dim, indices_, grad_out);
}

// Fold every leading axis of an activation into one row axis so the weight
// and bias gradients reduce with a plain 2-D gemm / column sum.
Tensor flatten_to_2d(const Tensor& t, int64_t features) {
    return ops::reshape(t, {-1, features});
}

std::tuple<Tensor, Tensor, Tensor> linear_backward_impl(
        const Tensor& self,
        const Tensor& grad_output,
        const Tensor& weight,
        const std::vector<bool>& output_mask) {
    TP_CHECK(output_mask.size() == 3,
             "linear_backward: output_mask must have 3 entries, got ",
             output_mask.size());
    if (!grad_output.defined()) {
        return {Tensor(), Tensor(), Tensor()};
    }
    TP_CHECK(weight.dim() == 2,
             "linear_backward: weight must be 2-dimensional, got ",
             weight.dim());
    const int64_t out_features = weight.size(0);
    const int64_t in_features = weight.size(1);
    const Tensor reshaped_grad = flatten_to_2d(grad_output, out_features);

    Tensor grad_input, grad_weight, grad_bias;
    if (output_mask[0]) {
        // grad_input = grad_output @ weight, restored to the input's shape.
        grad_input = ops::reshape(ops::mm(reshaped_grad, weight),
                                  static_cast<std::vector<int64_t>>(self.shape()));
    }
    if (output_mask[1]) {
        grad_weight = ops::mm(ops::t(reshaped_grad),
                              flatten_to_2d(self, in_features));
    }
    if (output_mask[2]) {
        grad_bias = ops::sum(reshaped_grad, {0}, false);
    }
    return {grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor> matmul_backward_impl(const Tensor& grad,
                                                const Tensor& self,
                                                const Tensor& other,
                                                const std::vector<bool>& mask) {
    TP_CHECK(mask.size() == 2,
             "matmul_backward: mask must have 2 entries, got ", mask.size());
    if (!grad.defined()) {
        return {Tensor(), Tensor()};
    }
    Tensor grad_self, grad_other;
    if (mask[0]) grad_self = ops::matmul_backward_self(grad, self, other);
    if (mask[1]) grad_other = ops::matmul_backward_other(grad, self, other);
    return {grad_self, grad_other};
}

// d/dx [x * Phi(x)] = Phi(x) + x * phi(x), evaluated in the promoted dtype so
// a half-precision input does not lose the erf/exp tail before the product.
Tensor infinitely_differentiable_gelu_backward_impl(const Tensor& grad,
                                                    const Tensor& self) {
    // 2/sqrt(pi) * 1/sqrt(2) * 1/2 == 1/sqrt(2*pi), the standard normal pdf
    // normalizer that scales the x * phi(x) term.
    constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
    const DType result_dtype = ops::result_type(grad, self);
    const DType math_dtype =
        result_dtype == DType::Float64 ? DType::Float64 : DType::Float32;
    const Tensor grad_ = grad.to(math_dtype);
    const Tensor self_ = self.to(math_dtype);
    const Tensor cdf = (ops::erf(self_ * Scalar(M_SQRT1_2)) + Scalar(1.0)) * Scalar(0.5);
    const Tensor pdf = ops::exp(self_ * self_ * Scalar(-0.5));
    return ((cdf + self_ * pdf * Scalar(kAlpha)) * grad_).to(result_dtype);
}

// d/dx log(1 + exp(-t*x)) = -t * exp(-t*x) / (1 + exp(-t*x)), averaged over
// the batch when the forward reduction was a mean.
Tensor soft_margin_loss_backward_impl(const Tensor& grad_output,
                                      const Tensor& self,
                                      const Tensor& target,
                                      int64_t reduction) {
    // reduction: 0 = none, 1 = mean, 2 = sum (Reduction::Mean == 1).
    const double norm = reduction == 1 ? 1.0 / static_cast<double>(self.numel()) : 1.0;
    const Tensor z = ops::exp(ops::neg(target) * self);
    return (target * z) * Scalar(-norm) / (z + Scalar(1.0)) * grad_output;
}

Tensor& soft_margin_loss_backward_grad_input(const Tensor& grad_output,
                                             const Tensor& self,
                                             const Tensor& target,
                                             int64_t reduction,
                                             Tensor& grad_input) {
    const Tensor value =
        soft_margin_loss_backward_impl(grad_output, self, target, reduction);
    grad_input.resize_(static_cast<std::vector<int64_t>>(value.shape()));
    grad_input.copy_(value);
    return grad_input;
}

// glu(x) = a * sigmoid(b) with x == cat(a, b) along `dim`.  The forward
// tangent is d(a) * sigmoid(b) + glu * (1 - sigmoid(b)) * d(b), the second
// term written through `glu` so the shared a*sigmoid(b) product is reused.
Tensor glu_jvp_impl(const Tensor& glu,
                    const Tensor& x,
                    const Tensor& dx,
                    int64_t dim) {
    const int64_t wrapped = wrap_dim(dim, x.dim());
    const int64_t glu_size = glu.size(wrapped);
    const Tensor b = ops::narrow(x, wrapped, glu_size, glu_size);
    const Tensor da = ops::narrow(dx, wrapped, 0, glu_size);
    const Tensor db = ops::narrow(dx, wrapped, glu_size, glu_size);
    const Tensor sig_b = ops::sigmoid(b);
    return da * sig_b + glu * (db - db * sig_b);
}

Tensor glu_backward_jvp_impl(const Tensor& grad_x,
                             const Tensor& grad_glu,
                             const Tensor& x,
                             const Tensor& dgrad_glu,
                             const Tensor& dx,
                             int64_t dim) {
    const int64_t wrapped = wrap_dim(dim, x.dim());
    const int64_t glu_size = grad_glu.size(wrapped);
    const Tensor a = ops::narrow(x, wrapped, 0, glu_size);
    const Tensor b = ops::narrow(x, wrapped, glu_size, glu_size);
    const Tensor da = ops::narrow(dx, wrapped, 0, glu_size);
    const Tensor db = ops::narrow(dx, wrapped, glu_size, glu_size);
    // grad_x splits the same way the input did: the first half is
    // grad_glu * sigmoid(b), the second grad_x_a * a * (1 - sigmoid(b)).
    const Tensor grad_x_a = ops::narrow(grad_x, wrapped, 0, glu_size);

    const Tensor sig_b = ops::sigmoid(b);
    const Tensor glu = a * sig_b;
    const Tensor db_neg_sig_b = db - db * sig_b;

    // d(grad_glu * sigmoid(b))
    const Tensor dgrad_x_a = dgrad_glu * sig_b + grad_x_a * db_neg_sig_b;
    // d(grad_x_a * a * (1 - sigmoid(b))), with a * (1 - sigmoid(b)) == a - glu.
    const Tensor dgrad_x_b =
        dgrad_x_a * (a - glu) + grad_x_a * (da - da * sig_b - glu * db_neg_sig_b);

    return ops::cat({dgrad_x_a, dgrad_x_b}, wrapped);
}

// For historical reasons to_dense() carries masked semantics: gradients for
// coordinates the sparse input never stored are dropped.  masked_grad=false
// asks for the unmasked reading, where the dense gradient is simply
// re-expressed in the input's layout.
Tensor to_dense_backward_impl(const Tensor& grad,
                              const Tensor& input,
                              std::optional<bool> masked_grad) {
    const bool masked = masked_grad.value_or(true);
    if (!input.is_sparse() && !input.is_sparse_compressed()) {
        return grad.to(input.dtype());
    }
    if (!input.is_sparse_compressed()) {
        // COO: autograd assumes the coalesced form, i.e. no duplicate values.
        return masked ? ops::sparse_mask(grad, ops::coalesce(input))
                      : ops::to_sparse(grad, input.sparse_dim());
    }
    // Compressed layouts: mask against the input's own COO reading, then put
    // the result back into the layout the input had.  sparse_mask answers in
    // COO, and the to_sparse_* entry points take a strided source, so the
    // masked reading is densified once in between.
    const Tensor source =
        masked ? ops::to_dense(ops::sparse_mask(grad, ops::to_sparse(input, input.sparse_dim())))
               : grad;
    if (input.is_sparse_csr()) return ops::to_sparse_csr(source);
    if (input.is_sparse_csc()) return ops::to_sparse_csc(source);
    const std::array<int64_t, 2> blocksize = input.sparse_blocksize();
    const std::vector<int64_t> bs = {blocksize[0], blocksize[1]};
    if (input.is_sparse_bsr()) return ops::to_sparse_bsr(source, bs);
    return ops::to_sparse_bsc(source, bs);
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Composite, GradientBridgeComposites) {
    m.impl("index_select_backward", index_select_backward_impl);
    m.impl("gather_backward", gather_backward_impl);
    m.impl("masked_scatter_backward", masked_scatter_backward_impl);
    m.impl("value_selecting_reduction_backward",
           value_selecting_reduction_backward_impl);
    m.impl("linear_backward", linear_backward_impl);
    m.impl("matmul_backward", matmul_backward_impl);
    m.impl("infinitely_differentiable_gelu_backward",
           infinitely_differentiable_gelu_backward_impl);
    m.impl("soft_margin_loss_backward", soft_margin_loss_backward_impl);
    m.impl("soft_margin_loss_backward.grad_input",
           soft_margin_loss_backward_grad_input);
    m.impl("glu_jvp", glu_jvp_impl);
    m.impl("glu_backward_jvp", glu_backward_jvp_impl);
    m.impl("to_dense_backward", to_dense_backward_impl);
}

} // namespace composite
} // namespace tensorplay
