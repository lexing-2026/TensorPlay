// Shape-only kernels for loss functions on the meta device.
//
// Losses combine two (or more) tensors elementwise and then optionally reduce
// the result to a scalar.  With reduction "none" the output keeps the
// broadcast geometry of the operands; with "mean" or "sum" it is a scalar.
// `nll_loss` additionally reports the total weight, a scalar, alongside the
// loss.  Only the geometry is observable here: the element values that the
// losses would combine do not exist on the meta device.

#include "MetaCommon.h"

#include "DType.h"
#include "DTypeNames.h"
#include "Scalar.h"
#include "TypePromotion.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

// The shape a reduced loss answers in: the broadcast operand geometry for
// reduction 0 (none), a scalar for mean/sum.
std::vector<int64_t> loss_out_shape(const std::vector<int64_t>& operand_shape,
                                    int64_t reduction) {
    if (reduction == 0) return operand_shape;
    if (reduction == 1 || reduction == 2) return {};
    TP_THROW(ValueError, "Invalid reduction mode");
}

std::vector<int64_t> broadcast_loss_shape(const Tensor& a, const Tensor& b,
                                          int64_t reduction) {
    return loss_out_shape(broadcast_shapes(shape_of(a), shape_of(b)), reduction);
}

// Whole-number operands answer a loss at single precision, because the loss
// arithmetic introduces a non-integral scalar.  Defined further down, next to
// the rest of the reduction helpers.
DType float_loss_dtype(DType dtype);

// ---------------------------------------------------------------------------
// Distance-loss gradients.  Each gradient blends the operand pair through a
// subtraction, a threshold blend and a multiply by the incoming gradient;
// the blend keeps the pair's promoted type, with two whole-number exits: the
// fractional threshold blend and the mean rescale run whole-number planes
// in single precision, and the incoming gradient joins the promotion.
// ---------------------------------------------------------------------------
std::vector<int64_t> loss_grad_shape(const Tensor& grad, const Tensor& input,
                                     const Tensor& target) {
    check_all_meta(input, target);
    check_all_meta(grad, grad);
    return broadcast_shapes(broadcast_shapes(shape_of(input), shape_of(target)),
                            shape_of(grad));
}

DType loss_grad_dtype(const Tensor& input, const Tensor& target,
                      const Tensor& grad, bool mean_rescale) {
    const DType pair = promoteTypes(input.dtype(), target.dtype());
    if (pair == DType::Bool) {
        TP_THROW(RuntimeError,
                 "Subtraction, the `-` operator, with two bool tensors is not "
                 "supported. Use the `^` or `logical_xor()` operator instead.");
    }
    DType out = pair;
    if (!isFloatingType(out) && !isComplexType(out)) out = DType::Float32;
    out = promoteTypes(out, grad.dtype());
    if (mean_rescale && !isFloatingType(out) && !isComplexType(out)) {
        out = DType::Float32;
    }
    return out;
}

Tensor smooth_l1_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                      const Tensor& target, int64_t reduction,
                                      double beta) {
    if (beta < 0) {
        TP_THROW(ValueError, "smooth_l1_loss does not support negative values for beta.");
    }
    return empty_meta(loss_grad_shape(grad_output, input, target),
                      loss_grad_dtype(input, target, grad_output, reduction == 1),
                      input.device());
}

Tensor huber_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                  const Tensor& target, int64_t reduction,
                                  double /*delta*/) {
    return empty_meta(loss_grad_shape(grad_output, input, target),
                      loss_grad_dtype(input, target, grad_output, reduction == 1),
                      input.device());
}

// ---------------------------------------------------------------------------
// Elementwise losses.
// ---------------------------------------------------------------------------

Tensor mse_loss_kernel(const Tensor& self, const Tensor& target,
                       int64_t reduction) {
    check_all_meta(self, target);
    // The squared difference runs in the pair's promoted type: a boolean pair
    // has no subtraction, and a whole-number pair answers its mean at single
    // precision (the sum keeps the pair type); every floating or complex
    // pair keeps its own type at every reduction.
    const DType pair = promoteTypes(self.dtype(), target.dtype());
    if (pair == DType::Bool) {
        TP_THROW(RuntimeError,
                 "Subtraction, the `-` operator, with two bool tensors is not "
                 "supported. Use the `^` or `logical_xor()` operator instead.");
    }
    DType out = pair;
    if (reduction == 1) {
        if (!isFloatingType(out) && !isComplexType(out)) out = DType::Float32;
    }
    return empty_meta(broadcast_loss_shape(self, target, reduction), out,
                      self.device());
}

Tensor binary_cross_entropy_kernel(
    const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction) {
    check_all_meta(self, target);
    std::vector<int64_t> out = broadcast_shapes(shape_of(self), shape_of(target));
    if (weight.has_value() && weight->defined()) {
        check_all_meta(self, *weight);
        out = broadcast_shapes(out, shape_of(*weight));
    }
    // The elementwise form blends three inputs: a logarithm of the input,
    // the target shifted by a fractional scalar, and an optional weight
    // multiply.  Whole-number inputs have no logarithm of their own type,
    // so every whole-number plane answers in single precision; floating and
    // complex planes keep their own type through all three blends.
    DType out_dt = self.dtype();
    if (!isFloatingType(out_dt) && !isComplexType(out_dt)) out_dt = DType::Float32;
    DType tgt = target.dtype();
    if (!isFloatingType(tgt) && !isComplexType(tgt)) tgt = DType::Float32;
    out_dt = promoteTypes(out_dt, tgt);
    if (weight.has_value() && weight->defined()) {
        out_dt = promoteTypes(out_dt, weight->dtype());
    }
    return empty_meta(loss_out_shape(out, reduction), out_dt, self.device());
}

Tensor smooth_l1_loss_kernel(const Tensor& self, const Tensor& target,
                             int64_t reduction, double beta) {
    if (beta < 0) {
        TP_THROW(ValueError,
                 "smooth_l1_loss does not support negative values for beta.");
    }
    check_all_meta(self, target);
    // Both branches of the elementwise switch blend in non-integral scalars,
    // so a whole-number pair answers in single precision at every reduction;
    // a boolean pair has no subtraction to feed the switch.
    const DType pair = promoteTypes(self.dtype(), target.dtype());
    if (pair == DType::Bool) {
        TP_THROW(RuntimeError,
                 "Subtraction, the `-` operator, with two bool tensors is not "
                 "supported. Use the `^` or `logical_xor()` operator instead.");
    }
    DType out = pair;
    if (!isFloatingType(out) && !isComplexType(out)) out = DType::Float32;
    return empty_meta(broadcast_loss_shape(self, target, reduction), out,
                      self.device());
}

Tensor l1_loss_kernel(const Tensor& input, const Tensor& target,
                      int64_t reduction) {
    check_all_meta(input, target);
    // The deviation is accumulated in floating literals, so whole-number
    // operands answer at single precision.
    return empty_meta(broadcast_loss_shape(input, target, reduction),
                      float_loss_dtype(input.dtype()), input.device());
}

Tensor kl_div_kernel(const Tensor& input, const Tensor& target,
                     int64_t reduction, bool /*log_target*/) {
    check_all_meta(input, target);
    return empty_meta(broadcast_loss_shape(input, target, reduction),
                      float_loss_dtype(input.dtype()), input.device());
}

Tensor poisson_nll_loss_kernel(const Tensor& input, const Tensor& target,
                               bool /*log_input*/, bool /*full*/, double /*eps*/,
                               int64_t reduction) {
    check_all_meta(input, target);
    return empty_meta(broadcast_loss_shape(input, target, reduction),
                      float_loss_dtype(input.dtype()), input.device());
}

// ---------------------------------------------------------------------------
// NLL loss: input (N, C) or (C), target (N) or a scalar.  The unreduced loss
// keeps the batch extent; reduced losses are scalars.  The total weight is
// always a scalar.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> nll_loss_kernel(
    const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction,
    int64_t /*ignore_index*/) {
    // The per-element walk reads the log probabilities at four element
    // widths; anything else has no form and is refused before the layout is
    // examined.
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64 &&
        self.dtype() != DType::BFloat16 && self.dtype() != DType::Half) {
        TP_THROW(NotImplementedError,
                 "nll_loss only supports Float32, Float64, BFloat16 and Half");
    }
    const bool no_batch = self.dim() == 1;
    const int64_t n_batch = no_batch ? 1 : self.size(0);
    const int64_t n_classes = self.size(no_batch ? 0 : 1);
    if (target.numel() != n_batch) {
        TP_THROW(ValueError, "nll_loss: expected target of ", n_batch,
                 " elements, got ", target.numel());
    }
    if (weight.has_value() && weight->defined() && weight->numel() != n_classes) {
        TP_THROW(RuntimeError,
                 "weight tensor should be defined either for all ", n_classes,
                 " classes or no classes");
    }
    const std::vector<int64_t> loss_shape =
        reduction == 0 ? (no_batch ? std::vector<int64_t>{} : std::vector<int64_t>{n_batch})
                       : std::vector<int64_t>{};
    return {empty_meta(loss_shape, self.dtype(), self.device()),
            empty_meta({}, self.dtype(), self.device())};
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Losses.
//
// The per-element loss is computed elsewhere; only the reduction decides the
// result shape, and a whole-number or truth-value promotion answers at single
// precision because the loss arithmetic blends in floating literals.
// ---------------------------------------------------------------------------
Tensor reduce_loss_meta(const std::vector<int64_t>& shape, DType dtype,
                        const Device& device, int64_t reduction) {
    if (reduction == 0) return empty_meta(shape, dtype, device);
    if (reduction == 1 || reduction == 2) return empty_meta({}, dtype, device);
    TP_THROW(ValueError, "Invalid reduction mode");
}

DType float_loss_dtype(DType dtype) {
    return isIntegralType(dtype, /*include_bool=*/true) ? DType::Float32 : dtype;
}

void check_reduction_mode(int64_t reduction, const char* name) {
    if (reduction != 0 && reduction != 1 && reduction != 2) {
        TP_THROW(ValueError, name, ": invalid reduction, expected 0 (none), 1 "
                                   "(mean) or 2 (sum) but got ", reduction);
    }
}

void check_float_loss_dtype(DType dtype, const char* impl) {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        TP_THROW(NotImplementedError, impl,
                 ": only supports Float32/Float64, got ",
                 scalarTypeName(dtype));
    }
}

Tensor huber_loss_kernel(const Tensor& self, const Tensor& target,
                         int64_t reduction, double delta) {
    if (delta <= 0) {
        TP_THROW(ValueError,
                 "huber_loss does not support non-positive values for delta.");
    }
    return reduce_loss_meta(
        broadcast2(self, target),
        float_loss_dtype(promoteTypes(self.dtype(), target.dtype())),
        self.device(), reduction);
}

Tensor soft_margin_loss_kernel(const Tensor& self, const Tensor& target,
                               int64_t reduction) {
    return reduce_loss_meta(
        broadcast2(self, target),
        float_loss_dtype(promoteTypes(self.dtype(), target.dtype())),
        self.device(), reduction);
}

Tensor hinge_embedding_loss_kernel(const Tensor& self, const Tensor& target,
                                   double /*margin*/, int64_t reduction) {
    // Both branches carry the input dtype, so the target only broadcasts.
    return reduce_loss_meta(broadcast2(self, target),
                            float_loss_dtype(self.dtype()), self.device(),
                            reduction);
}

Tensor margin_ranking_loss_kernel(const Tensor& input1, const Tensor& input2,
                                  const Tensor& target, double /*margin*/,
                                  int64_t reduction) {
    return reduce_loss_meta(
        broadcast_shapes(shape_of(input1), shape_of(input2), shape_of(target)),
        float_loss_dtype(promoteTypes(
            promoteTypes(input1.dtype(), input2.dtype()), target.dtype())),
        input1.device(), reduction);
}

Tensor cosine_embedding_loss_kernel(const Tensor& input1,
                                    const Tensor& input2,
                                    const Tensor& target, double /*margin*/,
                                    int64_t reduction) {
    const std::vector<int64_t> pair = broadcast2(input1, input2);
    std::vector<int64_t> cos_shape;
    if (pair.size() == 1) {
        TP_THROW(IndexError,
                 "Dimension out of range (expected to be in range of [-1, 0], "
                 "but got 1)");
    }
    if (pair.size() > 1) {
        cos_shape.assign(pair.begin(), pair.end());
        cos_shape.erase(cos_shape.begin() + 1);
    }
    return reduce_loss_meta(
        broadcast_shapes(cos_shape, shape_of(target)),
        float_loss_dtype(promoteTypes(input1.dtype(), input2.dtype())),
        input1.device(), reduction);
}

void multi_margin_shape_check(int64_t& nframe, int64_t& dim,
                              const Tensor& input, const Tensor& target,
                              const std::optional<Tensor>& weight) {
    const int64_t ndims = input.dim();
    if (!((ndims == 2 && input.size(1) != 0) ||
          (ndims == 1 && input.size(0) != 0) || ndims == 0)) {
        TP_THROW(RuntimeError,
                 "multi_margin_loss: Expected non-empty vector or matrix with "
                 "optional 0-dim batch size, but got: ",
                 input.shape().toString());
    }
    if (ndims <= 1) {
        nframe = 1;
        dim = ndims == 0 ? 1 : input.size(0);
    } else {
        nframe = input.size(0);
        dim = input.size(1);
    }
    if (!(target.dim() <= 1 && target.numel() == nframe)) {
        TP_THROW(RuntimeError,
                 "multi_margin_loss: target tensor should be 1-D with size "
                 "equal to the number of input samples (batch size). Expected "
                 "target size [", nframe, "], but got ",
                 target.shape().toString(), ". Input has shape ",
                 input.shape().toString(), ".");
    }
    if (weight.has_value() && weight->defined()) {
        if (!(weight->dim() <= 1 && weight->numel() == dim)) {
            TP_THROW(RuntimeError,
                     "multi_margin_loss: inconsistent weight size, expected ",
                     dim, " but got ", weight->shape().toString());
        }
    }
}

Tensor multi_margin_loss_kernel(const Tensor& self, const Tensor& target,
                                const Scalar& p, const Scalar& /*margin*/,
                                const std::optional<Tensor>& weight,
                                int64_t reduction) {
    check_reduction_mode(reduction, "multi_margin_loss");
    const int64_t power = p.to<int64_t>();
    if (power != 1 && power != 2) {
        TP_THROW(RuntimeError,
                 "multi_margin_loss: only p == 1 and p == 2 supported");
    }
    check_float_loss_dtype(self.dtype(), "multi_margin_loss_impl");
    int64_t nframe = 0, dim = 0;
    multi_margin_shape_check(nframe, dim, self, target, weight);
    if (reduction == 0 && target.dim() > 0) {
        return empty_meta({nframe}, self.dtype(), self.device());
    }
    return empty_meta({}, self.dtype(), self.device());
}

void multilabel_shape_check(int64_t& nframe, int64_t& dim,
                            const Tensor& input, const Tensor& target) {
    const int64_t ndims = input.dim();
    if (!((ndims == 2 && input.size(1) != 0) ||
          (ndims == 1 && input.size(0) != 0) || ndims == 0)) {
        TP_THROW(RuntimeError,
                 "multilabel_margin_loss: Expected non-empty vector or matrix "
                 "with optional 0-dim batch size, but got: ",
                 input.shape().toString());
    }
    if (ndims <= 1) {
        nframe = 1;
        dim = ndims == 0 ? 1 : input.size(0);
        if (!(target.dim() <= 1 && target.numel() == dim)) {
            TP_THROW(RuntimeError,
                     "multilabel_margin_loss: inconsistent target size: ",
                     target.shape().toString(), " for input of size: ",
                     input.shape().toString());
        }
    } else {
        nframe = input.size(0);
        dim = input.size(1);
        if (!(target.dim() == 2 && target.size(0) == nframe &&
              target.size(1) == dim)) {
            TP_THROW(RuntimeError,
                     "multilabel_margin_loss: inconsistent target size: ",
                     target.shape().toString(), " for input of size: ",
                     input.shape().toString());
        }
    }
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_kernel(
    const Tensor& self, const Tensor& target, int64_t reduction) {
    check_reduction_mode(reduction, "multilabel_margin_loss_forward");
    check_float_loss_dtype(self.dtype(), "multilabel_margin_loss_forward");
    int64_t nframe = 0, dim = 0;
    multilabel_shape_check(nframe, dim, self, target);
    const Tensor mask =
        empty_meta(shape_of(target), self.dtype(), self.device());
    if (reduction == 0 && target.dim() > 1) {
        return {empty_meta({nframe}, self.dtype(), self.device()), mask};
    }
    return {empty_meta({}, self.dtype(), self.device()), mask};
}

// ---------------------------------------------------------------------------
// Fixed-mean losses.  The plain loss names average their elementwise term
// over the whole broadcast geometry: the answer is always a scalar.  The
// arithmetic runs in double precision internally, so the answer comes back
// at the double type when the leading operand already has it and at single
// precision otherwise.
// ---------------------------------------------------------------------------

DType scalar_loss_dtype(DType dtype) {
    return dtype == DType::Float64 ? DType::Float64 : DType::Float32;
}

// Expanding a side operand onto the leading operand's geometry is validated
// even though no values are read: the side shape must broadcast onto the
// leading shape exactly.
void check_expandable_to(const Tensor& side,
                         const std::vector<int64_t>& target_shape,
                         const char* loss) {
    if (broadcast_shapes(shape_of(side), target_shape) != target_shape) {
        TP_THROW(RuntimeError, loss, ": operand of shape ",
                 side.shape().toString(),
                 " cannot be expanded to the leading operand's shape");
    }
}

// The row length is the last axis, so the unreduced answer is the paired
// geometry without that axis.
Tensor multilabel_soft_margin_loss_kernel(const Tensor& input,
                                          const Tensor& target,
                                          int64_t reduction) {
    check_reduction_mode(reduction, "multilabel_soft_margin_loss");
    check_all_meta(input, target);
    std::vector<int64_t> row_shape = broadcast2(input, target);
    if (!row_shape.empty()) row_shape.pop_back();
    return reduce_loss_meta(row_shape, scalar_loss_dtype(input.dtype()),
                            input.device(), reduction);
}

// These losses expand their side operands onto the leading operand's
// geometry, which is the validated step.
Tensor binary_cross_entropy_with_logits_kernel(
    const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& pos_weight, int64_t reduction) {
    check_reduction_mode(reduction, "binary_cross_entropy_with_logits");
    check_all_meta(self, target);
    check_expandable_to(target, shape_of(self),
                        "binary_cross_entropy_with_logits");
    // A defined-but-empty weighting tensor takes no part in the loss, so it
    // escapes the expansion check the same way it escapes the arithmetic.
    for (const std::optional<Tensor>* w : {&weight, &pos_weight}) {
        if (w->has_value() && (*w)->defined() && (*w)->numel() > 0) {
            check_all_meta(self, **w);
            check_expandable_to(**w, shape_of(self),
                                "binary_cross_entropy_with_logits");
        }
    }
    return reduce_loss_meta(shape_of(self), scalar_loss_dtype(self.dtype()),
                            self.device(), reduction);
}

// The embedding losses read their row geometry straight off the leading
// operand and index the side operands positionally, so no cross-operand
// broadcast is validated.
// The distances run along the last axis of the three joined operands, so the
// unreduced answer is that broadcast geometry without its last axis.  The
// dimension count of the three operands must agree: their rows are compared
// position by position.
Tensor triplet_margin_loss_kernel(const Tensor& anchor, const Tensor& positive,
                                  const Tensor& negative, double /*margin*/,
                                  double /*p*/, double /*eps*/, bool /*swap*/,
                                  int64_t reduction) {
    check_reduction_mode(reduction, "triplet_margin_loss");
    check_all_meta(anchor, positive);
    check_all_meta(anchor, negative);
    const int64_t a_dim = anchor.dim();
    const int64_t p_dim = positive.dim();
    const int64_t n_dim = negative.dim();
    if (a_dim != p_dim || p_dim != n_dim) {
        TP_THROW(RuntimeError,
                 "The anchor, positive, and negative tensors are expected to "
                 "have the same number of dimensions, but got: anchor ", a_dim,
                 "D, positive ", p_dim, "D, and negative ", n_dim, "D inputs");
    }
    std::vector<int64_t> row_shape = broadcast_shapes(
        broadcast_shapes(shape_of(anchor), shape_of(positive)),
        shape_of(negative));
    if (!row_shape.empty()) row_shape.pop_back();
    return reduce_loss_meta(row_shape, scalar_loss_dtype(anchor.dtype()),
                            anchor.device(), reduction);
}

// A single operand that has to be planned on the meta device.
void check_meta_operand(const Tensor& t) {
    if (!t.device().is_meta()) {
        TP_THROW(DeviceMismatchError, "expected a meta tensor but got ",
                 t.device().toString());
    }
}

// Both logits gradients answer one operand's gradient over the geometry every
// side operand broadcasts into; only the answer's width follows the operand the
// gradient is taken with respect to.
std::vector<int64_t> logits_grad_shape(const Tensor& grad_output,
                                       const Tensor& self,
                                       const Tensor& target,
                                       const std::optional<Tensor>& weight,
                                       const std::optional<Tensor>& pos_weight) {
    check_meta_operand(grad_output);
    std::vector<int64_t> shape =
        broadcast_shapes(shape_of(self), shape_of(target));
    for (const std::optional<Tensor>* side : {&weight, &pos_weight}) {
        if (side->has_value() && (*side)->defined() && (*side)->numel() > 0) {
            check_all_meta(self, **side);
            shape = broadcast_shapes(shape, shape_of(**side));
        }
    }
    return broadcast_shapes(shape, shape_of(grad_output));
}

Tensor binary_cross_entropy_with_logits_backward_kernel(
    const Tensor& grad_output, const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& pos_weight,
    int64_t reduction) {
    check_reduction_mode(reduction, "binary_cross_entropy_with_logits_backward");
    check_all_meta(self, target);
    return empty_meta(logits_grad_shape(grad_output, self, target, weight,
                                        pos_weight),
                      float_loss_dtype(self.dtype()), self.device());
}

Tensor binary_cross_entropy_with_logits_target_backward_kernel(
    const Tensor& grad_output, const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& pos_weight,
    int64_t reduction) {
    check_reduction_mode(reduction,
                         "binary_cross_entropy_with_logits_target_backward");
    check_all_meta(self, target);
    return empty_meta(logits_grad_shape(grad_output, self, target, weight,
                                        pos_weight),
                      float_loss_dtype(target.dtype()), target.device());
}

// A row answer hands one gradient per lane, so the input gradient keeps the
// paired geometry whatever the reduction did to the answer.
Tensor multilabel_soft_margin_loss_backward_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& target,
    int64_t reduction) {
    check_reduction_mode(reduction, "multilabel_soft_margin_loss_backward");
    check_all_meta(input, target);
    check_meta_operand(grad_output);
    return empty_meta(broadcast2(input, target),
                      float_loss_dtype(input.dtype()), input.device());
}

Tensor multilabel_soft_margin_loss_target_backward_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& target,
    int64_t reduction) {
    check_reduction_mode(reduction,
                         "multilabel_soft_margin_loss_target_backward");
    check_all_meta(input, target);
    check_meta_operand(grad_output);
    return empty_meta(broadcast2(input, target),
                      float_loss_dtype(target.dtype()), target.device());
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaLoss) {
    m.impl("mse_loss", mse_loss_kernel);
    m.impl("binary_cross_entropy", binary_cross_entropy_kernel);
    m.impl("smooth_l1_loss", smooth_l1_loss_kernel);
    m.impl("tp_l1_loss", l1_loss_kernel);
    m.impl("tp_kl_div", kl_div_kernel);
    m.impl("tp_poisson_nll_loss", poisson_nll_loss_kernel);
    m.impl("nll_loss_forward", nll_loss_kernel);
    m.impl("huber_loss", huber_loss_kernel);
    m.impl("huber_loss_backward", huber_loss_backward_kernel);
    m.impl("smooth_l1_loss_backward", smooth_l1_loss_backward_kernel);
    m.impl("tp_soft_margin_loss", soft_margin_loss_kernel);
    m.impl("tp_hinge_embedding_loss", hinge_embedding_loss_kernel);
    m.impl("tp_margin_ranking_loss", margin_ranking_loss_kernel);
    m.impl("tp_cosine_embedding_loss", cosine_embedding_loss_kernel);
    m.impl("multi_margin_loss", multi_margin_loss_kernel);
    m.impl("multilabel_margin_loss_forward",
           multilabel_margin_loss_forward_kernel);
    m.impl("multilabel_soft_margin_loss",
           multilabel_soft_margin_loss_kernel);
    m.impl("binary_cross_entropy_with_logits",
           binary_cross_entropy_with_logits_kernel);
    m.impl("binary_cross_entropy_with_logits_backward",
           binary_cross_entropy_with_logits_backward_kernel);
    m.impl("binary_cross_entropy_with_logits_target_backward",
           binary_cross_entropy_with_logits_target_backward_kernel);
    m.impl("multilabel_soft_margin_loss_backward",
           multilabel_soft_margin_loss_backward_kernel);
    m.impl("multilabel_soft_margin_loss_target_backward",
           multilabel_soft_margin_loss_target_backward_kernel);
    m.impl("triplet_margin_loss", triplet_margin_loss_kernel);
    m.impl("l1_loss", l1_loss_kernel);
    m.impl("kl_div", kl_div_kernel);
    m.impl("margin_ranking_loss", margin_ranking_loss_kernel);
    m.impl("hinge_embedding_loss", hinge_embedding_loss_kernel);
    m.impl("soft_margin_loss", soft_margin_loss_kernel);
    m.impl("cosine_embedding_loss", cosine_embedding_loss_kernel);
    m.impl("poisson_nll_loss", poisson_nll_loss_kernel);
}

} // namespace meta
} // namespace tensorplay
