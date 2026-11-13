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

// ---------------------------------------------------------------------------
// Elementwise losses.
// ---------------------------------------------------------------------------

Tensor mse_loss_kernel(const Tensor& self, const Tensor& target,
                       int64_t reduction) {
    check_all_meta(self, target);
    return empty_meta(broadcast_loss_shape(self, target, reduction),
                      self.dtype(), self.device());
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
    return empty_meta(loss_out_shape(out, reduction), self.dtype(),
                      self.device());
}

Tensor smooth_l1_loss_kernel(const Tensor& self, const Tensor& target,
                             int64_t reduction, double beta) {
    if (beta < 0) {
        TP_THROW(ValueError,
                 "smooth_l1_loss does not support negative values for beta.");
    }
    check_all_meta(self, target);
    return empty_meta(broadcast_loss_shape(self, target, reduction),
                      self.dtype(), self.device());
}

Tensor l1_loss_kernel(const Tensor& input, const Tensor& target,
                      int64_t reduction) {
    check_all_meta(input, target);
    return empty_meta(broadcast_loss_shape(input, target, reduction),
                      input.dtype(), input.device());
}

Tensor kl_div_kernel(const Tensor& input, const Tensor& target,
                     int64_t reduction, bool /*log_target*/) {
    check_all_meta(input, target);
    return empty_meta(broadcast_loss_shape(input, target, reduction),
                      input.dtype(), input.device());
}

Tensor poisson_nll_loss_kernel(const Tensor& input, const Tensor& target,
                               bool /*log_input*/, bool /*full*/, double /*eps*/,
                               int64_t reduction) {
    check_all_meta(input, target);
    return empty_meta(broadcast_loss_shape(input, target, reduction),
                      input.dtype(), input.device());
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

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaLoss) {
    m.impl("mse_loss", mse_loss_kernel);
    m.impl("binary_cross_entropy", binary_cross_entropy_kernel);
    m.impl("smooth_l1_loss", smooth_l1_loss_kernel);
    m.impl("tp_l1_loss", l1_loss_kernel);
    m.impl("tp_kl_div", kl_div_kernel);
    m.impl("tp_poisson_nll_loss", poisson_nll_loss_kernel);
    m.impl("nll_loss", nll_loss_kernel);
}

} // namespace meta
} // namespace tensorplay