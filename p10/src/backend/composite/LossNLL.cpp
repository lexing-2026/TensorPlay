#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor nll_loss_nd_native(const Tensor& self, const Tensor& target,
                          const std::optional<Tensor>& weight,
                          int64_t reduction, int64_t ignore_index) {
    if (self.dim() < 1) {
        TP_THROW(ValueError, "Expected input with at least one dimension");
    }
    if (self.dim() != 1) {
        TP_CHECK(target.dim() > 0,
                 "Expected target with a batch dimension");
        TP_CHECK(self.size(0) == target.size(0),
                 "Expected input and target batch sizes to match");
    }

    if (self.dim() == 1 || self.dim() == 2) {
        return std::get<0>(ops::nll_loss(
            self, target, weight, reduction, ignore_index));
    }
    if (self.dim() == 4) {
        return std::get<0>(ops::nll_loss2d(
            self, target, weight, reduction, ignore_index));
    }

    const int64_t n = self.size(0);
    const int64_t c = self.size(1);
    std::vector<int64_t> expected_target_shape;
    expected_target_shape.reserve(static_cast<size_t>(self.dim() - 1));
    expected_target_shape.push_back(n);
    for (int64_t d = 2; d < self.dim(); ++d) {
        expected_target_shape.push_back(self.size(d));
    }
    TP_CHECK(static_cast<std::vector<int64_t>>(target.shape()) ==
                 expected_target_shape,
             "Expected target shape to match input spatial dimensions");

    Tensor input_view = self.contiguous();
    Tensor target_view = target.contiguous();
    if (input_view.numel() > 0) {
        input_view = input_view.view({n, c, 1, -1});
    } else {
        input_view = input_view.view({n, c, 0, 0});
    }
    if (target_view.numel() > 0) {
        target_view = target_view.view({n, 1, -1});
    } else {
        target_view = target_view.view({n, 0, 0});
    }

    Tensor result = std::get<0>(ops::nll_loss2d(
        input_view, target_view, weight, reduction, ignore_index));
    if (reduction != 0) {
        return result;
    }
    return result.view(expected_target_shape);
}

// ---------------------------------------------------------------------------
// Cross entropy
//
// Three forward regimes:
//   - soft targets (input and target shapes identical): the loss reduces
//     -(log_softmax(input) * target) per class dimension, with optional
//     per-class weights and label smoothing folded into the target;
//   - hard targets with label smoothing: the plain nll term plus a
//     uniform-mixture term weighted by smoothing / n_classes;
//   - hard targets without smoothing: nll over log_softmax directly.
//
// Reductions: 0 = none, 1 = mean, 2 = sum.
// ---------------------------------------------------------------------------

namespace {

constexpr int64_t kReductionNone = 0;
constexpr int64_t kReductionMean = 1;
constexpr int64_t kReductionSum = 2;

int64_t cross_entropy_class_dim(const Tensor& self) {
    return self.dim() == 1 ? 0 : 1;
}

// Soft-target branch: `target` holds class probabilities with the same rank
// as the input.
Tensor cross_entropy_loss_prob_target(
    const Tensor& self, const Tensor& target_, const Tensor& weight,
    int64_t reduction, double label_smoothing) {
    const int64_t class_dim = cross_entropy_class_dim(self);
    const int64_t n_classes = self.size(class_dim);
    TP_CHECK(weight.numel() == 0 ||
                 (weight.dim() == 1 && weight.numel() == n_classes),
             "cross_entropy: weight tensor should be defined either for all ",
             n_classes, " classes or no classes");

    Tensor input = ops::log_softmax(self, class_dim, DType::Undefined);
    Tensor target = target_;
    if (label_smoothing > 0.0) {
        TP_CHECK(label_smoothing <= 1.0,
                 "label_smoothing must be between 0.0 and 1.0. Got: ",
                 label_smoothing);
        target = target * Scalar(1 - label_smoothing) +
                 Scalar(label_smoothing / n_classes);
    }

    if (weight.defined() && weight.numel() > 0) {
        // Expand the class weights so they broadcast against the input and
        // target ranks.
        Tensor weight_ = weight;
        if (input.dim() > 1) {
            std::vector<int64_t> weight_broadcast_shape(
                static_cast<size_t>(input.dim()), 1);
            weight_broadcast_shape[1] = weight.size(0);
            weight_ = weight.view(weight_broadcast_shape);
        }
        switch (reduction) {
            case kReductionMean: {
                if (input.numel() == 0) {
                    Tensor nan = ops::neg((input * target * weight_).sum());
                    ops::fill_(nan, Scalar(
                        std::numeric_limits<double>::quiet_NaN()));
                    return nan;
                }
                return ops::neg((input * target * weight_).sum()) /
                       Scalar(input.numel() / n_classes);
            }
            case kReductionSum:
                return ops::neg((input * target * weight_).sum());
            case kReductionNone:
                return ops::neg((input * target * weight_).sum({class_dim}));
            default:
                TP_THROW(RuntimeError,
                         "Invalid reduction type encountered in cross_entropy: ",
                         reduction);
        }
    }
    switch (reduction) {
        case kReductionMean: {
            if (input.numel() == 0) {
                Tensor nan = ops::neg((input * target).sum());
                ops::fill_(nan, Scalar(
                    std::numeric_limits<double>::quiet_NaN()));
                return nan;
            }
            return ops::neg((input * target).sum()) /
                   Scalar(input.numel() / n_classes);
        }
        case kReductionSum:
            return ops::neg((input * target).sum());
        case kReductionNone:
            return ops::neg((input * target).sum({class_dim}));
        default:
            TP_THROW(RuntimeError,
                     "Invalid reduction type encountered in cross_entropy: ",
                     reduction);
    }
}

// Hard-target branch with label smoothing: an nll term over the log softmax
// plus a uniform-mixture term over every class.
Tensor cross_entropy_loss_label_smoothing(
    const Tensor& self, const Tensor& target, const Tensor& weight,
    int64_t reduction, int64_t ignore_index, double label_smoothing) {
    const int64_t class_dim = cross_entropy_class_dim(self);
    Tensor input = ops::log_softmax(self, class_dim, DType::Undefined);
    Tensor nllloss = nll_loss_nd_native(
        input, target, weight, reduction, ignore_index);

    const int64_t n_classes = input.size(class_dim);

    Tensor smooth_loss;
    if (weight.defined() && weight.numel() > 0) {
        // Expand the class weights so they broadcast against the input rank.
        std::vector<int64_t> weight_broadcast_shape(
            static_cast<size_t>(input.dim()), 1);
        weight_broadcast_shape[class_dim] = weight.size(0);
        Tensor weight_ = weight.view(weight_broadcast_shape);
        smooth_loss = ops::neg((input * weight_).sum({class_dim}));
    } else {
        smooth_loss = ops::neg(input.sum({class_dim}));
    }

    Tensor ignore_mask = ops::eq(target, Scalar(static_cast<double>(ignore_index)));
    smooth_loss = ops::masked_fill(smooth_loss, ignore_mask,
                                   Scalar(0.0));

    Tensor ret;
    switch (reduction) {
        case kReductionMean: {
            if (weight.defined() && weight.numel() > 0) {
                // The loss is normalized by the selected weights so the
                // result stays consistent with the plain nll path.
                Tensor not_ignored = ops::logical_not(ignore_mask);
                Tensor selected = ops::flatten(ops::masked_select(target, not_ignored));
                ret = ops::sum(smooth_loss) / ops::sum(ops::gather(weight, 0, selected));
            } else {
                Tensor true_mask = ops::logical_not(ignore_mask);
                ret = ops::sum(smooth_loss) / ops::sum(true_mask);
            }
            break;
        }
        case kReductionSum:
            ret = ops::sum(smooth_loss);
            break;
        case kReductionNone:
            ret = smooth_loss;
            break;
        default:
            TP_THROW(RuntimeError,
                     "Invalid reduction type encountered in cross_entropy: ",
                     reduction);
    }
    return nllloss * Scalar(1 - label_smoothing) +
           ret * Scalar(label_smoothing / n_classes);
}

}  // namespace

Tensor cross_entropy_loss_native(
    const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction,
    int64_t ignore_index, double label_smoothing) {
    if (self.shape() == target.shape()) {
        // Identical shapes signal class-probability targets.
        TP_CHECK(tensorplay::isFloatingType(target.dtype()),
                 "Expected floating point type for target with class "
                 "probabilities, got ", target.dtype());
        TP_CHECK(ignore_index < 0,
                 "ignore_index is not supported for floating point target");
        const Tensor weight_ = weight.value_or(Tensor());
        return cross_entropy_loss_prob_target(
            self, target, weight_, reduction, label_smoothing);
    }
    if (label_smoothing > 0.0) {
        TP_CHECK(label_smoothing <= 1.0,
                 "label_smoothing must be between 0.0 and 1.0. Got: ",
                 label_smoothing);
        const Tensor weight_ = weight.value_or(Tensor());
        return cross_entropy_loss_label_smoothing(
            self, target, weight_, reduction, ignore_index, label_smoothing);
    }
    const int64_t class_dim = cross_entropy_class_dim(self);
    return nll_loss_nd_native(
        ops::log_softmax(self, class_dim, DType::Undefined),
        target, weight, reduction, ignore_index);
}

}  // namespace composite
}  // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Composite, LossNLL) {
    using namespace tensorplay::composite;
    m.impl("nll_loss_nd", nll_loss_nd_native);
    m.impl("cross_entropy_loss", cross_entropy_loss_native);
}
