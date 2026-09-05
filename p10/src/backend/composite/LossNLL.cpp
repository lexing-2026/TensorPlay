#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

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

}  // namespace composite
}  // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Composite, LossNLL) {
    using namespace tensorplay::composite;
    m.impl("nll_loss_nd", nll_loss_nd_native);
}
