// Composite kernel: bilinear.
//   out[..., o] = sum_ij x1[..., i] * W[o, i, j] * x2[..., j]
// (bilinear is the broadcast product of the two input projections; the
// contraction is the same one _trilinear performs for the 3-input case).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor bilinear_native(const Tensor& input1, const Tensor& input2,
                       const Tensor& weight,
                       const std::optional<Tensor>& bias) {
    if (input1.dim() != input2.dim()) {
        TP_THROW(RuntimeError,
                 "bilinear(): input1 and input2 must have the same number of dimensions");
    }
    if (input1.dim() < 1) {
        TP_THROW(RuntimeError, "bilinear(): inputs must have at least 1 dimension");
    }
    if (weight.dim() != 3) {
        TP_THROW(RuntimeError, "bilinear(): weight must be 3 dimensions");
    }
    for (int64_t d = 0; d + 1 < input1.dim(); ++d) {
        if (input1.size(d) != input2.size(d)) {
            TP_THROW(RuntimeError,
                     "bilinear(): input1 and input2 must have the same batch sizes");
        }
    }
    if (input1.size(-1) != weight.size(1) || input2.size(-1) != weight.size(2)) {
        TP_THROW(RuntimeError,
                 "bilinear(): input dimensions and weight dimensions do not match");
    }
    if (bias.has_value() && bias->dim() == 1 && bias->size(0) != weight.size(0)) {
        TP_THROW(RuntimeError,
                 "bilinear(): bias size and weight output size do not match");
    }

    const int64_t out_features = weight.size(0);
    const Tensor x1 = ops::reshape(input1, {-1, input1.size(-1)});
    const Tensor x2 = ops::reshape(input2, {-1, input2.size(-1)});
    // z[b, o, i] = sum_j W[o, i, j] * x2[b, j]
    // x2 -> (B, 1, 1, J), weight -> (1, O, I, J) -> product (B, O, I, J).
    const Tensor z = ops::sum(
        ops::mul(ops::unsqueeze(ops::unsqueeze(x2, 1), 1),
                 ops::unsqueeze(weight, 0)),
        {3}, false);
    // out[b, o] = sum_i z[b, o, i] * x1[b, i]
    Tensor out = ops::sum(ops::mul(z, ops::unsqueeze(x1, 1)), {2}, false);
    if (bias.has_value()) out = ops::add(out, *bias);

    const auto in_shape = input1.shape();
    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 1);
    out_shape.push_back(out_features);
    return ops::reshape(out, out_shape);
}

TENSORPLAY_LIBRARY_IMPL(Composite, LinearComposite) {
    m.impl("bilinear", bilinear_native);
}

} // namespace composite
} // namespace tensorplay
