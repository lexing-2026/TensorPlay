// Composite kernel: conv_tbc.
// convolution.  Equivalent to a strided-1 padded conv1d on the permuted NCT
// layout (both are cross-correlations with the same boundary handling).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <optional>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor conv_tbc_native(const Tensor& self, const Tensor& weight,
                       const Tensor& bias, int64_t pad) {
    if (self.dim() != 3) {
        TP_THROW(RuntimeError,
                 "Input must be 3 dims: time, batch, in_channel");
    }
    if (weight.dim() != 3) {
        TP_THROW(RuntimeError,
                 "Weight tensor must have 3 dims: kernel_width, in_channels, out_channels.");
    }
    if (bias.dim() != 1) {
        TP_THROW(RuntimeError, "Bias must be 1-D");
    }
    if (self.size(2) != weight.size(1)) {
        TP_THROW(RuntimeError,
                 "input channel size and weight input channel size do not match");
    }
    if (weight.size(2) != bias.size(0)) {
        TP_THROW(RuntimeError,
                 "output channel size and weight output channel size do not match");
    }
    const Tensor input_nct = ops::permute(self, {1, 2, 0});
    const Tensor weight_oct = ops::permute(weight, {2, 1, 0});
    const Tensor out = ops::conv1d(input_nct, weight_oct,
                                   std::optional<Tensor>(bias),
                                   std::vector<int64_t>{1},
                                   std::vector<int64_t>{pad},
                                   std::vector<int64_t>{1},
                                   1);
    return ops::permute(out, {2, 0, 1});
}

TENSORPLAY_LIBRARY_IMPL(Composite, ConvolutionTBCComposite) {
    m.impl("conv_tbc", conv_tbc_native);
}

} // namespace composite
} // namespace tensorplay
