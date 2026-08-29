// Composite kernel: native_channel_shuffle.
// (the op name keeps the historical "Chanel" spelling).

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor native_channel_shuffle_native(const Tensor& self, int64_t groups) {
    if (self.dim() <= 2) {
        TP_THROW(RuntimeError,
                 "channel_shuffle expects input with > 2 dims, but got input with ",
                 self.dim(), " dims");
    }
    if (groups <= 0) {
        TP_THROW(RuntimeError,
                 "channel_shuffle expects groups to be strictly positive");
    }
    const int64_t b = self.size(0);
    const int64_t c = self.size(1);
    if (c % groups != 0) {
        TP_THROW(RuntimeError,
                 "Number of channels must be divisible by groups. Got ", c,
                 " channels and ", groups, " groups.");
    }
    const int64_t channels_per_group = c / groups;
    const Tensor reshaped = ops::reshape(self, {b, groups, channels_per_group, -1});
    const Tensor shuffled = ops::contiguous(
        ops::permute(reshaped, {0, 2, 1, 3}), kContiguous);
    return ops::reshape(shuffled, static_cast<std::vector<int64_t>>(self.shape()));
}

TENSORPLAY_LIBRARY_IMPL(Composite, ChanelShuffleComposite) {
    m.impl("native_channel_shuffle", native_channel_shuffle_native);
}

} // namespace composite
} // namespace tensorplay
