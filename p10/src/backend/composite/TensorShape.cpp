// Composite kernels: reshape_as / unsafe_chunk / unsafe_split.
// differ from chunk/split by zeroing the results' version counters (an
// autograd-internal concern); the view structure is identical.

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

Tensor reshape_as_native(const Tensor& self, const Tensor& other) {
    return ops::reshape(self, static_cast<std::vector<int64_t>>(other.shape()));
}

std::vector<Tensor> unsafe_chunk_native(const Tensor& self, int64_t chunks,
                                        int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "chunk expects at least a 1-dimensional tensor");
    }
    if (chunks <= 0) {
        TP_THROW(RuntimeError,
                 "chunk expects chunks to be a positive integer");
    }
    const int64_t d = wrap_dim(dim, self.dim());
    const int64_t dim_size = self.size(d);
    const int64_t split_size = (dim_size + chunks - 1) / chunks;
    if (split_size == 0 && dim_size == 0) {
        std::vector<Tensor> pieces;
        pieces.reserve(static_cast<size_t>(chunks));
        for (int64_t i = 0; i < chunks; ++i) {
            pieces.push_back(ops::slice(self, d, 0, 0, 1));
        }
        return pieces;
    }
    return ops::split(self, split_size, d);
}

std::vector<Tensor> unsafe_split_native(const Tensor& self,
                                        int64_t split_size, int64_t dim) {
    return ops::split(self, split_size, dim);
}

Tensor fliplr_native(const Tensor& self) {
    return ops::flip(self, {-1});
}

Tensor flipud_native(const Tensor& self) {
    return ops::flip(self, {0});
}

Tensor& resize_as__native(Tensor& self, const Tensor& other,
                          int64_t /*memory_format*/) {
    ops::resize_(self, static_cast<std::vector<int64_t>>(other.shape()));
    return self;
}

TENSORPLAY_LIBRARY_IMPL(Composite, TensorShapeComposite) {
    m.impl("reshape_as", reshape_as_native);
    m.impl("unsafe_chunk", unsafe_chunk_native);
    m.impl("unsafe_split.Tensor", unsafe_split_native);
    m.impl("fliplr", fliplr_native);
    m.impl("flipud", flipud_native);
    m.impl("resize_as_", resize_as__native);
}

} // namespace composite
} // namespace tensorplay
