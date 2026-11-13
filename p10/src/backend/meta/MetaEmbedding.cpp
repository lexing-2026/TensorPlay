// Shape-only kernel for embedding lookup on the meta device.
//
// Embedding gathers whole rows of the weight table; the output repeats the
// index tensor's shape in the leading extents and appends the weight's
// trailing extents.  Only the geometry is observable here -- the gathered
// rows themselves are element data, which a meta tensor does not own.

#include "MetaCommon.h"

#include "DType.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

Tensor embedding_kernel(const Tensor& weight, const Tensor& indices,
                        int64_t /*padding_idx*/, bool /*scale_grad_by_freq*/,
                        bool /*sparse*/) {
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding: indices must be Int64 or Int32");
    }
    if (weight.dim() == 0) {
        TP_THROW(RuntimeError, "embedding: weight must be at least 1-dim");
    }
    std::vector<int64_t> out_shape = shape_of(indices);
    const std::vector<int64_t> weight_shape = shape_of(weight);
    out_shape.insert(out_shape.end(), weight_shape.begin() + 1,
                     weight_shape.end());
    return empty_meta(out_shape, weight.dtype(), weight.device());
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaEmbedding) {
    m.impl("embedding", embedding_kernel);
}

} // namespace meta
} // namespace tensorplay