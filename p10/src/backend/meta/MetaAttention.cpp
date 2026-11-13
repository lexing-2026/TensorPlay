// Shape-only kernels for scaled dot-product attention on the meta
// device.  The result keeps the query's leading extents and takes its
// last extent from the value plane; the log-sum-exp plane drops the
// query's last extent and repeats the head geometry.
#include "MetaCommon.h"

#include "Tensor.h"
#include "TensorImpl.h"
#include "Convolution.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Utils.h"
#include "Context.h"
#include "Generator.h"
#include "Scalar.h"
#include "DType.h"
#include "DTypeNames.h"
#include "DistributionDispatch.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "../BlasComposite.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace tensorplay {
namespace meta {
namespace {
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Attention. The result keeps the query's leading extents and takes its last
// one from the value; the attention weights themselves are summed out, so the
// log-sum-exp term drops that last extent and is reported in single precision.
// ---------------------------------------------------------------------------
std::vector<int64_t> attention_out_shape(const Tensor& query, const Tensor& value) {
    if (query.dim() < 1 || value.dim() < 1) {
        TP_THROW(ValueError, "scaled_dot_product_attention: query and value must "
                             "have at least one dimension");
    }
    std::vector<int64_t> out = shape_of(query);
    out.back() = value.size(-1);
    return out;
}

// What the attention the call describes requires of its operands: one head
// dimension read from both sides, a key and a value that describe the same
// weights, and head counts that differ only by whole groups when grouped
// attention is asked for.
void check_attention_shapes(const Tensor& query, const Tensor& key,
                            const Tensor& value, bool enable_gqa) {
    if (key.dim() < 1 || key.size(-1) != query.size(-1)) {
        TP_THROW(ValueError, "scaled_dot_product_attention: query and key head "
                             "dims must match");
    }
    if (shape_of(key) != shape_of(value)) {
        TP_THROW(ValueError, "scaled_dot_product_attention: key and value shapes "
                             "must match");
    }
    if (enable_gqa && query.dim() >= 3) {
        const int64_t heads = query.size(-3);
        const int64_t key_heads = key.size(-3);
        if (heads != key_heads && (key_heads == 0 || heads % key_heads != 0)) {
            TP_THROW(ValueError,
                     "scaled_dot_product_attention: enable_gqa requires the query "
                     "head count to be divisible by the key/value head count");
        }
    }
}

Tensor sdpa_kernel(const Tensor& query, const Tensor& key, const Tensor& value,
                   const std::optional<Tensor>& /*attn_mask*/, double /*dropout_p*/,
                   bool /*is_causal*/, std::optional<double> /*scale*/,
                   bool enable_gqa) {
    check_attention_shapes(query, key, value, enable_gqa);
    return empty_meta(attention_out_shape(query, value), query.dtype(),
                      query.device());
}

std::tuple<Tensor, Tensor> sdpa_with_lse_kernel(const Tensor& query,
                                                const Tensor& key,
                                                const Tensor& value,
                                                bool /*is_causal*/,
                                                int64_t /*impl*/) {
    check_attention_shapes(query, key, value, /*enable_gqa=*/false);
    std::vector<int64_t> out = attention_out_shape(query, value);
    std::vector<int64_t> lse(out.begin(), out.end() - 1);
    return {empty_meta(out, query.dtype(), query.device()),
            empty_meta(lse, DType::Float32, query.device())};
}
} // anonymous namespace
TENSORPLAY_LIBRARY_IMPL(Meta, MetaAttention) {
    m.impl("scaled_dot_product_attention", sdpa_kernel);
    m.impl("_scaled_dot_product_attention_with_lse", sdpa_with_lse_kernel);
}

} // namespace meta
} // namespace tensorplay
