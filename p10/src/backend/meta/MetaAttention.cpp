// Shape-only kernels for scaled dot-product attention on the meta
// device.  The composed attention builds its answer from recordable
// primitives -- a transposed score product, an optional additive mask,
// a last-dim softmax and a weighted value product -- so the checks the
// primitives would run are run here, and the result keeps the score
// product's batch geometry with the query's row extent and the value's
// head extent.  The log-sum-exp plane drops the score plane's last
// extent and is always reported in single precision.
#include "MetaCommon.h"

#include "Tensor.h"
#include "TensorImpl.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "DType.h"
#include "DTypeNames.h"
#include "TypeProperties.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace meta {
namespace {

namespace ops = tpx::ops;

// What the composed attention the call describes requires of its operands:
// one head dimension read from the query, a key and a value that describe
// the same weights, and head counts that differ only by whole groups when
// grouped attention is asked for.  `key_shape` and `value_shape` come back
// with the query's head count when the group expansion applies, because the
// composed form repeats the key/value heads before any product runs.
void check_attention_shapes(const Tensor& query, const Tensor& key,
                            const Tensor& value, bool enable_gqa,
                            std::vector<int64_t>& key_shape,
                            std::vector<int64_t>& value_shape) {
    key_shape = shape_of(key);
    value_shape = shape_of(value);
    if (enable_gqa) {
        if (query.dim() < 3 || key.dim() < 3) {
            TP_THROW(ValueError, "sdpa math: enable_gqa requires 4D inputs");
        }
        const int64_t hq = query.size(-3);
        const int64_t hk = key.size(-3);
        if (hq != hk) {
            if (hk == 0 || hq % hk != 0) {
                TP_THROW(ValueError,
                         "sdpa math: enable_gqa requires the query head count "
                         "to be divisible by the key/value head count");
            }
            key_shape[key_shape.size() - 3] = hq;
            value_shape[value_shape.size() - 3] = hq;
        }
    }
    if (key_shape != value_shape) {
        TP_THROW(ValueError, "sdpa math: key and value shapes must match");
    }
    if (key.size(-1) != query.size(-1)) {
        TP_THROW(ValueError, "sdpa math: query and key head dims must match");
    }
    if (query.dim() >= 3 && key.dim() == query.dim()) {
        std::vector<int64_t> q_lead = shape_of(query);
        q_lead.resize(q_lead.size() - 2);
        std::vector<int64_t> k_lead(key_shape.begin(), key_shape.end() - 2);
        if (q_lead != k_lead) {
            TP_THROW(ValueError,
                     "sdpa math: query and key/value leading dims must match");
        }
    }
}

// The composed attention is built from recordable primitives, so the checks
// the primitives would run are run here against real meta tensors: the
// transposed score product and the weighted value product both go through
// the meta matrix product (rank, inner extent, element type and batch
// broadcast contracts), and an additive mask is lifted to the accumulation
// type first and then goes through the meta add (broadcast contract).
std::tuple<Tensor, Tensor> sdpa_planes(const Tensor& query, Tensor key,
                                       Tensor value,
                                       const std::optional<Tensor>& attn_mask,
                                       DType working) {
    Tensor scores = ops::matmul(query, ops::transpose(key, -2, -1));
    if (attn_mask.has_value()) {
        scores = ops::add(scores, ops::to(*attn_mask, working));
    }
    Tensor out = ops::matmul(scores, value);
    std::vector<int64_t> lse = shape_of(scores);
    lse.pop_back();
    return {out, empty_meta(lse, DType::Float32, query.device())};
}

Tensor sdpa_kernel(const Tensor& query, const Tensor& key, const Tensor& value,
                   const std::optional<Tensor>& attn_mask, double /*dropout_p*/,
                   bool /*is_causal*/, std::optional<double> /*scale*/,
                   bool enable_gqa) {
    const DType origin = query.dtype();
    if (origin != DType::Float32 && origin != DType::Float64 &&
        origin != DType::Float16 && origin != DType::BFloat16) {
        TP_THROW(NotImplementedError,
                 "sdpa math: expected float32/float64/float16/bfloat16");
    }
    if (query.size(-1) == 0) {
        TP_THROW(ValueError, "sdpa math: head dimension must be non-zero");
    }
    std::vector<int64_t> key_shape;
    std::vector<int64_t> value_shape;
    check_attention_shapes(query, key, value, enable_gqa, key_shape,
                           value_shape);
    // The group expansion repeats the key and value heads before the score
    // product runs; meta tensors carry no data, so empty planes stand in for
    // the expanded operands and every contract still runs on their shapes.
    const Tensor k = empty_meta(key_shape, key.dtype(), key.device());
    const Tensor v = empty_meta(value_shape, value.dtype(), value.device());
    // Reduced-width operands accumulate in single precision; every other
    // width computes in its own type.
    const DType working = (origin == DType::Float16 || origin == DType::BFloat16)
                              ? DType::Float32
                              : origin;
    Tensor out;
    Tensor lse;
    std::tie(out, lse) = sdpa_planes(query, k, v, attn_mask, working);
    (void)lse;
    // ... and come back to the width they arrived in.
    return empty_meta(shape_of(out), origin, query.device());
}

// The variant that also reports the softmax normalizing constant skips the
// key/value geometry checks the plain form makes: its score product runs on
// whatever it is handed, and a disagreement surfaces as the matrix product's
// own refusal.
std::tuple<Tensor, Tensor> sdpa_with_lse_kernel(const Tensor& query,
                                                const Tensor& key,
                                                const Tensor& value,
                                                bool /*is_causal*/,
                                                int64_t /*impl*/) {
    const DType origin = query.dtype();
    if (origin != DType::Float32 && origin != DType::Float64 &&
        origin != DType::Float16 && origin != DType::BFloat16) {
        TP_THROW(NotImplementedError,
                 "sdpa math: expected float32/float64/float16/bfloat16");
    }
    Tensor out;
    Tensor lse;
    std::tie(out, lse) = sdpa_planes(query, key, value, std::nullopt, origin);
    return {empty_meta(shape_of(out), query.dtype(), query.device()), lse};
}
} // anonymous namespace
TENSORPLAY_LIBRARY_IMPL(Meta, MetaAttention) {
    m.impl("scaled_dot_product_attention", sdpa_kernel);
    m.impl("_scaled_dot_product_attention_with_lse", sdpa_with_lse_kernel);
}

} // namespace meta
} // namespace tensorplay
