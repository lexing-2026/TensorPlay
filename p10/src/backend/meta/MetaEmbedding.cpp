// Shape-only kernel for embedding lookup on the meta device.
//
// Embedding gathers whole rows of the weight table; the output repeats the
// index tensor's shape in the leading extents and appends the weight's
// trailing extents.  Only the geometry is observable here -- the gathered
// rows themselves are element data, which a meta tensor does not own.

#include "MetaCommon.h"

#include "DType.h"
#include "Exception.h"
#include "cpu/EmbeddingBagKernels.h"

#include <cstdint>
#include <optional>
#include <string>
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

// Shared dtype gates of the bagged-embedding kernels: gradients arrive in a
// floating width, indices in one of the two stored index widths.
void bag_check_float_dtype(DType dtype, const char* what) {
    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::Float16 && dtype != DType::BFloat16) {
        TP_THROW(TypeError, "embedding_bag: ", what,
                 " must be a floating point tensor");
    }
}

void bag_check_index_dtype(const Tensor& t, const char* what) {
    if (t.dtype() != DType::Int64 && t.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding_bag: ", what, " must be Int32 or Int64");
    }
}

// The dense gradient table repeats the gradient's row per weight row; the
// trailing extents come from the gradient's axes beyond the lookup's.
// Out-of-range lookups are rejected while reading the stored indices, which
// a meta tensor does not carry, so the range validation stays with a
// materializing backend.
Tensor embedding_dense_backward_kernel(const Tensor& grad_output,
                                       const Tensor& indices,
                                       int64_t num_weights,
                                       int64_t /*padding_idx*/,
                                       bool /*scale_grad_by_freq*/) {
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError,
                 "embedding_dense_backward: indices must be Int64 or Int32");
    }
    std::vector<int64_t> grad_weight_shape{num_weights};
    const int64_t weight_dims = grad_output.dim() - indices.dim();
    for (int64_t i = 0; i < weight_dims; ++i) {
        grad_weight_shape.push_back(grad_output.size(indices.dim() + i));
    }
    if (indices.numel() == 0) {
        return empty_meta(grad_weight_shape, grad_output.dtype(),
                          grad_output.device());
    }
    switch (grad_output.dtype()) {
        case DType::Float8_e4m3fn:
        case DType::Float8_e4m3fnuz:
        case DType::Float8_e5m2:
        case DType::Float8_e5m2fnuz:
        case DType::Float8_e8m0fnu:
            TP_THROW(TypeError,
                     "embedding_dense_backward: grad_output dtype is not supported");
        case DType::Bool:
            TP_THROW(RuntimeError,
                     "embedding_dense_backward: grad_output cannot be Bool");
        default:
            break;
    }
    return empty_meta(grad_weight_shape, grad_output.dtype(),
                      grad_output.device());
}

// The bagged backward answers the same (num_weights, embedding width) table;
// only the validations differ per reduction mode.  The max mode cross-checks
// the recorded winner buffer against the gradient plane; the other modes
// require one offset2bag entry per lookup.  Scatter itself reads element
// data, so only the geometry is answered here.
Tensor embedding_bag_dense_backward_kernel(
        const Tensor& grad, const Tensor& indices, const Tensor& offset2bag,
        const Tensor& /*bag_size*/, const Tensor& maximum_indices,
        int64_t num_weights, bool /*scale_grad_by_freq*/, int64_t mode,
        const std::optional<Tensor>& /*per_sample_weights*/,
        int64_t /*padding_idx*/) {
    bag_check_float_dtype(grad.dtype(), "grad");
    bag_check_index_dtype(indices, "indices");
    if (grad.dim() != 2) {
        TP_THROW(RuntimeError, "embedding_bag_backward: grad must be 2-D");
    }
    if (num_weights < 0) {
        TP_THROW(ValueError,
                 "embedding_bag_backward: num_weights must be non-negative");
    }
    const int64_t embedding_dim = grad.size(1);
    if (mode == cpu::kBagMax) {
        if (maximum_indices.numel() != grad.size(0) * embedding_dim) {
            TP_THROW(RuntimeError,
                     "embedding_bag_backward: max index buffer does not match the gradient shape");
        }
    } else if (indices.numel() != 0) {
        if (offset2bag.numel() != indices.numel()) {
            TP_THROW(RuntimeError,
                     "embedding_bag_backward: offset2bag must have one entry per index");
        }
    }
    return empty_meta({num_weights, embedding_dim}, grad.dtype(), grad.device());
}

// The per-sample-weights backward folds the weight plane away: one scalar
// per lookup.  Only the sum reduction carries per-sample weights at all.
Tensor embedding_bag_per_sample_weights_backward_kernel(
        const Tensor& grad, const Tensor& weight, const Tensor& indices,
        const Tensor& offsets, const Tensor& offset2bag, int64_t mode,
        int64_t /*padding_idx*/) {
    if (mode != cpu::kBagSum) {
        TP_THROW(RuntimeError,
                 "embedding_bag_backward: per_sample_weights is only supported in sum mode");
    }
    bag_check_float_dtype(grad.dtype(), "grad");
    bag_check_index_dtype(indices, "indices");
    if (grad.dim() != 2 || weight.dim() != 2) {
        TP_THROW(RuntimeError,
                 "embedding_bag_backward: grad and weight must be 2-D");
    }
    if (grad.size(1) != weight.size(1)) {
        TP_THROW(RuntimeError,
                 "embedding_bag_backward: grad and weight must agree on the embedding size");
    }
    // An empty offset2bag means the caller skipped the forward bookkeeping
    // and the offsets rebuild it; that path gates the offsets dtype first.
    if (offset2bag.numel() == 0) {
        bag_check_index_dtype(offsets, "offsets");
    } else if (offset2bag.numel() != indices.numel()) {
        TP_THROW(RuntimeError,
                 "embedding_bag_backward: offset2bag must have one entry per index");
    }
    return empty_meta({indices.numel()}, grad.dtype(), grad.device());
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaEmbedding) {
    m.impl("embedding", embedding_kernel);
    m.impl("embedding_dense_backward", embedding_dense_backward_kernel);
    m.impl("_embedding_bag_dense_backward", embedding_bag_dense_backward_kernel);
    m.impl("_embedding_bag_per_sample_weights_backward",
           embedding_bag_per_sample_weights_backward_kernel);
}

} // namespace meta
} // namespace tensorplay