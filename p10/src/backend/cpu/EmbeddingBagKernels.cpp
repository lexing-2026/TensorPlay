#include "cpu/EmbeddingBagKernels.h"

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

// Bagged-embedding operators.
//
// This translation unit owns everything that does not benefit from being
// recompiled per instruction set: argument validation, the bag layout derived
// from `offsets`, output allocation, and the index sort that gives the dense
// backward a lock-free partition of output rows.  The reduction loops
// themselves live in EmbeddingBagKernelsImpl.cpp behind the stubs declared in
// cpu/EmbeddingBagKernels.h.

namespace tensorplay {
namespace cpu {

DEFINE_DISPATCH(bag_forward_stub);
DEFINE_DISPATCH(bag_dense_backward_stub);
DEFINE_DISPATCH(bag_psw_backward_stub);

namespace {

// Per-bag [start, end) ranges over the flat `indices` vector.  Positions that
// no bag covers (possible when the trailing offset stops short of the index
// count) stay out of every range and take no part in the reduction.
struct BagLayout {
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
};

int64_t read_index(const Tensor& t, int64_t i) {
    return t.dtype() == DType::Int64 ? t.data_ptr<int64_t>()[i]
                                     : static_cast<int64_t>(t.data_ptr<int32_t>()[i]);
}

void check_index_dtype(const Tensor& t, const char* what) {
    if (t.dtype() != DType::Int64 && t.dtype() != DType::Int32) {
        TP_THROW(TypeError,
                 std::string("embedding_bag: ") + what + " must be Int32 or Int64");
    }
}

void check_float_dtype(DType dtype, const char* what) {
    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::Float16 && dtype != DType::BFloat16) {
        TP_THROW(TypeError,
                 std::string("embedding_bag: ") + what + " must be a floating point tensor");
    }
}

// The reduction kernels read one index width; Int32 callers pay a single
// widening pass over `indices`, which is negligible next to the row gathers.
Tensor as_int64(const Tensor& t) {
    return t.dtype() == DType::Int64 ? t.contiguous()
                                     : t.to(DType::Int64).contiguous();
}

int64_t bag_count(const Tensor& offsets, bool include_last_offset) {
    const int64_t n = offsets.numel();
    if (include_last_offset) {
        if (n < 1) {
            TP_THROW(RuntimeError,
                     "embedding_bag: include_last_offset requires at least one offset");
        }
        return n - 1;
    }
    return n;
}

BagLayout make_bag_layout(const Tensor& offsets, int64_t numel, int64_t num_bags,
                          bool include_last_offset) {
    BagLayout layout;
    layout.starts.resize(num_bags);
    layout.ends.resize(num_bags);
    const int64_t n_offsets = offsets.numel();
    int64_t previous = 0;
    for (int64_t b = 0; b < num_bags; ++b) {
        int64_t start = read_index(offsets, b);
        if (b == 0 && start != 0) {
            TP_THROW(RuntimeError, "embedding_bag: offsets[0] must be 0");
        }
        if (start < previous) {
            TP_THROW(RuntimeError, "embedding_bag: offsets must be non-decreasing");
        }
        previous = start;
        int64_t end = (b + 1 < n_offsets) ? read_index(offsets, b + 1) : numel;
        if (!include_last_offset && b + 1 == num_bags) {
            end = numel;
        }
        start = std::min(std::max<int64_t>(start, 0), numel);
        end = std::min(std::max(end, start), numel);
        layout.starts[b] = start;
        layout.ends[b] = end;
    }
    return layout;
}

// Positions outside every bag range keep -1, so both backward kernels can skip
// them without consulting the layout again.
void mark_uncovered_positions(const BagLayout& layout, int64_t numel,
                              Tensor& offset2bag) {
    if (numel == 0) return;
    int64_t* o2b = offset2bag.data_ptr<int64_t>();
    int64_t covered_to = 0;
    for (size_t b = 0; b < layout.starts.size(); ++b) {
        for (int64_t i = covered_to; i < layout.starts[b]; ++i) o2b[i] = -1;
        covered_to = std::max(covered_to, layout.ends[b]);
    }
    for (int64_t i = covered_to; i < numel; ++i) o2b[i] = -1;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_forward(
        const Tensor& weight, const Tensor& indices_arg, const Tensor& offsets_arg,
        int64_t mode, const std::optional<Tensor>& per_sample_weights_opt,
        bool include_last_offset, int64_t padding_idx) {
    if (weight.dim() != 2) {
        TP_THROW(RuntimeError, "embedding_bag: weight must be 2-D");
    }
    check_float_dtype(weight.dtype(), "weight");
    check_index_dtype(indices_arg, "indices");
    check_index_dtype(offsets_arg, "offsets");
    if (indices_arg.dim() != 1) {
        TP_THROW(RuntimeError, "embedding_bag: indices must be 1-D");
    }
    if (offsets_arg.dim() != 1) {
        TP_THROW(RuntimeError, "embedding_bag: offsets must be 1-D");
    }
    if (mode != kBagSum && mode != kBagMean && mode != kBagMax) {
        TP_THROW(ValueError, "embedding_bag: mode must be 0 (sum), 1 (mean) or 2 (max)");
    }

    Tensor per_sample_weights =
        per_sample_weights_opt.has_value() ? *per_sample_weights_opt : Tensor();
    if (per_sample_weights.defined() && per_sample_weights.numel() > 0) {
        if (mode != kBagSum) {
            TP_THROW(RuntimeError,
                     "embedding_bag: per_sample_weights is only supported in sum mode");
        }
        if (per_sample_weights.dim() != 1 ||
            per_sample_weights.numel() != indices_arg.numel()) {
            TP_THROW(RuntimeError,
                     "embedding_bag: per_sample_weights must be 1-D with one entry per index");
        }
        if (per_sample_weights.dtype() != weight.dtype()) {
            per_sample_weights = per_sample_weights.to(weight.dtype());
        }
        per_sample_weights = per_sample_weights.contiguous();
    }

    const Tensor w = weight.contiguous();
    const Tensor indices = as_int64(indices_arg);
    const Tensor offsets = offsets_arg.contiguous();

    const int64_t numel = indices.numel();
    const int64_t num_bags = bag_count(offsets, include_last_offset);
    const int64_t D = w.size(1);
    const BagLayout layout = make_bag_layout(offsets, numel, num_bags, include_last_offset);

    Tensor output = Tensor::empty({num_bags, D}, w.dtype(), w.device());
    // The reduction writes offset2bag for every covered position, so only the
    // gaps between bags need the "no bag owns this" marker written here.
    Tensor offset2bag = Tensor::empty({numel}, DType::Int64, w.device());
    Tensor bag_size = Tensor::empty({num_bags}, DType::Int64, w.device());
    Tensor max_indices = mode == kBagMax
        ? Tensor::empty({num_bags, D}, DType::Int64, w.device())
        : Tensor::zeros({num_bags}, DType::Int64, w.device());
    mark_uncovered_positions(layout, numel, offset2bag);

    if (num_bags == 0) {
        return {output, offset2bag, bag_size, max_indices};
    }

    BagForwardArgs args{};
    args.dtype = w.dtype();
    args.weight = w.data_ptr();
    args.per_sample_weights =
        per_sample_weights.defined() && per_sample_weights.numel() > 0
            ? per_sample_weights.data_ptr() : nullptr;
    args.indices = numel > 0 ? indices.data_ptr<int64_t>() : nullptr;
    args.starts = layout.starts.data();
    args.ends = layout.ends.data();
    args.num_rows = w.size(0);
    args.embedding_dim = D;
    args.num_indices = numel;
    args.num_bags = num_bags;
    args.mode = mode;
    args.padding_idx = padding_idx;
    args.output = output.data_ptr();
    args.offset2bag = numel > 0 ? offset2bag.data_ptr<int64_t>() : nullptr;
    args.bag_size = bag_size.data_ptr<int64_t>();
    args.max_indices = mode == kBagMax ? max_indices.data_ptr<int64_t>() : nullptr;
    bag_forward_stub(DeviceType::CPU, args);

    return {output, offset2bag, bag_size, max_indices};
}

// Sorting (row, position) pairs groups every occurrence of a table row into
// one contiguous run, so the dense backward can hand each worker a distinct
// set of output rows: no locks, and a reproducible accumulation order.
struct SortedIndices {
    std::vector<int64_t> rows;
    std::vector<int64_t> positions;
    std::vector<int64_t> segment_starts;
};

SortedIndices sort_indices(const Tensor& indices) {
    const int64_t numel = indices.numel();
    const int64_t* idx = numel > 0 ? indices.data_ptr<int64_t>() : nullptr;

    std::vector<std::pair<int64_t, int64_t>> pairs(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        pairs[static_cast<size_t>(i)] = {idx[i], i};
    }
    std::sort(pairs.begin(), pairs.end());

    SortedIndices sorted;
    sorted.rows.resize(static_cast<size_t>(numel));
    sorted.positions.resize(static_cast<size_t>(numel));
    sorted.segment_starts.reserve(static_cast<size_t>(numel) + 1);
    for (size_t k = 0; k < pairs.size(); ++k) {
        sorted.rows[k] = pairs[k].first;
        sorted.positions[k] = pairs[k].second;
        if (k == 0 || pairs[k].first != pairs[k - 1].first) {
            sorted.segment_starts.push_back(static_cast<int64_t>(k));
        }
    }
    sorted.segment_starts.push_back(static_cast<int64_t>(pairs.size()));
    return sorted;
}

} // namespace

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_cpu(
        const Tensor& weight, const Tensor& indices, const Tensor& offsets,
        bool scale_grad_by_freq, int64_t mode, bool sparse,
        const std::optional<Tensor>& per_sample_weights,
        bool include_last_offset, int64_t padding_idx) {
    // Both flags only shape the backward; the forward reduction is the same.
    (void)scale_grad_by_freq;
    (void)sparse;
    return embedding_bag_forward(weight, indices, offsets, mode, per_sample_weights,
                                 include_last_offset, padding_idx);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_forward_only_cpu(
        const Tensor& weight, const Tensor& indices, const Tensor& offsets,
        bool scale_grad_by_freq, int64_t mode, bool sparse,
        const std::optional<Tensor>& per_sample_weights,
        bool include_last_offset, int64_t padding_idx) {
    return _embedding_bag_cpu(weight, indices, offsets, scale_grad_by_freq, mode, sparse,
                              per_sample_weights, include_last_offset, padding_idx);
}

Tensor _embedding_bag_dense_backward_cpu(
        const Tensor& grad_arg, const Tensor& indices_arg, const Tensor& offset2bag_arg,
        const Tensor& bag_size_arg, const Tensor& maximum_indices_arg,
        int64_t num_weights, bool scale_grad_by_freq, int64_t mode,
        const std::optional<Tensor>& per_sample_weights_opt, int64_t padding_idx) {
    check_float_dtype(grad_arg.dtype(), "grad");
    check_index_dtype(indices_arg, "indices");
    if (grad_arg.dim() != 2) {
        TP_THROW(RuntimeError, "embedding_bag_backward: grad must be 2-D");
    }
    if (num_weights < 0) {
        TP_THROW(ValueError, "embedding_bag_backward: num_weights must be non-negative");
    }

    const Tensor grad = grad_arg.contiguous();
    const Tensor bag_size = bag_size_arg.contiguous();
    const int64_t D = grad.size(1);

    Tensor grad_weight = Tensor::zeros({num_weights, D}, grad.dtype(), grad.device());
    if (num_weights == 0 || D == 0) return grad_weight;

    BagDenseBackwardArgs args{};
    args.dtype = grad.dtype();
    args.grad = grad.data_ptr();
    args.bag_size = bag_size.numel() > 0 ? bag_size.data_ptr<int64_t>() : nullptr;
    args.num_bags = grad.size(0);
    args.embedding_dim = D;
    args.num_weights = num_weights;
    args.mode = mode;
    args.padding_idx = padding_idx;
    args.grad_weight = grad_weight.data_ptr();

    if (mode == kBagMax) {
        const Tensor max_indices = maximum_indices_arg.contiguous();
        if (max_indices.numel() != grad.size(0) * D) {
            TP_THROW(RuntimeError,
                     "embedding_bag_backward: max index buffer does not match the gradient shape");
        }
        args.max_indices = max_indices.data_ptr<int64_t>();
        bag_dense_backward_stub(DeviceType::CPU, args);
        return grad_weight;
    }

    const Tensor indices = as_int64(indices_arg);
    if (indices.numel() == 0) return grad_weight;
    const Tensor offset2bag = as_int64(offset2bag_arg);
    if (offset2bag.numel() != indices.numel()) {
        TP_THROW(RuntimeError,
                 "embedding_bag_backward: offset2bag must have one entry per index");
    }

    Tensor per_sample_weights =
        per_sample_weights_opt.has_value() ? *per_sample_weights_opt : Tensor();
    if (per_sample_weights.defined() && per_sample_weights.numel() > 0) {
        if (per_sample_weights.dtype() != grad.dtype()) {
            per_sample_weights = per_sample_weights.to(grad.dtype());
        }
        per_sample_weights = per_sample_weights.contiguous();
    }

    const SortedIndices sorted = sort_indices(indices);
    args.per_sample_weights =
        per_sample_weights.defined() && per_sample_weights.numel() > 0
            ? per_sample_weights.data_ptr() : nullptr;
    args.sorted_rows = sorted.rows.data();
    args.sorted_pos = sorted.positions.data();
    args.segment_starts = sorted.segment_starts.data();
    args.num_segments = static_cast<int64_t>(sorted.segment_starts.size()) - 1;
    args.offset2bag = offset2bag.data_ptr<int64_t>();
    args.scale_grad_by_freq = scale_grad_by_freq;
    bag_dense_backward_stub(DeviceType::CPU, args);
    return grad_weight;
}

Tensor _embedding_bag_per_sample_weights_backward_cpu(
        const Tensor& grad_arg, const Tensor& weight_arg, const Tensor& indices_arg,
        const Tensor& offsets_arg, const Tensor& offset2bag_arg, int64_t mode,
        int64_t padding_idx) {
    if (mode != kBagSum) {
        TP_THROW(RuntimeError,
                 "embedding_bag_backward: per_sample_weights is only supported in sum mode");
    }
    check_float_dtype(grad_arg.dtype(), "grad");
    check_index_dtype(indices_arg, "indices");
    if (grad_arg.dim() != 2 || weight_arg.dim() != 2) {
        TP_THROW(RuntimeError, "embedding_bag_backward: grad and weight must be 2-D");
    }
    if (grad_arg.size(1) != weight_arg.size(1)) {
        TP_THROW(RuntimeError,
                 "embedding_bag_backward: grad and weight must agree on the embedding size");
    }

    const Tensor grad = grad_arg.contiguous();
    const Tensor weight = weight_arg.to(grad.dtype()).contiguous();
    const Tensor indices = as_int64(indices_arg);
    const int64_t numel = indices.numel();

    Tensor output = Tensor::zeros({numel}, grad.dtype(), grad.device());
    if (numel == 0) return output;

    // A caller that skipped the forward bookkeeping can pass an empty
    // offset2bag; rebuild it from the offsets in that case.
    Tensor offset2bag = offset2bag_arg;
    if (offset2bag.numel() == 0) {
        check_index_dtype(offsets_arg, "offsets");
        const Tensor offsets = offsets_arg.contiguous();
        const bool include_last_offset = offsets.numel() == grad.size(0) + 1;
        const int64_t num_bags = bag_count(offsets, include_last_offset);
        const BagLayout layout =
            make_bag_layout(offsets, numel, num_bags, include_last_offset);
        offset2bag = Tensor::full({numel}, Scalar(static_cast<int64_t>(-1)),
                                  DType::Int64, grad.device());
        int64_t* o2b = offset2bag.data_ptr<int64_t>();
        for (int64_t b = 0; b < num_bags; ++b) {
            for (int64_t i = layout.starts[b]; i < layout.ends[b]; ++i) o2b[i] = b;
        }
    } else {
        if (offset2bag.numel() != numel) {
            TP_THROW(RuntimeError,
                     "embedding_bag_backward: offset2bag must have one entry per index");
        }
        offset2bag = as_int64(offset2bag);
    }

    BagPerSampleWeightsArgs args{};
    args.dtype = grad.dtype();
    args.grad = grad.data_ptr();
    args.weight = weight.data_ptr();
    args.indices = indices.data_ptr<int64_t>();
    args.offset2bag = offset2bag.data_ptr<int64_t>();
    args.num_rows = weight.size(0);
    args.num_bags = grad.size(0);
    args.embedding_dim = grad.size(1);
    args.num_indices = numel;
    args.padding_idx = padding_idx;
    args.output = output.data_ptr();
    bag_psw_backward_stub(DeviceType::CPU, args);
    return output;
}

TENSORPLAY_LIBRARY_IMPL(CPU, EmbeddingBagKernels) {
    m.impl("_embedding_bag", _embedding_bag_cpu);
    m.impl("_embedding_bag_forward_only", _embedding_bag_forward_only_cpu);
    m.impl("_embedding_bag_dense_backward", _embedding_bag_dense_backward_cpu);
    m.impl("_embedding_bag_per_sample_weights_backward",
           _embedding_bag_per_sample_weights_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay
