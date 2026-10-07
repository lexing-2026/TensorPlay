#pragma once
// Second derivatives through the backward kernels that move data around:
// pooling, unpooling, embedding lookups and bags.
//
// Each of these kernels is linear in its incoming gradient and reads the
// forward input only for its shape (or, for max pooling, through the saved
// argmax indices), so the derivative with respect to grad_output is the
// transpose of the kernel: a gather where the kernel scatters, a scatter
// where it gathers, the forward op where it is the forward's adjoint.
// The formulas that use these helpers sit next to the kernels' schemas in
// derivatives.yaml.  Everything is written with differentiable operations so
// a third pass records as well.

#include "Autograd.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace tpx {

namespace shape_bwd_detail {

inline int64_t product(const std::vector<int64_t>& v) {
    int64_t n = 1;
    for (int64_t x : v) n *= x;
    return n;
}

// The leading (batch / channel) extents of `shape` before the last `dim`
// pooled extents, followed by a -1 that flattens the pooled extents.
inline std::vector<int64_t> flatten_plane_shape(const std::vector<int64_t>& shape,
                                                int64_t dim) {
    std::vector<int64_t> flat(shape.begin(), shape.end() - dim);
    flat.push_back(-1);
    return flat;
}

inline Tensor int64_indices(const Tensor& indices) {
    return indices.dtype() == DType::Int64 ? indices : ops::to(indices, DType::Int64);
}

// A 0/1 mask in the type of `like`.
inline Tensor mask_like(const Tensor& mask, const Tensor& like) {
    return ops::to(mask, like.dtype());
}

// Reshape a per-row vector so it multiplies the rows of `rows`.
inline Tensor as_column(const Tensor& v, const Tensor& rows) {
    std::vector<int64_t> shape(static_cast<size_t>(rows.dim()), 1);
    shape[0] = -1;
    return ops::reshape(v, shape);
}

// How often each table row occurs in `rows` (rows already in range), as a
// float vector of `num_weights` entries in the type of `like`.
inline Tensor row_counts(const Tensor& rows, int64_t num_weights, const Tensor& like,
                         const Tensor& weights_per_row) {
    const Tensor zeros = ops::zeros({num_weights}, like.dtype(), like.device());
    return ops::scatter_add(zeros, 0, rows, weights_per_row);
}

}  // namespace shape_bwd_detail

// The gather that a max-pool style backward (a scatter-add of grad_output at
// the saved argmax positions) transposes to: pick `grad` at every index.
// `indices` are flat offsets into each plane of the last `dim` extents.
inline Tensor max_pool_double_backward(const Tensor& grad, const Tensor& indices,
                                       int64_t dim) {
    const std::vector<int64_t> index_shape = indices.shape();
    if (indices.numel() == 0) {
        return ops::zeros(index_shape, grad.dtype(), grad.device());
    }
    const std::vector<int64_t> flat = shape_bwd_detail::flatten_plane_shape(index_shape, dim);
    const Tensor picked = ops::gather(ops::reshape(grad, flat),
                                      static_cast<int64_t>(flat.size()) - 1,
                                      ops::reshape(indices, flat));
    return ops::reshape(picked, index_shape);
}

// The scatter-add that max_pool_double_backward's gather transposes to:
// `grad` has the shape of `indices`; the result has the planes of
// `output_size` and accumulates grad at every index.
inline Tensor max_pool_scatter(const Tensor& grad, const Tensor& indices,
                               const std::vector<int64_t>& output_size, int64_t dim) {
    const std::vector<int64_t> index_shape = indices.shape();
    std::vector<int64_t> out_shape(index_shape.begin(), index_shape.end() - dim);
    out_shape.insert(out_shape.end(), output_size.end() - dim, output_size.end());
    if (grad.numel() == 0) {
        return ops::zeros(out_shape, grad.dtype(), grad.device());
    }
    const std::vector<int64_t> flat = shape_bwd_detail::flatten_plane_shape(index_shape, dim);
    std::vector<int64_t> plane_shape(index_shape.begin(), index_shape.end() - dim);
    plane_shape.push_back(shape_bwd_detail::product(
        std::vector<int64_t>(output_size.end() - dim, output_size.end())));
    const Tensor target = ops::zeros(plane_shape, grad.dtype(), grad.device());
    const Tensor scattered = ops::scatter_add(
        target, static_cast<int64_t>(plane_shape.size()) - 1,
        ops::reshape(indices, flat), ops::reshape(grad, flat));
    return ops::reshape(scattered, out_shape);
}

// max_pool{2,3}d_with_indices_backward differentiated in grad_output.  A
// caller that did not pass the indices gets them recomputed from the input.
inline Tensor max_pool_with_indices_double_backward(
        const Tensor& grad, const Tensor& input, const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, bool ceil_mode,
        const std::optional<Tensor>& indices, int64_t dim) {
    Tensor index;
    if (indices.has_value() && indices->defined()) {
        index = *indices;
    } else {
        const Tensor plain = ops::detach(input);
        index = dim == 2
            ? std::get<1>(ops::max_pool2d_with_indices(plain, kernel_size, stride, padding,
                                                       dilation, ceil_mode))
            : std::get<1>(ops::max_pool3d_with_indices(plain, kernel_size, stride, padding,
                                                       dilation, ceil_mode));
    }
    return max_pool_double_backward(grad, index, dim);
}

// The last `dim` extents of a shape: the pooled size an adaptive pool
// produced, read off its gradient.
inline std::vector<int64_t> trailing_extents(const std::vector<int64_t>& shape, int64_t dim) {
    return std::vector<int64_t>(shape.end() - dim, shape.end());
}

// The 2D adaptive-average backward kernels take 4D tensors only while the
// forward also accepts an unbatched (C, H, W) input; lift it to a batch of one.
inline Tensor adaptive_avg_pool2d_backward_any(const Tensor& grad, const Tensor& input) {
    if (input.dim() == 3) {
        return ops::squeeze(ops::adaptive_avg_pool2d_backward(
            ops::unsqueeze(grad, 0), ops::unsqueeze(input, 0)), 0);
    }
    return ops::adaptive_avg_pool2d_backward(grad, input);
}

// Flat in-volume argmax offsets of adaptive_max_pool3d, which keeps no
// indices.  The bins are the floor / ceil windows of the pool; a box maximum
// splits into three one-axis maxima (W, then H, then D) that carry the offset
// along, and a tie goes to the first element in row-major order like the
// kernel's.
inline Tensor adaptive_max_pool3d_indices(const Tensor& input,
                                          const std::vector<int64_t>& output_shape) {
    Tensor value = ops::detach(input);
    const bool batched = value.dim() == 5;
    if (!batched) value = ops::unsqueeze(value, 0);
    const std::vector<int64_t> in_shape = value.shape();
    const std::vector<int64_t> out_size = trailing_extents(output_shape, 3);

    Tensor offset = ops::reshape(
        ops::arange(Scalar(static_cast<int64_t>(0)),
                    Scalar(in_shape[2] * in_shape[3] * in_shape[4]),
                    Scalar(static_cast<int64_t>(1)), DType::Int64, value.device()),
        {1, 1, in_shape[2], in_shape[3], in_shape[4]});
    offset = ops::contiguous(ops::expand(offset, in_shape));

    for (int64_t axis = 4; axis >= 2; --axis) {
        const int64_t extent = in_shape[axis];
        const int64_t bins = out_size[axis - 2];
        std::vector<Tensor> values, offsets;
        values.reserve(static_cast<size_t>(bins));
        offsets.reserve(static_cast<size_t>(bins));
        for (int64_t o = 0; o < bins; ++o) {
            const int64_t start = o * extent / bins;
            const int64_t end = 1 + ((o + 1) * extent - 1) / bins;
            const Tensor window = ops::narrow(value, axis, start, end - start);
            const Tensor window_offset = ops::narrow(offset, axis, start, end - start);
            const Tensor arg = std::get<1>(ops::max(window, axis, true));
            values.push_back(ops::gather(window, axis, arg));
            offsets.push_back(ops::gather(window_offset, axis, arg));
        }
        value = ops::cat(values, axis);
        offset = ops::cat(offsets, axis);
    }
    return batched ? offset : ops::squeeze(offset, 0);
}

inline Tensor adaptive_max_pool3d_double_backward(const Tensor& grad, const Tensor& input,
                                                  const std::vector<int64_t>& output_shape) {
    return max_pool_double_backward(grad, adaptive_max_pool3d_indices(input, output_shape), 3);
}

inline Tensor adaptive_max_pool2d_double_backward(const Tensor& grad, const Tensor& input,
                                                  const std::vector<int64_t>& output_shape) {
    const Tensor detached = ops::detach(input);
    const bool batched = detached.dim() == 4;
    const Tensor value = batched ? detached : ops::unsqueeze(detached, 0);
    const std::vector<int64_t> pooled_shape = trailing_extents(output_shape, 2);
    const Tensor indices = std::get<1>(ops::adaptive_max_pool2d_with_indices(
        value, pooled_shape));
    return max_pool_double_backward(grad, batched ? indices : ops::squeeze(indices, 0), 2);
}

// embedding_dense_backward differentiated in grad_output: the kernel adds
// row i of grad_output into table row indices[i] (dividing by the row's
// occurrence count under scale_grad_by_freq, and skipping padding_idx), so
// the transpose reads those table rows back, with the same scale and mask.
inline Tensor embedding_dense_double_backward(const Tensor& grad, const Tensor& indices,
                                              int64_t num_weights, int64_t padding_idx,
                                              bool scale_grad_by_freq) {
    namespace d = shape_bwd_detail;
    const std::vector<int64_t> index_shape = indices.shape();
    std::vector<int64_t> out_shape = index_shape;
    const std::vector<int64_t> grad_shape = grad.shape();
    out_shape.insert(out_shape.end(), grad_shape.begin() + 1, grad_shape.end());
    if (indices.numel() == 0) {
        return ops::zeros(out_shape, grad.dtype(), grad.device());
    }

    const Tensor flat = ops::reshape(d::int64_indices(indices), {-1});
    const Tensor kept = d::mask_like(ops::ne(flat, Scalar(padding_idx)), grad);
    // Negative indices count from the end of the table, like the kernel.
    const Tensor rows = ops::clamp_min(
        ops::where(ops::lt(flat, Scalar(static_cast<int64_t>(0))),
                   ops::add(flat, Scalar(num_weights)), flat),
        Scalar(static_cast<int64_t>(0)));
    Tensor picked = ops::index_select(grad, 0, rows);
    Tensor scale = kept;
    if (scale_grad_by_freq) {
        const Tensor counts = d::row_counts(rows, num_weights, grad, kept);
        scale = ops::div(kept, ops::clamp_min(ops::index_select(counts, 0, rows),
                                              Scalar(1.0)));
    }
    picked = ops::mul(picked, d::as_column(scale, picked));
    return ops::reshape(picked, out_shape);
}

namespace shape_bwd_detail {

// Per-position facts of a sum / mean bag backward: which positions feed a
// bag (a bag owner and a row other than padding_idx), the bag and row each
// reads, and the factor each contributes before mean scaling.
struct BagPositions {
    Tensor rows;    // int64 table row per position, clamped into range
    Tensor bags;    // int64 bag per position, clamped into range
    Tensor scale;   // 0/1 mask times 1/occurrences under scale_grad_by_freq
};

inline BagPositions bag_positions(const Tensor& like, const Tensor& indices,
                                  const Tensor& offset2bag, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t padding_idx) {
    BagPositions p;
    const Tensor flat = ops::reshape(int64_indices(indices), {-1});
    const Tensor bag = ops::reshape(int64_indices(offset2bag), {-1});
    const Scalar zero(static_cast<int64_t>(0));
    p.rows = ops::clamp_min(flat, zero);
    p.bags = ops::clamp_min(bag, zero);
    p.scale = ops::mul(mask_like(ops::ge(bag, zero), like),
                       mask_like(ops::ne(flat, Scalar(padding_idx)), like));
    if (scale_grad_by_freq) {
        // The occurrence count includes every position of the row.
        const Tensor counts =
            row_counts(p.rows, num_weights, like, ops::ones_like(flat, like.dtype()));
        p.scale = ops::div(p.scale, ops::clamp_min(ops::index_select(counts, 0, p.rows),
                                                   Scalar(1.0)));
    }
    return p;
}

}  // namespace shape_bwd_detail

// _embedding_bag_dense_backward differentiated in grad_output (the per-bag
// gradient): the kernel adds bag rows into table rows, so the transpose reads
// table rows of `grad` back into bags (sum or mean, optionally weighted per
// sample), or picks the recorded argmax row of every column for max mode.
inline Tensor embedding_bag_dense_double_backward(
        const Tensor& grad, const Tensor& indices, const Tensor& offset2bag,
        const Tensor& bag_size, const Tensor& maximum_indices, int64_t num_weights,
        bool scale_grad_by_freq, int64_t mode, const std::optional<Tensor>& per_sample_weights,
        int64_t padding_idx) {
    namespace d = shape_bwd_detail;
    constexpr int64_t kBagMean = 1;
    constexpr int64_t kBagMax = 2;
    const int64_t num_bags = bag_size.numel();
    const int64_t width = grad.size(1);
    const Scalar zero(static_cast<int64_t>(0));
    if (num_bags == 0) {
        return ops::zeros({0, width}, grad.dtype(), grad.device());
    }
    if (mode == kBagMax) {
        const Tensor nonempty = d::mask_like(ops::gt(ops::reshape(bag_size, {-1, 1}), zero), grad);
        const Tensor rows = ops::clamp_min(d::int64_indices(maximum_indices), zero);
        return ops::mul(ops::gather(grad, 0, rows), nonempty);
    }
    if (indices.numel() == 0) {
        return ops::zeros({num_bags, width}, grad.dtype(), grad.device());
    }

    const d::BagPositions p =
        d::bag_positions(grad, indices, offset2bag, num_weights, scale_grad_by_freq, padding_idx);
    Tensor scale = p.scale;
    if (per_sample_weights.has_value() && per_sample_weights->defined() &&
        per_sample_weights->numel() > 0) {
        scale = ops::mul(scale, ops::to(ops::reshape(*per_sample_weights, {-1}), grad.dtype()));
    }
    if (mode == kBagMean) {
        const Tensor sizes = ops::index_select(ops::to(ops::reshape(bag_size, {-1}), grad.dtype()),
                                               0, p.bags);
        scale = ops::div(scale, ops::clamp_min(sizes, Scalar(1.0)));
    }
    const Tensor weighted = ops::mul(ops::index_select(grad, 0, p.rows), d::as_column(scale, grad));
    return ops::index_add(ops::zeros({num_bags, width}, grad.dtype(), grad.device()), 0,
                          p.bags, weighted);
}

// d/d per_sample_weights of the sum-mode bag backward: position i adds
// weight_i * grad_output[bag_i] into table row indices[i], so its slope is
// the dot product of that bag row with the incoming gradient's table row.
inline Tensor embedding_bag_dense_double_backward_per_sample_weights(
        const Tensor& grad, const Tensor& grad_output, const Tensor& indices,
        const Tensor& offset2bag, int64_t num_weights, bool scale_grad_by_freq, int64_t mode,
        int64_t padding_idx) {
    namespace d = shape_bwd_detail;
    constexpr int64_t kBagSum = 0;
    if (mode != kBagSum) return Tensor();
    if (indices.numel() == 0) {
        return ops::zeros({0}, grad.dtype(), grad.device());
    }
    const d::BagPositions p =
        d::bag_positions(grad, indices, offset2bag, num_weights, scale_grad_by_freq, padding_idx);
    const Tensor dots = ops::sum(
        ops::mul(ops::index_select(grad_output, 0, p.bags), ops::index_select(grad, 0, p.rows)),
        std::vector<int64_t>{1}, false);
    return ops::mul(dots, p.scale);
}

// _embedding_bag_per_sample_weights_backward writes, for every position i,
// dot(grad_output[bag_i], weight[row_i]).  Its derivatives in grad_output
// and in weight scatter the incoming per-position gradient `grad` the other
// way: a position's weight row goes to its bag, its bag row to its table row.
inline Tensor embedding_bag_psw_double_backward_grad_output(
        const Tensor& grad, const Tensor& weight, const Tensor& indices,
        const Tensor& offset2bag, const std::vector<int64_t>& grad_output_shape,
        int64_t padding_idx) {
    namespace d = shape_bwd_detail;
    if (offset2bag.numel() == 0) return Tensor();
    const int64_t num_bags = grad_output_shape[0];
    const int64_t width = grad_output_shape[1];
    Tensor out = ops::zeros({num_bags, width}, grad.dtype(), grad.device());
    if (indices.numel() == 0 || num_bags == 0) return out;
    const d::BagPositions p = d::bag_positions(grad, indices, offset2bag, 0, false, padding_idx);
    const Tensor rows = ops::index_select(ops::to(weight, grad.dtype()), 0, p.rows);
    const Tensor weighted =
        ops::mul(rows, d::as_column(ops::mul(ops::reshape(grad, {-1}), p.scale), rows));
    return ops::index_add(out, 0, p.bags, weighted);
}

inline Tensor embedding_bag_psw_double_backward_weight(
        const Tensor& grad, const Tensor& grad_output, const Tensor& indices,
        const Tensor& offset2bag, const std::vector<int64_t>& weight_shape,
        int64_t padding_idx) {
    namespace d = shape_bwd_detail;
    if (offset2bag.numel() == 0) return Tensor();
    Tensor out = ops::zeros(weight_shape, grad.dtype(), grad.device());
    if (indices.numel() == 0 || grad_output.numel() == 0) return out;
    const d::BagPositions p = d::bag_positions(grad, indices, offset2bag, 0, false, padding_idx);
    const Tensor bags = ops::index_select(grad_output, 0, p.bags);
    const Tensor weighted =
        ops::mul(bags, d::as_column(ops::mul(ops::reshape(grad, {-1}), p.scale), bags));
    return ops::index_add(out, 0, p.rows, weighted);
}

}  // namespace tpx
}  // namespace tensorplay
