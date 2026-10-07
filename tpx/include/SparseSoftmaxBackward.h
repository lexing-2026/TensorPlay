#pragma once

#include "Autograd.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <map>
#include <vector>

namespace tensorplay {
namespace tpx {
namespace sparse_softmax_detail {

inline Tensor canonical(const Tensor& input) {
    return input.is_coalesced() ? input : input.coalesce();
}

inline Tensor from_values(const Tensor& pattern, const Tensor& values);
inline Tensor values_at(const Tensor& input, const Tensor& pattern);

// Values and COO construction are adjoints on the stored coordinates.
// Keep their history separate from the raw storage accessors.
struct ValuesBackward : Node {
    explicit ValuesBackward(const Tensor& pattern) : pattern_(pattern.detach()) {}
    variable_list apply(variable_list&& inputs) override {
        if (!inputs[0].defined()) return {Tensor()};
        return {from_values(pattern_, inputs[0])};
    }
    Tensor pattern_;
};

struct FromValuesBackward : Node {
    explicit FromValuesBackward(const Tensor& pattern) : pattern_(pattern.detach()) {}
    variable_list apply(variable_list&& inputs) override {
        if (!inputs[0].defined()) return {Tensor()};
        return {values_at(inputs[0], pattern_)};
    }
    Tensor pattern_;
};

inline Tensor values(const Tensor& input) {
    Tensor pattern = canonical(input);
    Tensor result = pattern._values().detach();
    if (GradMode::is_enabled() && input.requires_grad()) {
        auto node = std::make_shared<ValuesBackward>(pattern);
        node->add_next_edge_list(collect_next_edges(input));
        impl::set_requires_grad(result, true);
        impl::set_grad_fn(result, std::move(node));
    }
    return result;
}

inline Tensor from_values(const Tensor& pattern, const Tensor& payload) {
    Tensor result = Tensor::make_sparse_coo_tensor(
        pattern._indices(), payload, pattern.shape(), true);
    if (GradMode::is_enabled() && payload.requires_grad()) {
        auto node = std::make_shared<FromValuesBackward>(pattern);
        node->add_next_edge_list(collect_next_edges(payload));
        impl::set_requires_grad(result, true);
        impl::set_grad_fn(result, std::move(node));
    }
    return result;
}

inline Tensor index_tensor(const std::vector<int64_t>& entries, Device device) {
    Tensor result = Tensor::empty({static_cast<int64_t>(entries.size())}, DType::Int64);
    std::copy(entries.begin(), entries.end(), result.data_ptr<int64_t>());
    return result.to(device);
}

inline std::vector<int64_t> coordinate(const Tensor& indices, int64_t entry,
                                       int64_t omitted = -1) {
    std::vector<int64_t> result;
    const int64_t* data = indices.data_ptr<int64_t>();
    for (int64_t d = 0; d < indices.size(0); ++d) {
        if (d != omitted) result.push_back(data[d * indices.size(1) + entry]);
    }
    return result;
}

// Match coordinates before reducing: missing gradient entries contribute
// zero, and entries outside the output's support contribute nothing.
inline Tensor values_at(const Tensor& input, const Tensor& pattern) {
    Tensor target_indices = pattern._indices().to(Device(DeviceType::CPU)).contiguous();
    std::vector<int64_t> positions(static_cast<size_t>(pattern._nnz()));
    if (!input.is_sparse()) {
        int64_t cells = 1;
        for (int64_t d = 0; d < pattern.sparse_dim(); ++d) cells *= pattern.size(d);
        std::vector<int64_t> shape{cells};
        for (int64_t d = pattern.sparse_dim(); d < pattern.dim(); ++d)
            shape.push_back(pattern.size(d));
        for (int64_t i = 0; i < pattern._nnz(); ++i) {
            int64_t offset = 0;
            const auto coords = coordinate(target_indices, i);
            for (int64_t d = 0; d < pattern.sparse_dim(); ++d)
                offset = offset * pattern.size(d) + coords[static_cast<size_t>(d)];
            positions[static_cast<size_t>(i)] = offset;
        }
        return ops::index_select(ops::reshape(input, shape), 0,
                                 index_tensor(positions, input.device()));
    }
    Tensor source = canonical(input);
    Tensor source_indices = source._indices().to(Device(DeviceType::CPU)).contiguous();
    std::map<std::vector<int64_t>, int64_t> lookup;
    for (int64_t i = 0; i < source._nnz(); ++i)
        lookup.emplace(coordinate(source_indices, i), i);
    for (int64_t i = 0; i < pattern._nnz(); ++i) {
        const auto found = lookup.find(coordinate(target_indices, i));
        positions[static_cast<size_t>(i)] = found == lookup.end() ? source._nnz() : found->second;
    }
    Tensor payload = values(input);
    auto zero_shape = static_cast<std::vector<int64_t>>(payload.shape());
    zero_shape[0] = 1;
    Tensor padded = ops::cat({payload, Tensor::zeros(zero_shape, payload.dtype(), payload.device())}, 0);
    return ops::index_select(padded, 0, index_tensor(positions, input.device()));
}

// Sum and broadcast within a COO pool using compact segment IDs. Storage
// scales with nnz and the dense payload, independent of the sparse extents.
inline Tensor pool_sum(const Tensor& payload, const Tensor& pattern, int64_t dim) {
    if (dim < 0) dim += pattern.dim();
    if (dim >= pattern.sparse_dim())
        return ops::sum(payload, std::vector<int64_t>{dim - pattern.sparse_dim() + 1}, true);
    Tensor indices = pattern._indices().to(Device(DeviceType::CPU)).contiguous();
    std::map<std::vector<int64_t>, int64_t> pools;
    std::vector<int64_t> entries(static_cast<size_t>(pattern._nnz()));
    for (int64_t i = 0; i < pattern._nnz(); ++i) {
        auto inserted = pools.emplace(coordinate(indices, i, dim), static_cast<int64_t>(pools.size()));
        entries[static_cast<size_t>(i)] = inserted.first->second;
    }
    Tensor pool_ids = index_tensor(entries, payload.device());
    auto shape = static_cast<std::vector<int64_t>>(payload.shape());
    shape[0] = static_cast<int64_t>(pools.size());
    Tensor sums = ops::index_add(Tensor::zeros(shape, payload.dtype(), payload.device()),
                                  0, pool_ids, payload);
    return ops::index_select(sums, 0, pool_ids);
}

}  // namespace sparse_softmax_detail

inline Tensor sparse_softmax_double_backward_grad_output(
    const Tensor& grad, const Tensor& grad_output, const Tensor& output,
    int64_t dim, bool logarithmic) {
    using namespace sparse_softmax_detail;
    Tensor pattern = canonical(output);
    Tensor incoming = values_at(grad, pattern);
    Tensor probability = logarithmic ? ops::exp(values(output)) : values(output);
    Tensor weighted = ops::mul(probability, incoming);
    Tensor result = logarithmic
        ? ops::sub(incoming, pool_sum(weighted, pattern, dim))
        : ops::mul(probability, ops::sub(incoming, pool_sum(weighted, pattern, dim)));
    Tensor target = canonical(grad_output);
    return from_values(target, values_at(from_values(pattern, result), target));
}

inline Tensor sparse_softmax_double_backward_output(
    const Tensor& grad, const Tensor& grad_output, const Tensor& output,
    int64_t dim, bool logarithmic) {
    using namespace sparse_softmax_detail;
    Tensor pattern = canonical(output);
    Tensor incoming = values_at(grad, pattern);
    Tensor g = values_at(grad_output, pattern);
    Tensor out = values(output);
    Tensor result;
    if (logarithmic) {
        result = ops::neg(ops::mul(ops::mul(incoming, ops::exp(out)), pool_sum(g, pattern, dim)));
    } else {
        result = ops::sub(
            ops::sub(ops::mul(incoming, g),
                     ops::mul(incoming, pool_sum(ops::mul(out, g), pattern, dim))),
            ops::mul(g, pool_sum(ops::mul(out, incoming), pattern, dim)));
    }
    return from_values(pattern, result);
}

}  // namespace tpx
}  // namespace tensorplay
