#pragma once

#include <algorithm>
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay::indexing::native {

inline bool has_contiguous_subspace(const std::vector<Tensor>& indices) {
    auto first = std::find_if(indices.begin(), indices.end(),
                              [](const Tensor& t) { return t.defined(); });
    if (first == indices.end()) return true;
    auto last = std::find_if(indices.rbegin(), indices.rend(),
                             [](const Tensor& t) { return t.defined(); });
    return std::find_if(first, last.base(),
                       [](const Tensor& t) { return !t.defined(); }) == last.base();
}

inline std::pair<Tensor, std::vector<Tensor>> prepare_indices(
    Tensor self, const std::vector<std::optional<Tensor>>& original,
    bool ensure_same_device = false,
    bool normalize_int32 = true) {
    TP_CHECK_INDEX(original.size() <= static_cast<size_t>(self.dim()),
                   "too many indices for tensor of dimension ", self.dim());
    std::vector<Tensor> indices;
    for (const auto& optional : original) {
        if (!optional.has_value() || !optional->defined()) {
            indices.emplace_back();
            continue;
        }
        Tensor index = *optional;
        const auto dtype = index.dtype();
        TP_CHECK_INDEX(dtype == DType::Int64 || dtype == DType::Int32 ||
                       dtype == DType::Bool || dtype == DType::UInt8,
                       "tensors used as indices must be long, int, byte or bool tensors");
        if (dtype == DType::Bool || dtype == DType::UInt8) {
            if (dtype == DType::UInt8) {
                TP_WARN("indexing with dtype uint8 is deprecated; use bool instead");
            }
            for (int64_t d = 0; d < index.dim(); ++d) {
                const int64_t source_dim = static_cast<int64_t>(indices.size()) + d;
                TP_CHECK_INDEX(source_dim < self.dim() &&
                               index.size(d) == self.size(source_dim),
                               "The shape of the mask does not match the indexed tensor");
            }
            Tensor nonzero = tpx::ops::nonzero(index);
            if (ensure_same_device && nonzero.device() != self.device()) {
                nonzero = nonzero.to(self.device());
            }
            for (int64_t d = 0; d < index.dim(); ++d) {
                indices.emplace_back(nonzero.select(1, d));
            }
        } else {
            if (ensure_same_device && index.device() != self.device()) {
                index = index.to(self.device());
            }
            indices.emplace_back(std::move(index));
        }
    }

    std::vector<int64_t> replacement;
    for (const auto& index : indices) {
        if (!index.defined()) continue;
        const auto shape = static_cast<std::vector<int64_t>>(index.shape());
        const size_t ndim = std::max(replacement.size(), shape.size());
        std::vector<int64_t> broadcast(ndim, 1);
        for (size_t i = 0; i < ndim; ++i) {
            const int64_t a = i < replacement.size()
                ? replacement[replacement.size() - 1 - i] : 1;
            const int64_t b = i < shape.size() ? shape[shape.size() - 1 - i] : 1;
            TP_CHECK_INDEX(a == b || a == 1 || b == 1,
                           "shape mismatch: indexing tensors could not be broadcast together");
            broadcast[ndim - 1 - i] = a == 1 ? b : a;
        }
        replacement = std::move(broadcast);
    }
    for (auto& index : indices) {
        if (index.defined()) index = index.expand(replacement);
    }
    TP_CHECK_INDEX(indices.size() <= static_cast<size_t>(self.dim()),
                   "too many indices for tensor of dimension ", self.dim());
    indices.resize(static_cast<size_t>(self.dim()));
    if (!has_contiguous_subspace(indices)) {
        std::vector<int64_t> permutation;
        std::vector<Tensor> reordered;
        for (int defined = 1; defined >= 0; --defined) {
            for (size_t d = 0; d < indices.size(); ++d) {
                if (indices[d].defined() == static_cast<bool>(defined)) {
                    permutation.push_back(static_cast<int64_t>(d));
                    reordered.push_back(indices[d]);
                }
            }
        }
        self = self.permute(permutation);
        indices = std::move(reordered);
    }
    if (normalize_int32) {
        for (auto& index : indices) {
            if (index.defined() && index.dtype() == DType::Int32) {
                index = index.to(DType::Int64);
            }
        }
    }
    return {std::move(self), std::move(indices)};
}

inline std::vector<int64_t> indexed_shape(
    const Tensor& self, const std::vector<std::optional<Tensor>>& original) {
    auto [source, indices] = prepare_indices(self, original, false, false);
    int64_t before = 0;
    int64_t indexed = 0;
    std::vector<int64_t> replacement;
    for (const auto& index : indices) {
        if (!index.defined()) {
            if (indexed == 0) ++before;
        } else {
            ++indexed;
            replacement = static_cast<std::vector<int64_t>>(index.shape());
        }
    }
    auto shape = static_cast<std::vector<int64_t>>(source.shape());
    shape.erase(shape.begin() + before, shape.begin() + before + indexed);
    shape.insert(shape.begin() + before, replacement.begin(), replacement.end());
    return shape;
}

struct AdvancedIndex {
    Tensor source;
    std::vector<Tensor> indices;
    std::vector<int64_t> indexed_sizes;
    std::vector<int64_t> indexed_strides;

    AdvancedIndex(const Tensor& self,
                  const std::vector<std::optional<Tensor>>& original) {
        auto [src, expanded] = prepare_indices(self, original, true);
        int64_t before = 0, after = 0, indexed = 0;
        std::vector<int64_t> replacement;
        for (size_t d = 0; d < expanded.size(); ++d) {
            if (!expanded[d].defined()) {
                if (indexed == 0) ++before;
                else ++after;
            } else {
                ++indexed;
                replacement = static_cast<std::vector<int64_t>>(expanded[d].shape());
                indexed_sizes.push_back(src.size(d));
                indexed_strides.push_back(src.strides()[d] * elementSize(src.dtype()));
            }
        }
        TP_CHECK_INDEX(
            std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) == indexed_sizes.end() ||
            std::find(replacement.begin(), replacement.end(), 0) != replacement.end(),
            "index is out of bounds for dimension with size 0");
        auto shape = static_cast<std::vector<int64_t>>(src.shape());
        auto strides = src.strides();
        shape.erase(shape.begin() + before, shape.begin() + before + indexed);
        strides.erase(strides.begin() + before, strides.begin() + before + indexed);
        shape.insert(shape.begin() + before, replacement.begin(), replacement.end());
        strides.insert(strides.begin() + before, replacement.size(), 0);
        source = src.as_strided(shape, strides);
        for (const auto& index : expanded) {
            if (!index.defined()) continue;
            std::vector<int64_t> index_shape(static_cast<size_t>(before), 1);
            const auto original_shape = static_cast<std::vector<int64_t>>(index.shape());
            index_shape.insert(index_shape.end(), original_shape.begin(), original_shape.end());
            index_shape.insert(index_shape.end(), static_cast<size_t>(after), 1);
            indices.push_back(index.reshape(index_shape));
        }
    }
};

} // namespace tensorplay::indexing::native
