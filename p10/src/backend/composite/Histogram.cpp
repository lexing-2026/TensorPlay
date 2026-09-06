// Composite kernel: histc.
// Linear bin mapping with bincount accumulation.  Boundary semantics keep the
// rightmost bin closed, while NaNs and out-of-range values are skipped.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

void histc_expand_constant_range(DType dtype, double& lo, double& hi) {
    switch (dtype) {
        case DType::Float64:
            lo = std::min(
                std::nexttoward(lo, std::numeric_limits<double>::lowest()),
                lo - 1.0);
            hi = std::max(
                std::nexttoward(hi, std::numeric_limits<double>::max()),
                hi + 1.0);
            break;
        case DType::Float32:
            lo = std::min(
                static_cast<double>(std::nexttoward(
                    static_cast<float>(lo),
                    std::numeric_limits<float>::lowest())),
                lo - 1.0);
            hi = std::max(
                static_cast<double>(std::nexttoward(
                    static_cast<float>(hi),
                    std::numeric_limits<float>::max())),
                hi + 1.0);
            break;
        default:
            lo -= 1.0;
            hi += 1.0;
            break;
    }
}

bool histogramdd_supported_dtype(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::BFloat16 ||
           dtype == DType::Float32 || dtype == DType::Float64;
}

void histogramdd_check_input(const Tensor& self) {
    if (self.dim() < 2) {
        TP_THROW(ValueError,
                 "histogramdd: input tensor should have at least 2 dimensions, but got ",
                 self.dim());
    }
    if (!histogramdd_supported_dtype(self.dtype())) {
        TP_THROW(TypeError,
                 "histogramdd(): input tensor must have a floating-point dtype");
    }
}

int64_t histogramdd_sample_count(const Tensor& self) {
    int64_t samples = 1;
    for (int64_t dim = 0; dim + 1 < self.dim(); ++dim) {
        const int64_t size = self.size(dim);
        if (size != 0 && samples > std::numeric_limits<int64_t>::max() / size) {
            TP_THROW(RuntimeError, "histogramdd: input shape is too large");
        }
        samples *= size;
    }
    return samples;
}

void histogramdd_check_weight(const Tensor& self, int64_t samples,
                              const std::optional<Tensor>& weight) {
    if (!weight.has_value()) return;
    if (weight->dtype() != self.dtype()) {
        TP_THROW(ValueError,
                 "histogramdd: if weight tensor is provided, input tensor and weight tensor should have the same dtype");
    }
    if (weight->device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "histogramdd: weight tensor must be on the same device as the input");
    }

    std::vector<int64_t> expected =
        static_cast<std::vector<int64_t>>(self.shape());
    expected.pop_back();
    std::vector<int64_t> actual =
        static_cast<std::vector<int64_t>>(weight->shape());
    if (actual.empty()) actual = {1};
    if (actual != expected) {
        TP_THROW(ValueError,
                 "histogramdd: if weight tensor is provided it should have the same shape as the input tensor excluding its innermost dimension");
    }
    if (weight->numel() != samples) {
        TP_THROW(ValueError,
                 "histogramdd: weight tensor has an invalid number of elements");
    }
}

void histogramdd_check_bin_count(const Tensor& self,
                                 const std::vector<int64_t>& bin_counts) {
    const int64_t dimensions = self.size(-1);
    if (static_cast<int64_t>(bin_counts.size()) != dimensions) {
        TP_THROW(ValueError,
                 "histogramdd: expected ", dimensions,
                 " bin counts for a ", dimensions,
                 "-dimensional histogram but got ", bin_counts.size());
    }
    for (int64_t dim = 0; dim < dimensions; ++dim) {
        if (bin_counts[static_cast<size_t>(dim)] <= 0) {
            TP_THROW(ValueError,
                     "histogramdd: bins must be positive for dimension ", dim);
        }
    }
}

void histogramdd_check_bin_edges(const Tensor& self,
                                 const std::vector<Tensor>& bin_edges) {
    const int64_t dimensions = self.size(-1);
    if (static_cast<int64_t>(bin_edges.size()) != dimensions) {
        TP_THROW(ValueError,
                 "histogramdd: expected ", dimensions,
                 " sequences of bin edges for a ", dimensions,
                 "-dimensional histogram but got ", bin_edges.size());
    }
    for (int64_t dim = 0; dim < dimensions; ++dim) {
        const Tensor& edge = bin_edges[static_cast<size_t>(dim)];
        if (edge.dtype() != self.dtype()) {
            TP_THROW(ValueError,
                     "histogramdd: input tensor and bins tensors should have the same dtype for dimension ",
                     dim);
        }
        if (edge.device() != self.device()) {
            TP_THROW(DeviceMismatchError,
                     "histogramdd: bins tensor must be on the same device as the input");
        }
        if (edge.dim() != 1) {
            TP_THROW(ValueError,
                     "histogramdd: bins tensor should have one dimension for dimension ",
                     dim, ", but got ", edge.dim());
        }
        if (edge.numel() < 2) {
            TP_THROW(ValueError,
                     "histogramdd: bins must contain at least two edges for dimension ",
                     dim);
        }
    }
}

std::pair<std::vector<double>, std::vector<double>>
histogramdd_outer_edges(const Tensor& reshaped,
                        const std::optional<std::vector<double>>& range) {
    const int64_t dimensions = reshaped.size(1);
    std::vector<double> leftmost(static_cast<size_t>(dimensions), 0.0);
    std::vector<double> rightmost(static_cast<size_t>(dimensions), 1.0);

    if (range.has_value()) {
        if (static_cast<int64_t>(range->size()) != 2 * dimensions) {
            TP_THROW(ValueError,
                     "histogramdd: range should have ", 2 * dimensions,
                     " elements, but got ", range->size());
        }
        for (int64_t dim = 0; dim < dimensions; ++dim) {
            leftmost[static_cast<size_t>(dim)] = (*range)[2 * dim];
            rightmost[static_cast<size_t>(dim)] = (*range)[2 * dim + 1];
        }
    } else if (reshaped.size(0) > 0 && dimensions > 0) {
        auto extrema = Tensor::aminmax(reshaped, {0}, false);
        const Tensor& min_values = std::get<0>(extrema);
        const Tensor& max_values = std::get<1>(extrema);
        for (int64_t dim = 0; dim < dimensions; ++dim) {
            leftmost[static_cast<size_t>(dim)] =
                min_values.select(0, dim).item().toDouble();
            rightmost[static_cast<size_t>(dim)] =
                max_values.select(0, dim).item().toDouble();
        }
    }

    for (int64_t dim = 0; dim < dimensions; ++dim) {
        double& lo = leftmost[static_cast<size_t>(dim)];
        double& hi = rightmost[static_cast<size_t>(dim)];
        if (!std::isfinite(lo) || !std::isfinite(hi)) {
            TP_THROW(ValueError,
                     "histogramdd: dimension ", dim,
                     " range is not finite");
        }
        if (lo > hi) {
            TP_THROW(ValueError,
                     "histogramdd: min should not exceed max for dimension ",
                     dim);
        }
        if (lo == hi) {
            lo -= 0.5;
            hi += 0.5;
        }
    }
    return {std::move(leftmost), std::move(rightmost)};
}

std::vector<Tensor> histogramdd_make_bin_edges(
        const Tensor& self, const std::vector<int64_t>& bin_counts,
        const std::optional<std::vector<double>>& range) {
    histogramdd_check_input(self);
    histogramdd_check_bin_count(self, bin_counts);

    const int64_t samples = histogramdd_sample_count(self);
    const int64_t dimensions = self.size(-1);
    const Tensor reshaped = self.reshape({samples, dimensions});
    auto outer = histogramdd_outer_edges(reshaped, range);

    std::vector<Tensor> bin_edges;
    bin_edges.reserve(static_cast<size_t>(dimensions));
    for (int64_t dim = 0; dim < dimensions; ++dim) {
        bin_edges.push_back(Tensor::linspace(
            Scalar(outer.first[static_cast<size_t>(dim)]),
            Scalar(outer.second[static_cast<size_t>(dim)]),
            bin_counts[static_cast<size_t>(dim)] + 1,
            self.dtype(), self.device()));
    }
    return bin_edges;
}

int64_t histogramdd_total_bins(const std::vector<int64_t>& bin_counts) {
    int64_t total = 1;
    for (int64_t count : bin_counts) {
        if (count != 0 && total > std::numeric_limits<int64_t>::max() / count) {
            TP_THROW(RuntimeError, "histogramdd: the bin shape is too large");
        }
        total *= count;
    }
    return total;
}

Tensor histogramdd_accumulate(const Tensor& self,
                              const std::vector<Tensor>& bin_edges,
                              const std::optional<Tensor>& weight,
                              bool density) {
    histogramdd_check_input(self);
    histogramdd_check_bin_edges(self, bin_edges);
    const int64_t samples = histogramdd_sample_count(self);
    histogramdd_check_weight(self, samples, weight);

    const int64_t dimensions = self.size(-1);
    std::vector<int64_t> bin_counts;
    bin_counts.reserve(static_cast<size_t>(dimensions));
    for (const Tensor& edge : bin_edges) {
        bin_counts.push_back(edge.numel() - 1);
    }
    const int64_t total_bins = histogramdd_total_bins(bin_counts);

    if (dimensions == 0) {
        Tensor hist = Tensor::zeros({}, self.dtype(), self.device());
        return density ? hist.div(hist.sum()) : hist;
    }

    const Tensor reshaped = self.reshape({samples, dimensions});
    Tensor flat_indices = Tensor::zeros({samples}, DType::Int64, self.device());
    Tensor in_range = Tensor::ones({samples}, DType::Bool, self.device());

    std::vector<int64_t> strides(static_cast<size_t>(dimensions), 1);
    int64_t stride = 1;
    for (int64_t dim = dimensions - 1; dim >= 0; --dim) {
        strides[static_cast<size_t>(dim)] = stride;
        stride *= bin_counts[static_cast<size_t>(dim)];
    }

    std::vector<Tensor> contiguous_edges;
    contiguous_edges.reserve(static_cast<size_t>(dimensions));
    for (const Tensor& edge : bin_edges) contiguous_edges.push_back(edge.contiguous());

    for (int64_t dim = 0; dim < dimensions; ++dim) {
        const Tensor& edge = contiguous_edges[static_cast<size_t>(dim)];
        const int64_t count = bin_counts[static_cast<size_t>(dim)];
        const Tensor values = reshaped.select(1, dim);
        const Tensor dim_in_range = Tensor::logical_and(
            values.ge(edge.narrow(0, 0, 1)),
            values.le(edge.narrow(0, count, 1)));
        in_range = Tensor::logical_and(in_range, dim_in_range);

        Tensor positions = Tensor::searchsorted(edge, values, false, true)
                                .sub(1)
                                .clamp(0, count - 1);
        if (strides[static_cast<size_t>(dim)] != 1) {
            positions = positions.mul(
                Scalar(strides[static_cast<size_t>(dim)]));
        }
        flat_indices = flat_indices.add(positions);
    }

    const Tensor selected_indices = flat_indices.masked_select(in_range);
    std::optional<Tensor> selected_weight;
    if (weight.has_value()) {
        const Tensor flat_weight = weight->reshape({samples});
        selected_weight = flat_weight.masked_select(in_range);
    }
    Tensor hist = Tensor::bincount(
        selected_indices, selected_weight, total_bins).to(self.dtype());
    hist = hist.reshape(bin_counts);

    if (density) {
        hist = hist.div(hist.sum());
        for (int64_t dim = 0; dim < dimensions; ++dim) {
            const int64_t count = bin_counts[static_cast<size_t>(dim)];
            const Tensor widths = contiguous_edges[static_cast<size_t>(dim)]
                                      .narrow(0, 1, count)
                                      .sub(contiguous_edges[static_cast<size_t>(dim)]
                                               .narrow(0, 0, count));
            std::vector<int64_t> width_shape(static_cast<size_t>(dimensions), 1);
            width_shape[static_cast<size_t>(dim)] = count;
            hist = hist.div(widths.reshape(width_shape));
        }
    }
    return hist;
}

} // anonymous namespace

Tensor histc_native(const Tensor& self, int64_t bins, const Scalar& min,
                    const Scalar& max) {
    if (bins <= 0) TP_THROW(RuntimeError, "histc(): bins must be positive");
    if (!isFloatingType(self.dtype())) {
        TP_THROW(NotImplementedError, "histc(): expected a floating-point tensor, got ",
                 toString(self.dtype()));
    }
    double lo = min.toDouble();
    double hi = max.toDouble();
    if (lo == hi && self.numel() > 0) {
        auto extrema = ops::aminmax(self);
        lo = std::get<0>(extrema).item().toDouble();
        hi = std::get<1>(extrema).item().toDouble();
    }
    if (lo == hi) {
        histc_expand_constant_range(self.dtype(), lo, hi);
        histc_expand_constant_range(self.dtype(), lo, hi);
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        TP_THROW(RuntimeError, "histc: range of [", lo, ", ", hi,
                 "] is not finite");
    }
    if (!(lo < hi)) {
        TP_THROW(RuntimeError, "histc: max must be larger than min");
    }

    const Tensor flat = ops::reshape(self, {-1});
    const Tensor in_range = ops::logical_and(ops::ge(flat, Scalar(lo)),
                                             ops::le(flat, Scalar(hi)));
    const Tensor safe = Tensor::where(in_range, flat, Tensor::zeros_like(flat));
    Tensor idx = ops::div(
        ops::mul(ops::sub(safe, Scalar(lo)), Scalar(bins)),
        Scalar(hi - lo)).to(DType::Int64);
    idx = ops::clamp(idx, Scalar(int64_t(0)), Scalar(bins - 1));
    const Tensor counted = ops::masked_select(idx, in_range);
    const Tensor counts = ops::bincount(counted, std::nullopt, bins);
    return counts.to(self.dtype());
}

std::vector<Tensor> histogramdd_bin_edges_native(
        const Tensor& self, const std::vector<int64_t>& bins,
        std::optional<std::vector<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    (void)weight;
    (void)density;
    return histogramdd_make_bin_edges(self, bins, range);
}

Tensor histogramdd_from_bin_cts_native(
        const Tensor& self, const std::vector<int64_t>& bins,
        std::optional<std::vector<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    std::vector<Tensor> bin_edges =
        histogramdd_make_bin_edges(self, bins, range);
    return histogramdd_accumulate(self, bin_edges, weight, density);
}

Tensor histogramdd_from_bin_tensors_native(
        const Tensor& self, const std::vector<Tensor>& bins,
        const std::optional<Tensor>& weight, bool density) {
    histogramdd_check_input(self);
    histogramdd_check_bin_edges(self, bins);
    std::vector<Tensor> contiguous_bins;
    contiguous_bins.reserve(bins.size());
    for (const Tensor& edge : bins) contiguous_bins.push_back(edge.contiguous());
    return histogramdd_accumulate(self, contiguous_bins, weight, density);
}

std::tuple<Tensor, std::vector<Tensor>> histogramdd_native(
        const Tensor& self, const std::vector<int64_t>& bins,
        std::optional<std::vector<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    std::vector<Tensor> bin_edges =
        histogramdd_make_bin_edges(self, bins, range);
    Tensor hist = histogramdd_accumulate(self, bin_edges, weight, density);
    return {std::move(hist), std::move(bin_edges)};
}

std::tuple<Tensor, std::vector<Tensor>> histogramdd_int_bins_native(
        const Tensor& self, int64_t bins,
        std::optional<std::vector<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    histogramdd_check_input(self);
    std::vector<int64_t> bin_counts(
        static_cast<size_t>(self.size(-1)), bins);
    return histogramdd_native(self, bin_counts, std::move(range), weight, density);
}

std::tuple<Tensor, std::vector<Tensor>> histogramdd_tensorlist_bins_native(
        const Tensor& self, const std::vector<Tensor>& bins,
        std::optional<std::vector<double>> range,
        const std::optional<Tensor>& weight, bool density) {
    (void)range;
    Tensor hist = histogramdd_from_bin_tensors_native(self, bins, weight, density);
    return {std::move(hist), bins};
}

TENSORPLAY_LIBRARY_IMPL(Composite, HistogramComposite) {
    m.impl("histc", histc_native);
    m.impl("_histogramdd_bin_edges", histogramdd_bin_edges_native);
    m.impl("_histogramdd_from_bin_cts", histogramdd_from_bin_cts_native);
    m.impl("_histogramdd_from_bin_tensors", histogramdd_from_bin_tensors_native);
    m.impl("histogramdd", histogramdd_native);
    m.impl("histogramdd.int_bins", histogramdd_int_bins_native);
    m.impl("histogramdd.TensorList_bins", histogramdd_tensorlist_bins_native);
}

} // namespace composite
} // namespace tensorplay
