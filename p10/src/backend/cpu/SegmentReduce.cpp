#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

// Segment reductions: max / mean / min / sum / prod.
enum class SegmentReduction { Max, Mean, Min, Sum, Prod };

template <typename T>
struct TypeTag { using type = T; };

SegmentReduction get_segment_reduction(std::string_view reduce) {
    if (reduce == "max" || reduce == "amax") return SegmentReduction::Max;
    if (reduce == "mean") return SegmentReduction::Mean;
    if (reduce == "min" || reduce == "amin") return SegmentReduction::Min;
    if (reduce == "sum") return SegmentReduction::Sum;
    if (reduce == "prod") return SegmentReduction::Prod;
    TP_THROW(RuntimeError,
             "reduce argument must be either sum, prod, mean, amax or amin, got ",
             std::string(reduce));
}

int64_t wrap_axis(int64_t axis, int64_t ndim) {
    const int64_t dim_min = -ndim;
    const int64_t dim_max = ndim - 1;
    TP_CHECK(axis >= dim_min && axis <= dim_max,
             "Dimension out of range (expected to be in range of [", dim_min,
             ", ", dim_max, "], but got ", axis, ")");
    return axis < 0 ? axis + ndim : axis;
}

// Dispatch over the floating types a segment reduction accepts.
template <typename F>
void dispatch_segment_dtype(DType dtype, F&& f) {
    switch (dtype) {
        case DType::Float32: f(TypeTag<float>{}); return;
        case DType::Float64: f(TypeTag<double>{}); return;
        case DType::Float16: f(TypeTag<Half>{}); return;
        case DType::BFloat16: f(TypeTag<BFloat16>{}); return;
        default:
            TP_THROW(NotImplementedError,
                     "segment_reduce: unsupported input dtype");
    }
}

// One pass over the segments.  `is_offsets_like` selects between offsets
// (cumulative boundaries, one extra entry per outer row) and lengths.
template <typename T, bool is_offsets_like = false, typename L = int64_t>
void segment_reduce_lengths_kernel(
    SegmentReduction reduction,
    const Tensor& data,
    const L* lengths_data,
    int64_t axis,
    const std::optional<Scalar>& initial,
    Tensor& output,
    int64_t segment_count,
    int64_t lengths_stride_axis) {
    int64_t outer_offset = 1, inner_offset = 1;
    for (int64_t d = 0; d < axis; d++) {
        outer_offset *= output.size(d);
    }
    for (int64_t d = axis + 1; d < output.dim(); d++) {
        inner_offset *= output.size(d);
    }
    const int64_t lengths_size_axis =
        is_offsets_like ? segment_count + 1 : segment_count;
    const auto data_stride_axis = data.stride(axis);
    const auto data_size_axis = data.size(axis);
    const auto output_stride_axis = output.stride(axis);
    const auto output_size_axis = output.size(axis);
    auto* output_data = output.data_ptr<T>();
    const auto* values_data = data.data_ptr<T>();

    for (int64_t outer_idx = 0; outer_idx < outer_offset; ++outer_idx) {
        int64_t segment_start, segment_length;
        int64_t segment_end = is_offsets_like
            ? static_cast<int64_t>(lengths_data[
                outer_idx * lengths_stride_axis * lengths_size_axis])
            : 0;
        for (int64_t dim_idx = 0; dim_idx < segment_count; ++dim_idx) {
            segment_start = segment_end;
            const int64_t lengths_idx =
                outer_idx * lengths_stride_axis * lengths_size_axis + dim_idx;
            if (is_offsets_like) {
                segment_end = static_cast<int64_t>(lengths_data[lengths_idx + 1]);
                segment_length = segment_end - segment_start;
            } else {
                segment_length = static_cast<int64_t>(lengths_data[lengths_idx]);
                segment_end += segment_length;
            }
            for (int64_t inner_idx = 0; inner_idx < inner_offset; ++inner_idx) {
                // Starting value: the user initial, else the reduction's
                // identity (NaN-propagating extrema for min/max).
                T initial_value = 0;
                if (initial.has_value()) {
                    initial_value = initial->template to<T>();
                } else if (reduction == SegmentReduction::Max) {
                    initial_value = -std::numeric_limits<T>::infinity();
                } else if (reduction == SegmentReduction::Mean ||
                           reduction == SegmentReduction::Sum) {
                    initial_value = 0;
                } else if (reduction == SegmentReduction::Min) {
                    initial_value = std::numeric_limits<T>::infinity();
                } else if (reduction == SegmentReduction::Prod) {
                    initial_value = 1;
                }

                for (int64_t j = segment_start; j < segment_end; ++j) {
                    const int64_t data_index =
                        outer_idx * data_stride_axis * data_size_axis +
                        j * data_stride_axis + inner_idx;
                    const T val = values_data[data_index];
                    if (reduction == SegmentReduction::Max) {
                        initial_value = std::isnan(static_cast<double>(val))
                            ? val
                            : std::max(initial_value, val);
                    } else if (reduction == SegmentReduction::Mean ||
                               reduction == SegmentReduction::Sum) {
                        initial_value = initial_value + val;
                    } else if (reduction == SegmentReduction::Min) {
                        initial_value = std::isnan(static_cast<double>(val))
                            ? val
                            : std::min(initial_value, val);
                    } else if (reduction == SegmentReduction::Prod) {
                        initial_value = initial_value * val;
                    }
                }

                TP_CHECK(segment_length >= 0,
                         "segment_reduce: segment length must be non-negative");

                if (segment_length == 0 && !initial.has_value() &&
                    reduction == SegmentReduction::Mean) {
                    // A mean over an empty segment is undefined without an
                    // explicit initial value.
                    initial_value = static_cast<T>(NAN);
                } else if (reduction == SegmentReduction::Mean &&
                           segment_length > 0 &&
                           !std::isnan(static_cast<double>(initial_value))) {
                    initial_value = initial_value / static_cast<T>(segment_length);
                }
                const int64_t output_index =
                    outer_idx * output_stride_axis * output_size_axis +
                    dim_idx * output_stride_axis + inner_idx;
                output_data[output_index] = initial_value;
            }
        }
    }
}

template <typename T, bool is_offsets_like = false, typename L = int64_t>
void segment_reduce_lengths_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    SegmentReduction reduction,
    const L* lengths_data,
    int64_t axis,
    const std::optional<Scalar>& initial,
    Tensor& grad_input,
    int64_t segment_count,
    int64_t lengths_stride_axis) {
    int64_t outer_offset = 1, inner_offset = 1;
    for (int64_t d = 0; d < axis; d++) {
        outer_offset *= output_contig.size(d);
    }
    for (int64_t d = axis + 1; d < output_contig.dim(); d++) {
        inner_offset *= output_contig.size(d);
    }
    const int64_t lengths_size_axis =
        is_offsets_like ? segment_count + 1 : segment_count;
    const auto data_stride_axis = data_contig.stride(axis);
    const auto data_size_axis = data_contig.size(axis);
    const auto output_stride_axis = output_contig.stride(axis);
    const auto output_size_axis = output_contig.size(axis);
    const auto* output_data = output_contig.data_ptr<T>();
    const auto* grad_data = grad_contig.data_ptr<T>();
    auto* grad_input_data = grad_input.data_ptr<T>();
    const auto* values_data = data_contig.data_ptr<T>();
    // Exclusive-product seed for the prod backward.
    T initial_prod_value;
    if (reduction == SegmentReduction::Prod) {
        initial_prod_value = initial.has_value()
            ? initial->template to<T>()
            : static_cast<T>(1);
    }

    for (int64_t outer_idx = 0; outer_idx < outer_offset; ++outer_idx) {
        int64_t segment_start, segment_length;
        int64_t segment_end = is_offsets_like
            ? static_cast<int64_t>(lengths_data[
                outer_idx * lengths_stride_axis * lengths_size_axis])
            : 0;
        for (int64_t dim_idx = 0; dim_idx < segment_count; ++dim_idx) {
            segment_start = segment_end;
            const int64_t lengths_idx =
                outer_idx * lengths_stride_axis * lengths_size_axis + dim_idx;
            if (is_offsets_like) {
                segment_end =
                    static_cast<int64_t>(lengths_data[lengths_idx + 1]);
                segment_length = segment_end - segment_start;
            } else {
                segment_length =
                    static_cast<int64_t>(lengths_data[lengths_idx]);
                segment_end += segment_length;
            }
            if (segment_length == 0) {
                continue;
            }
            for (int64_t inner_idx = 0; inner_idx < inner_offset; ++inner_idx) {
                const int64_t output_index =
                    outer_idx * output_stride_axis * output_size_axis +
                    dim_idx * output_stride_axis + inner_idx;
                if (reduction == SegmentReduction::Max ||
                    reduction == SegmentReduction::Min) {
                    // Route the output gradient to every element that
                    // attains the extremum (NaN counts as attaining); when
                    // several do, the gradient is averaged over them.
                    int64_t counter = 0;
                    for (int64_t j = segment_start; j < segment_end; ++j) {
                        const int64_t data_index =
                            outer_idx * data_stride_axis * data_size_axis +
                            j * data_stride_axis + inner_idx;
                        if (std::isnan(static_cast<double>(values_data[data_index])) ||
                            values_data[data_index] == output_data[output_index]) {
                            grad_input_data[data_index] = grad_data[output_index];
                            counter++;
                        }
                    }
                    if (counter < 2) {
                        continue;
                    }
                    for (int64_t j = segment_start; j < segment_end; ++j) {
                        const int64_t data_index =
                            outer_idx * data_stride_axis * data_size_axis +
                            j * data_stride_axis + inner_idx;
                        if (grad_input_data[data_index] > T(0)) {
                            grad_input_data[data_index] =
                                grad_input_data[data_index] / static_cast<T>(counter);
                        }
                    }
                } else if (reduction == SegmentReduction::Mean) {
                    const T grad_val =
                        grad_data[output_index] / static_cast<T>(segment_length);
                    for (int64_t j = segment_start; j < segment_end; ++j) {
                        const int64_t data_index =
                            outer_idx * data_stride_axis * data_size_axis +
                            j * data_stride_axis + inner_idx;
                        grad_input_data[data_index] = grad_val;
                    }
                } else if (reduction == SegmentReduction::Sum) {
                    const T grad_val = grad_data[output_index];
                    for (int64_t j = segment_start; j < segment_end; ++j) {
                        const int64_t data_index =
                            outer_idx * data_stride_axis * data_size_axis +
                            j * data_stride_axis + inner_idx;
                        grad_input_data[data_index] = grad_val;
                    }
                } else if (reduction == SegmentReduction::Prod) {
                    const T grad_val =
                        grad_data[output_index] * output_data[output_index];
                    for (int64_t j = segment_start; j < segment_end; ++j) {
                        const int64_t data_index =
                            outer_idx * data_stride_axis * data_size_axis +
                            j * data_stride_axis + inner_idx;
                        if (std::isnan(static_cast<double>(values_data[data_index])) ||
                            values_data[data_index] == T(0)) {
                            // Zero/NaN participation needs the exclusive
                            // product computed explicitly.
                            T exclusive_prod = initial_prod_value;
                            for (int64_t k = segment_start; k < segment_end; ++k) {
                                if (k != j) {
                                    const int64_t idx =
                                        outer_idx * data_stride_axis *
                                            data_size_axis +
                                        k * data_stride_axis + inner_idx;
                                    exclusive_prod *= values_data[idx];
                                }
                            }
                            grad_input_data[data_index] =
                                grad_data[output_index] * exclusive_prod;
                        } else {
                            grad_input_data[data_index] =
                                grad_val / values_data[data_index];
                        }
                    }
                }
            }
        }
    }
}

Tensor segment_reduce_lengths_cpu(
    SegmentReduction reduction, const Tensor& data, const Tensor& lengths,
    int64_t axis, const std::optional<Scalar>& initial) {
    TP_CHECK(data.is_contiguous(), "Expected data to be contiguous.");
    TP_CHECK(lengths.is_contiguous(), "Expected lengths to be contiguous.");
    // The reduction axis is the last dimension of the boundary tensor.
    axis = lengths.dim() - 1;
    const int64_t segment_count = lengths.size(axis);
    const int64_t lengths_stride_axis = lengths.stride(axis);
    std::vector<int64_t> output_shape(
        static_cast<size_t>(data.dim()));
    for (int64_t d = 0; d < data.dim(); ++d) output_shape[d] = data.size(d);
    output_shape[axis] = segment_count;
    Tensor output = Tensor::empty(output_shape, data.dtype(), data.device());

    if (lengths.dtype() == DType::Int32) {
        const auto* lengths_data = lengths.data_ptr<int32_t>();
        dispatch_segment_dtype(data.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            segment_reduce_lengths_kernel<scalar_t>(
                reduction, data, lengths_data, axis, initial, output,
                segment_count, lengths_stride_axis);
        });
    } else {
        TP_CHECK(lengths.dtype() == DType::Int64,
                 "segment_reduce: lengths must be int32 or int64");
        const auto* lengths_data = lengths.data_ptr<int64_t>();
        dispatch_segment_dtype(data.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            segment_reduce_lengths_kernel<scalar_t>(
                reduction, data, lengths_data, axis, initial, output,
                segment_count, lengths_stride_axis);
        });
    }
    return output;
}

Tensor segment_reduce_offsets_cpu(
    SegmentReduction reduction, const Tensor& data, const Tensor& offsets,
    int64_t axis, const std::optional<Scalar>& initial) {
    TP_CHECK(data.is_contiguous(), "Expected data to be contiguous.");
    TP_CHECK(offsets.is_contiguous(), "Expected offsets to be contiguous.");
    axis = offsets.dim() - 1;
    const int64_t segment_count = offsets.size(axis) - 1;
    const int64_t offsets_stride_axis = offsets.stride(axis);
    std::vector<int64_t> output_shape(static_cast<size_t>(data.dim()));
    for (int64_t d = 0; d < data.dim(); ++d) output_shape[d] = data.size(d);
    output_shape[axis] = segment_count;
    Tensor output = Tensor::empty(output_shape, data.dtype(), data.device());

    if (offsets.dtype() == DType::Int32) {
        const auto* offsets_data = offsets.data_ptr<int32_t>();
        dispatch_segment_dtype(data.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            segment_reduce_lengths_kernel<scalar_t, /*is_offsets_like=*/true>(
                reduction, data, offsets_data, axis, initial, output,
                segment_count, offsets_stride_axis);
        });
    } else {
        TP_CHECK(offsets.dtype() == DType::Int64,
                 "segment_reduce: offsets must be int32 or int64");
        const auto* offsets_data = offsets.data_ptr<int64_t>();
        dispatch_segment_dtype(data.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            segment_reduce_lengths_kernel<scalar_t, /*is_offsets_like=*/true>(
                reduction, data, offsets_data, axis, initial, output,
                segment_count, offsets_stride_axis);
        });
    }
    return output;
}

Tensor segment_reduce_lengths_backward_cpu(
    const Tensor& grad_contig, const Tensor& output_contig,
    const Tensor& data_contig, SegmentReduction reduction,
    const Tensor& lengths_contig, int64_t axis,
    const std::optional<Scalar>& initial) {
    axis = lengths_contig.dim() - 1;
    const int64_t segment_count = lengths_contig.size(axis);
    const int64_t lengths_stride_axis = lengths_contig.stride(axis);
    std::vector<int64_t> grad_shape(static_cast<size_t>(data_contig.dim()));
    for (int64_t d = 0; d < data_contig.dim(); ++d) {
        grad_shape[d] = data_contig.size(d);
    }
    Tensor grad_input = Tensor::zeros(grad_shape, grad_contig.dtype(),
                                      grad_contig.device());

    if (lengths_contig.dtype() == DType::Int32) {
        const auto* lengths_data = lengths_contig.data_ptr<int32_t>();
        dispatch_segment_dtype(data_contig.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            segment_reduce_lengths_backward_kernel<scalar_t>(
                grad_contig, output_contig, data_contig, reduction,
                lengths_data, axis, initial, grad_input, segment_count,
                lengths_stride_axis);
        });
    } else {
        TP_CHECK(lengths_contig.dtype() == DType::Int64,
                 "segment_reduce: lengths must be int32 or int64");
        const auto* lengths_data = lengths_contig.data_ptr<int64_t>();
        dispatch_segment_dtype(data_contig.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            segment_reduce_lengths_backward_kernel<scalar_t>(
                grad_contig, output_contig, data_contig, reduction,
                lengths_data, axis, initial, grad_input, segment_count,
                lengths_stride_axis);
        });
    }
    return grad_input;
}

Tensor segment_reduce_offsets_backward_cpu(
    const Tensor& grad_contig, const Tensor& output_contig,
    const Tensor& data_contig, SegmentReduction reduction,
    const Tensor& offsets_contig, int64_t axis,
    const std::optional<Scalar>& initial) {
    axis = offsets_contig.dim() - 1;
    const int64_t segment_count = offsets_contig.size(axis) - 1;
    const int64_t offsets_stride_axis = offsets_contig.stride(axis);
    std::vector<int64_t> grad_shape(static_cast<size_t>(data_contig.dim()));
    for (int64_t d = 0; d < data_contig.dim(); ++d) {
        grad_shape[d] = data_contig.size(d);
    }
    Tensor grad_input = Tensor::zeros(grad_shape, grad_contig.dtype(),
                                      grad_contig.device());

    if (offsets_contig.dtype() == DType::Int32) {
        const auto* offsets_data = offsets_contig.data_ptr<int32_t>();
        dispatch_segment_dtype(data_contig.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            segment_reduce_lengths_backward_kernel<
                scalar_t, /*is_offsets_like=*/true>(
                grad_contig, output_contig, data_contig, reduction,
                offsets_data, axis, initial, grad_input, segment_count,
                offsets_stride_axis);
        });
    } else {
        TP_CHECK(offsets_contig.dtype() == DType::Int64,
                 "segment_reduce: offsets must be int32 or int64");
        const auto* offsets_data = offsets_contig.data_ptr<int64_t>();
        dispatch_segment_dtype(data_contig.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            segment_reduce_lengths_backward_kernel<
                scalar_t, /*is_offsets_like=*/true>(
                grad_contig, output_contig, data_contig, reduction,
                offsets_data, axis, initial, grad_input, segment_count,
                offsets_stride_axis);
        });
    }
    return grad_input;
}

}  // namespace

Tensor segment_reduce_cpu(
    const Tensor& data, std::string reduce,
    std::optional<Tensor> lengths, std::optional<Tensor> indices,
    std::optional<Tensor> offsets, int64_t axis, bool unsafe,
    std::optional<Scalar> initial) {
    axis = wrap_axis(axis, data.dim());
    TP_CHECK(data.numel() >= 0, "segment_reduce: data must not be negative sized");

    const bool lengths_has_value = lengths.has_value();
    const bool offsets_has_value = offsets.has_value();
    TP_CHECK(!indices.has_value(),
             "segment_reduce(): indices based reduction is not supported yet.");
    TP_CHECK(lengths_has_value || offsets_has_value,
             "segment_reduce(): Either lengths or offsets must be defined.");

    const SegmentReduction reduction = get_segment_reduction(reduce);
    const Tensor data_contig = data.contiguous();

    if (offsets_has_value) {
        const Tensor& offsets_value = *offsets;
        TP_CHECK(data.device() == offsets_value.device(),
                 "segment_reduce: data and offsets must be on the same device");
        TP_CHECK(data.dim() >= offsets_value.dim(),
                 "segment_reduce: data must have at least as many dimensions "
                 "as offsets");
        TP_CHECK(axis == offsets_value.dim() - 1,
                 "segment_reduce(): Expected axis to be the last dimension of "
                 "offsets but got ", axis, ".");

        const Tensor offsets_contig = offsets_value.contiguous();
        return segment_reduce_offsets_cpu(
            reduction, data_contig, offsets_contig, axis, initial);
    }

    const Tensor& lengths_value = *lengths;
    TP_CHECK(data.device() == lengths_value.device(),
             "segment_reduce: data and lengths must be on the same device");
    TP_CHECK(data.dim() >= lengths_value.dim(),
             "segment_reduce: data must have at least as many dimensions as "
             "lengths");
    TP_CHECK(axis == lengths_value.dim() - 1,
             "segment_reduce(): Expected axis to be the last dimension of "
             "lengths but got ", axis, ".");

    if (!unsafe) {
        const Tensor min_length_t = lengths_value.min();
        const int64_t min_length = min_length_t.item().to<int64_t>();
        TP_CHECK(min_length >= 0, "lengths contains negative value!");
        // Every row of lengths (the last axis) must cover the reduction
        // axis of data exactly when the safety check is requested.
        const Tensor row_sums = lengths_value.sum({static_cast<int64_t>(
            lengths_value.dim() - 1)});
        const bool sums_match = row_sums.eq(Scalar(static_cast<double>(
            data.size(axis)))).all().item().to<bool>();
        TP_CHECK(sums_match,
                 "segment_reduce(): Expected all rows of lengths along axis ",
                 axis, " to sum to data.size(lengths.dim()-1) when !unsafe.");
    }

    const Tensor lengths_contig = lengths_value.contiguous();
    return segment_reduce_lengths_cpu(
        reduction, data_contig, lengths_contig, axis, initial);
}

// The forward and backward sweeps duplicate the segment-boundary walk; the
// forward result is not cached across the two.
Tensor _segment_reduce_backward_cpu(
    const Tensor& grad, const Tensor& output, const Tensor& data,
    std::string reduce, std::optional<Tensor> lengths,
    std::optional<Tensor> offsets, int64_t axis, std::optional<Scalar> initial) {
    axis = wrap_axis(axis, data.dim());
    const bool lengths_has_value =
        lengths.has_value() && lengths->defined();
    const bool offsets_has_value =
        offsets.has_value() && offsets->defined();
    TP_CHECK(lengths_has_value || offsets_has_value,
             "segment_reduce(): Either lengths or offsets must be defined.");

    const Tensor grad_contig = grad.contiguous();
    const Tensor output_contig = output.contiguous();
    const Tensor data_contig = data.contiguous();
    const SegmentReduction reduction = get_segment_reduction(reduce);

    if (offsets_has_value) {
        const Tensor offsets_contig = offsets->contiguous();
        return segment_reduce_offsets_backward_cpu(
            grad_contig, output_contig, data_contig, reduction,
            offsets_contig, axis, initial);
    }
    const Tensor lengths_contig = lengths->contiguous();
    return segment_reduce_lengths_backward_cpu(
        grad_contig, output_contig, data_contig, reduction, lengths_contig,
        axis, initial);
}

}  // namespace cpu
}  // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(CPU, SegmentReduce) {
    using namespace tensorplay::cpu;
    m.impl("segment_reduce", segment_reduce_cpu);
    m.impl("_segment_reduce_backward", _segment_reduce_backward_cpu);
}
