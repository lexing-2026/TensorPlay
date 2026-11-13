// Shape-only kernels for normalization on the meta device.
//
// Layer norm normalizes each row over the trailing `normalized_shape`
// extents; the output repeats the input's geometry while the per-row mean
// and reciprocal standard deviation come back as one value per row.  Batch
// norm normalizes per channel; in training it reports per-channel running
// statistics, and in evaluation it reports none.  Statistics of a
// reduced-precision input are accumulated and reported in single precision.

#include "MetaCommon.h"

#include "DType.h"
#include "TypeProperties.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

// Statistics of a reduced-precision input are accumulated in single
// precision and come back as such; every other input keeps its own type.
DType norm_stat_dtype(DType input_dtype) {
    return (input_dtype == DType::Float16 || input_dtype == DType::BFloat16)
               ? DType::Float32
               : input_dtype;
}

// ---------------------------------------------------------------------------
// Layer norm.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_kernel(
    const Tensor& input, const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& bias,
    double /*eps*/) {
    if (normalized_shape.empty()) {
        TP_THROW(RuntimeError, "native_layer_norm: normalized_shape must not be empty");
    }
    const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
    const int64_t input_ndim = input.dim();
    if (norm_ndim > input_ndim) {
        TP_THROW(RuntimeError, "layer_norm: normalized_shape dim larger than input dim");
    }
    const int64_t outer_dims = input_ndim - norm_ndim;
    int64_t inner_size = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(outer_dims + i) != normalized_shape[static_cast<size_t>(i)]) {
            TP_THROW(RuntimeError, "layer_norm: Input shape mismatch with normalized_shape");
        }
        inner_size *= normalized_shape[static_cast<size_t>(i)];
    }
    for (const auto* affine : {&weight, &bias}) {
        if (affine->has_value() && (*affine)->defined() &&
            shape_of(**affine) != normalized_shape) {
            TP_THROW(RuntimeError,
                     affine == &weight
                         ? "layer_norm: weight shape mismatch with normalized_shape"
                         : "layer_norm: bias shape mismatch with normalized_shape");
        }
    }
    const int64_t outer_size = inner_size == 0 ? 0 : input.numel() / inner_size;
    const DType stats_dtype = norm_stat_dtype(input.dtype());
    return {empty_meta(shape_of(input), input.dtype(), input.device()),
            empty_meta({outer_size}, stats_dtype, input.device()),
            empty_meta({outer_size}, stats_dtype, input.device())};
}

// The public spelling returns only the normalized output; the per-row
// statistics are requested through the native op separately.
Tensor layer_norm_kernel(const Tensor& input,
                         const std::vector<int64_t>& normalized_shape,
                         const std::optional<Tensor>& weight,
                         const std::optional<Tensor>& bias, double eps) {
    return std::get<0>(
        native_layer_norm_kernel(input, normalized_shape, weight, bias, eps));
}

// ---------------------------------------------------------------------------
// Batch norm.
// ---------------------------------------------------------------------------

// One value per channel, in the statistics dtype, when a parameter or a
// running statistic is supplied.
void check_bn_parameter(const std::optional<Tensor>& parameter,
                        int64_t channels, const char* name) {
    if (!parameter.has_value() || !parameter->defined()) return;
    if (parameter->dim() != 1 || parameter->numel() != channels) {
        TP_THROW(RuntimeError, std::string("batch_norm: ") + name +
                                  " must have one value per channel");
    }
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_kernel(
    const Tensor& input, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var, bool training, double /*momentum*/,
    double /*eps*/) {
    if (input.dim() < 2 || input.dim() > 5) {
        TP_THROW(RuntimeError, "batch_norm: Input must be between 2D and 5D");
    }
    if (!isFloatingType(input.dtype())) {
        TP_THROW(NotImplementedError,
                 std::string("batch_norm: only floating point dtypes are supported (got ") +
                     toString(input.dtype()) + ")");
    }
    const int64_t channels = input.size(1);
    check_bn_parameter(weight, channels, "weight");
    check_bn_parameter(bias, channels, "bias");
    check_bn_parameter(running_mean, channels, "running_mean");
    check_bn_parameter(running_var, channels, "running_var");
    const bool has_mean =
        running_mean.has_value() && running_mean->defined();
    const bool has_var = running_var.has_value() && running_var->defined();
    if (has_mean != has_var) {
        TP_THROW(RuntimeError,
                 "batch_norm: running_mean and running_var must either both be "
                 "defined or both be absent");
    }
    if (!training && !(has_mean && has_var)) {
        TP_THROW(RuntimeError,
                 "batch_norm: running_mean and running_var must be defined in "
                 "evaluation mode");
    }
    const DType stats_dtype = norm_stat_dtype(input.dtype());
    const std::vector<int64_t> stats_shape =
        training ? std::vector<int64_t>{channels} : std::vector<int64_t>{0};
    return {empty_meta(shape_of(input), input.dtype(), input.device()),
            empty_meta(stats_shape, stats_dtype, input.device()),
            empty_meta(stats_shape, stats_dtype, input.device())};
}

// The public `batch_norm` spelling returns only the normalized output; the
// autograd layer obtains the statistics through the native op separately.
Tensor batch_norm_kernel(const Tensor& input,
                         const std::optional<Tensor>& weight,
                         const std::optional<Tensor>& bias,
                         const std::optional<Tensor>& running_mean,
                         const std::optional<Tensor>& running_var, bool training,
                         double momentum, double eps) {
    return std::get<0>(native_batch_norm_kernel(
        input, weight, bias, running_mean, running_var, training, momentum, eps));
}

// Root-mean-square normalization scales each trailing block by the root mean
// square of its entries, so the answer keeps the input's geometry.  The
// normalized shape must match the input's trailing extents.
Tensor rms_norm_kernel(const Tensor& input,
                       const std::vector<int64_t>& normalized_shape,
                       const std::optional<Tensor>& weight,
                       std::optional<double> /*eps*/) {
    const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
    const int64_t input_ndim = input.dim();
    if (norm_ndim > input_ndim) {
        TP_THROW(RuntimeError,
                 "rms_norm: normalized_shape dim larger than input dim");
    }
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(input_ndim - norm_ndim + i) != normalized_shape[i]) {
            TP_THROW(RuntimeError,
                     "rms_norm: Input shape mismatch with normalized_shape");
        }
    }
    if (!isFloatingType(input.dtype())) {
        TP_THROW(NotImplementedError, "rms_norm: unsupported dtype");
    }
    (void)weight;
    return empty_meta(shape_of(input), input.dtype(), input.device());
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Normalization. Group norm normalizes per (sample, group) over the channels
// and the spatial extents together, so its result has the input's geometry
// and its per-group statistics hold one value per group.
// ---------------------------------------------------------------------------
// The statistics of a reduced-precision input are accumulated in single
// precision and come back as such; every other input keeps its own type.
DType group_norm_stat_dtype(const Tensor& input) {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        return DType::Float32;
    }
    return input.dtype();
}

Tensor group_norm_kernel(const Tensor& input, int64_t num_groups,
                         const std::optional<Tensor>& weight,
                         const std::optional<Tensor>& bias, double /*eps*/) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "group_norm requires at least 2 dims");
    }
    const int64_t channels = input.size(1);
    if (channels <= 0 || num_groups <= 0 || channels % num_groups != 0) {
        TP_THROW(RuntimeError,
                 "group_norm: num_channels must be divisible by num_groups");
    }
    (void)weight;
    (void)bias;
    return empty_meta(shape_of(input), input.dtype(), input.device());
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_kernel(
    const Tensor& input, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias, int64_t N, int64_t C, int64_t HxW,
    int64_t group, double /*eps*/) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "native_group_norm requires at least 2 dims");
    }
    int64_t spatial = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        spatial *= input.size(dim);
    }
    if (N != input.size(0) || C != input.size(1) || HxW != spatial) {
        TP_THROW(RuntimeError, "native_group_norm dimensions do not match input");
    }
    if (C <= 0 || spatial <= 0 || group <= 0 || C % group != 0) {
        TP_THROW(RuntimeError, "native_group_norm has invalid group dimensions");
    }
    // One scale and one shift per channel, when either is given.
    const auto check_parameter = [C](const std::optional<Tensor>& parameter,
                                     const char* name) {
        if (!parameter.has_value() || !parameter->defined()) return;
        if (parameter->dim() != 1 || parameter->numel() != C) {
            TP_THROW(RuntimeError, std::string("native_group_norm ") + name +
                                     " must have one value per channel");
        }
    };
    check_parameter(weight, "weight");
    check_parameter(bias, "bias");
    const DType stat = group_norm_stat_dtype(input);
    return {empty_meta(shape_of(input), input.dtype(), input.device()),
            empty_meta({N, group}, stat, input.device()),
            empty_meta({N, group}, stat, input.device())};
}

// Instance norm normalizes each (sample, channel) plane, so the result keeps
// the input geometry. It requires a spatial axis, a floating input and, when
// the input statistics are not used, both running statistics.
Tensor instance_norm_kernel(const Tensor& input,
                            const std::optional<Tensor>& /*weight*/,
                            const std::optional<Tensor>& /*bias*/,
                            const std::optional<Tensor>& running_mean,
                            const std::optional<Tensor>& running_var,
                            bool use_input_stats, double /*momentum*/,
                            double /*eps*/) {
    if (input.dim() < 3) {
        TP_THROW(RuntimeError,
                 "instance_norm: input must have at least 3 dimensions");
    }
    if (!isFloatingType(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "instance_norm: only floating point dtypes are supported "
                 "(got ", toString(input.dtype()), ")");
    }
    if (!use_input_stats &&
        !(running_mean.has_value() && running_mean->defined() &&
          running_var.has_value() && running_var->defined())) {
        TP_THROW(RuntimeError, "instance_norm: running_mean and running_var "
                               "must be defined when use_input_stats is false");
    }
    return empty_meta(shape_of(input), input.dtype(), input.device());
}
} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaNormalization) {
    m.impl("native_layer_norm", native_layer_norm_kernel);
    m.impl("layer_norm", layer_norm_kernel);
    m.impl("native_batch_norm", native_batch_norm_kernel);
    m.impl("batch_norm", batch_norm_kernel);
    m.impl("rms_norm", rms_norm_kernel);
    m.impl("group_norm", group_norm_kernel);
    m.impl("native_group_norm", native_group_norm_kernel);
    m.impl("instance_norm", instance_norm_kernel);
}

} // namespace meta
} // namespace tensorplay
