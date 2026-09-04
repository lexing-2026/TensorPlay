#pragma once

// Shared validation and operand normalization for the searchsorted /
// bucketize family.  Both the CPU and CUDA kernels consume the same contract:
//
// - `boundaries` (the sorted sequence) is 1-D, or N-D matching every leading
//   dimension of the query input (lookup tables share the innermost axis);
// - `side` ("left"/"right") and the `right` flag select the same bound
//   direction and must not contradict each other;
// - `sorter` carries the permutation that orders an unsorted boundary tensor;
// - integer output is Int32 with `out_int32` and Int64 otherwise.

#include "DType.h"
#include "Exception.h"
#include "Scalar.h"
#include "Tensor.h"
#include "TypePromotion.h"

#include <limits>
#include <optional>
#include <string>
#include <tuple>

namespace tensorplay {
namespace bucketization {

// True when `boundaries` matches every dimension of `input` except possibly
// the innermost one (the shape contract for row-wise lookup tables).
inline bool dims_matched_before_last_dim(const Tensor& boundaries,
                                         const Tensor& input) {
    if (boundaries.dim() != input.dim()) return false;
    for (int64_t d = 0; d + 1 < boundaries.dim(); ++d) {
        if (boundaries.size(d) != input.size(d)) return false;
    }
    return true;
}

// Validation shared by every overload of searchsorted and bucketize.
inline void pre_check(const Tensor& boundaries, const Tensor& input,
                      const Tensor& output, bool out_int32, bool right,
                      const std::optional<std::string>& side_opt,
                      const Tensor& sorter) {
    if (side_opt.has_value()) {
        const std::string& side = *side_opt;
        TP_CHECK(side == "left" || side == "right",
                 "searchsorted(): side can only be 'left' or 'right' but got ",
                 side);
        TP_CHECK(!right || side == "right",
                 "searchsorted(): side and right can't be set to opposites, got "
                 "side of ", side, " while right was True");
    }
    TP_CHECK(boundaries.device() == input.device(),
             "searchsorted(): boundaries and input value tensors should have "
             "the same device, but got boundaries device ",
             boundaries.device().toString(), " and input device ",
             input.device().toString());
    if (sorter.defined()) {
        TP_CHECK(sorter.device() == boundaries.device(),
                 "searchsorted(): sorter and boundary tensors should have the "
                 "same device, but got sorter device ",
                 sorter.device().toString(), " and boundaries device ",
                 boundaries.device().toString());
        TP_CHECK(sorter.shape() == boundaries.shape(),
                 "searchsorted(): boundary and sorter must have the same size, "
                 "but got boundary shape ", boundaries.shape(),
                 " and sorter shape ", sorter.shape());
        TP_CHECK(sorter.dtype() == DType::Int64,
                 "searchsorted(): sorter must be a tensor of long dtype but "
                 "got dtype ", toString(sorter.dtype()));
        if (sorter.numel() > 0) {
            auto sorter_minmax = Tensor::aminmax(sorter, {}, false);
            const int64_t vmin = std::get<0>(sorter_minmax).item<int64_t>();
            const int64_t vmax = std::get<1>(sorter_minmax).item<int64_t>();
            TP_CHECK(vmin >= 0 && vmax < sorter.size(-1),
                     "searchsorted(): sorter index out of range");
        }
    }
    TP_CHECK(input.dim() > 0 ||
                 (input.dim() == 0 && input.numel() == 1 && boundaries.dim() == 1),
             "searchsorted(): input value can be a scalar only when boundaries "
             "tensor dimension is 1, but we got boundaries dim(",
             boundaries.dim(), ") and input value dim(", input.dim(),
             ") numel(", input.numel(), ")");
    TP_CHECK(boundaries.dim() != 0,
             "searchsorted(): boundaries tensor should have positive "
             "dimension, but got 0 dimension");
    TP_CHECK(boundaries.dim() == 1 || dims_matched_before_last_dim(boundaries, input),
             "searchsorted(): boundaries tensor should be 1 dimension or the "
             "first N-1 dimensions of boundaries tensor and input value tensor "
             "must match, but we got boundaries shape ", boundaries.shape(),
             " and input value shape ", input.shape());
    const DType out_dtype = output.dtype();
    TP_CHECK((out_dtype == DType::Int64 && !out_int32) ||
                 (out_dtype == DType::Int32 && out_int32),
             "searchsorted(): output tensor's dtype is wrong, it can only be "
             "Int32 or Int64 depending on whether the out_int32 flag is True, "
             "but we got output dtype ", toString(out_dtype),
             " and out_int32 flag is ", out_int32 ? "True" : "False");
    if (out_int32) {
        TP_CHECK(boundaries.size(-1) < std::numeric_limits<int32_t>::max(),
                 "searchsorted(): the size of boundaries' last dimension should "
                 "be less than INT_MAX, but we got ", boundaries.size(-1));
    }
}

// Contiguous copies for non-contiguous operands, then dtype unification via
// the common promotion type when input and boundaries disagree.
inline void maybe_trim_input_tensors(Tensor& trimmed_input,
                                     Tensor& trimmed_boundaries,
                                     const Tensor& raw_input,
                                     const Tensor& raw_boundaries) {
    if (!raw_input.is_contiguous()) trimmed_input = raw_input.contiguous();
    if (!raw_boundaries.is_contiguous()) {
        trimmed_boundaries = raw_boundaries.contiguous();
    }
    if (raw_input.dtype() != raw_boundaries.dtype()) {
        const DType common = promoteTypes(raw_input.dtype(),
                                          raw_boundaries.dtype());
        TP_CHECK(common != DType::Undefined,
                 "searchsorted(): input and boundaries have incompatible "
                 "dtypes ", toString(raw_input.dtype()), " and ",
                 toString(raw_boundaries.dtype()));
        if (common != raw_input.dtype()) {
            trimmed_input = trimmed_input.defined()
                ? trimmed_input.to(common) : raw_input.to(common);
        }
        if (common != raw_boundaries.dtype()) {
            trimmed_boundaries = trimmed_boundaries.defined()
                ? trimmed_boundaries.to(common) : raw_boundaries.to(common);
        }
    }
}

// 0-dim tensor holding the scalar query on the boundaries' device, typed by
// the scalar itself; the shared dtype-unification pass reconciles it with the
// boundaries before the search runs.  Built through the `full` dispatch so a
// device-resident result is filled on that device.
inline Tensor scalar_tensor(const Scalar& scalar, const Device& device) {
    return Tensor::full({}, scalar, scalar.dtype(), device);
}

} // namespace bucketization
} // namespace tensorplay
