// Shape-only kernels for the assorted value-free ops on the meta device:
// alpha dropout, the histogram contract, the floating decomposition frexp,
// the segment-reduce and grid-sampler backwards, the sparse COO factories
// and the list-valued foreach copy/mm pairs.
//
// Alpha dropout samples a keep mask and rescales the input; only the
// geometries of the (output, mask) pair are observable.  The histogram bin
// count is the answer's whole extent, so the bin validations are scalar
// checks that run unchanged; deriving the span from the stored extrema when
// min equals max reads element data, so that path answers the shape and
// leaves the span validation to a materializing backend.

#include "MetaCommon.h"

#include "DType.h"
#include "Exception.h"
#include "Scalar.h"

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

std::tuple<Tensor, Tensor> native_alpha_dropout_kernel(const Tensor& input,
                                                       double p) {
    if (p < 0 || p >= 1) {
        TP_THROW(ValueError, "alpha_dropout: p must be in [0, 1)");
    }
    // The keep mask is sampled in single precision regardless of the input;
    // the affine rescale lands on the promoted type.
    return {empty_meta(shape_of(input),
                       promoteTypes(input.dtype(), DType::Float32), input.device()),
            empty_meta(shape_of(input), DType::Float32, input.device())};
}

Tensor histc_kernel(const Tensor& self, int64_t bins, const Scalar& min,
                    const Scalar& max) {
    if (bins <= 0) {
        TP_THROW(RuntimeError, "histc(): bins must be positive");
    }
    if (!isFloatingType(self.dtype())) {
        TP_THROW(NotImplementedError, "histc(): expected a floating-point tensor, got ",
                 toString(self.dtype()));
    }
    const double lo = min.toDouble();
    const double hi = max.toDouble();
    if (lo == hi) {
        // An equal range derives the span from the stored extrema, which a
        // metadata-only device cannot read; answer the shape and leave the
        // span validation to a materializing backend.
        return empty_meta({bins}, self.dtype(), self.device());
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        TP_THROW(RuntimeError, "histc: range of [", lo, ", ", hi, "] is not finite");
    }
    if (!(lo < hi)) {
        TP_THROW(RuntimeError, "histc: max must be larger than min");
    }
    return empty_meta({bins}, self.dtype(), self.device());
}

std::tuple<Tensor, Tensor> frexp_kernel(const Tensor& self) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(RuntimeError, "frexp(): only supports floating-point dtypes");
    }
    // The mantissa in [0.5, 1) (or exactly 0) carries the input dtype and the
    // exponent plane is 32-bit.
    return {empty_meta(shape_of(self), self.dtype(), self.device()),
            empty_meta(shape_of(self), DType::Int32, self.device())};
}

// ---------------------------------------------------------------------------
// Segment reduce backward.
//
// The gradient spreads back over the reduced segments: the answer repeats
// the data geometry and carries the gradient's dtype.  The reductions are
// named by a string, so the name gate runs unchanged; the per-segment
// arithmetic reads element data and stays with a materializing backend.
// ---------------------------------------------------------------------------

int64_t seg_reduce_wrap_axis(int64_t axis, int64_t ndim) {
    const int64_t dim_min = -ndim;
    const int64_t dim_max = ndim - 1;
    TP_CHECK(axis >= dim_min && axis <= dim_max,
             "Dimension out of range (expected to be in range of [", dim_min,
             ", ", dim_max, "], but got ", axis, ")");
    return axis < 0 ? axis + ndim : axis;
}

Tensor segment_reduce_backward_kernel(const Tensor& grad,
                                      const Tensor& /*output*/,
                                      const Tensor& data,
                                      const std::string& reduce,
                                      const std::optional<Tensor>& lengths,
                                      const std::optional<Tensor>& offsets,
                                      int64_t axis,
                                      const std::optional<Scalar>& /*initial*/) {
    seg_reduce_wrap_axis(axis, data.dim());
    const bool lengths_has_value = lengths.has_value() && lengths->defined();
    const bool offsets_has_value = offsets.has_value() && offsets->defined();
    TP_CHECK(lengths_has_value || offsets_has_value,
             "segment_reduce(): Either lengths or offsets must be defined.");
    if (reduce != "sum" && reduce != "prod" && reduce != "mean" &&
        reduce != "max" && reduce != "min" && reduce != "amax" &&
        reduce != "amin") {
        TP_THROW(RuntimeError,
                 "reduce argument must be either sum, prod, mean, amax or amin, got ",
                 reduce);
    }
    return empty_meta(shape_of(data), grad.dtype(), grad.device());
}

// ---------------------------------------------------------------------------
// Grid sampler 3D backward.
//
// Both gradients are always produced: the input gradient repeats the input
// geometry, the grid gradient appends the three coordinate components to the
// grid's sampling extents.  The requested-slot mask only decides whether the
// computing backend writes zeros or leaves the buffer uninitialized, which a
// meta tensor cannot observe.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> grid_sampler_3d_backward_kernel(
        const Tensor& grad_output, const Tensor& input, const Tensor& grid,
        int64_t interpolation_mode, int64_t /*padding_mode*/,
        bool /*align_corners*/, const std::vector<bool>& /*output_mask*/) {
    if (input.dim() != 5 || grid.dim() != 5) {
        TP_THROW(RuntimeError,
                 "grid_sampler_3d_backward: expected 5D input and grid");
    }
    constexpr int64_t kInterpBicubic = 2;
    if (interpolation_mode == kInterpBicubic) {
        TP_THROW(RuntimeError,
                 "grid_sampler_3d_backward: bicubic only supports 4D input");
    }
    switch (input.dtype()) {
        case DType::Float32:
        case DType::Float64:
        case DType::Float16:
        case DType::BFloat16:
            break;
        default:
            TP_THROW(TypeError, "grid_sampler_3d_backward: unsupported dtype");
    }
    const std::vector<int64_t> grid_shape = shape_of(grid);
    return {empty_meta(shape_of(input), input.dtype(), input.device()),
            empty_meta({grid_shape[0], grid_shape[1], grid_shape[2],
                        grid_shape[3], 3},
                       grid.dtype(), grid.device())};
}

// ---------------------------------------------------------------------------
// Sparse COO.
//
// Construction validates the index plane and the values' dense extents
// against the requested size, all of which is geometry; a meta tensor reuses
// the same shared constructor.  Inferring the size from the stored index
// maxima reads element data, so the meta device requires the size outright.
// ---------------------------------------------------------------------------

Tensor sparse_coo_tensor_kernel(const Tensor& indices, const Tensor& values,
                                const std::optional<std::vector<int64_t>>& size,
                                bool is_coalesced) {
    if (!size.has_value()) {
        TP_THROW(NotImplementedError,
                 "sparse_coo_tensor: an explicit size is required on the meta device");
    }
    return Tensor::make_sparse_coo_tensor(indices, values, *size, is_coalesced);
}

Tensor sparse_sum_kernel(const Tensor& input) {
    TP_CHECK(input.is_sparse() && !input.is_sparse_compressed(),
             "_sparse_sum(): expected a sparse COO tensor");
    // The reduction collapses every extent into a scalar in the values'
    // dtype; coalescing itself reads element data.
    return empty_meta({}, input.dtype(), input.device());
}

Tensor sparse_addmm_kernel(const Tensor& self, const Tensor& mat1,
                           const Tensor& mat2, const Scalar& /*beta*/,
                           const Scalar& /*alpha*/) {
    if (!mat1.is_sparse()) {
        TP_THROW(RuntimeError,
                 "_sparse_addmm(): expected 'mat1' to be a sparse tensor");
    }
    if (mat1.dim() != 2) {
        TP_THROW(RuntimeError, "_sparse_addmm(): matrices expected, got ",
                 mat1.dim(), "-D tensor");
    }
    const bool blocked_input = mat1.is_sparse_bsr() || mat1.is_sparse_bsc();
    if (mat1.sparse_dim() != 2 ||
        (mat1._values().dim() != 1 && !blocked_input)) {
        TP_THROW(RuntimeError,
                 "_sparse_addmm(): hybrid sparse tensors are not supported");
    }
    if (mat2.dim() != 2) {
        TP_THROW(RuntimeError, "_sparse_addmm(): matrices expected, got ",
                 mat2.dim(), "-D tensor");
    }
    if (mat1.device() != mat2.device() || mat1.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "_sparse_addmm(): all operands must be on the same device");
    }
    if (mat2.dtype() != mat1.dtype() || self.dtype() != mat1.dtype()) {
        TP_THROW(TypeError, "_sparse_addmm(): all operands must share one dtype");
    }
    const int64_t rows = mat1.size(0);
    const int64_t inner = mat1.size(1);
    const int64_t cols = mat2.size(1);
    if (mat2.size(0) != inner) {
        TP_THROW(RuntimeError,
                 "_sparse_addmm(): mat1 and mat2 shapes cannot be multiplied (",
                 rows, "x", inner, " and ", mat2.size(0), "x", cols, ")");
    }
    return empty_meta({rows, cols}, mat1.dtype(), mat1.device());
}

// ---------------------------------------------------------------------------
// Foreach copy and mm.
//
// Both walk the operand lists pairwise and answer one result per pair: the
// copy is a clone of the source (so the result carries the source's shape,
// dtype and device), and mm is the plain matrix product of the pair.  The
// per-pair kernels are dispatched, so their own shape gates apply unchanged.
// ---------------------------------------------------------------------------

std::vector<Tensor> foreach_copy_kernel(const std::vector<Tensor>& self,
                                        const std::vector<Tensor>& src,
                                        bool /*non_blocking*/) {
    if (self.size() != src.size()) {
        TP_THROW(ValueError,
                 "foreach tensor list arguments must have the same length");
    }
    std::vector<Tensor> out;
    out.reserve(src.size());
    for (const Tensor& source : src) {
        out.push_back(source.clone());
    }
    return out;
}

std::vector<Tensor> foreach_mm_kernel(const std::vector<Tensor>& self,
                                      const std::vector<Tensor>& mat2) {
    if (self.size() != mat2.size()) {
        TP_THROW(ValueError,
                 "foreach tensor list arguments must have the same length");
    }
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
        out.push_back(self[i].mm(mat2[i]));
    }
    return out;
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaMisc) {
    m.impl("native_alpha_dropout", native_alpha_dropout_kernel);
    m.impl("histc", histc_kernel);
    m.impl("frexp", frexp_kernel);
    m.impl("_segment_reduce_backward", segment_reduce_backward_kernel);
    m.impl("grid_sampler_3d_backward", grid_sampler_3d_backward_kernel);
    m.impl("sparse_coo_tensor", sparse_coo_tensor_kernel);
    m.impl("_sparse_addmm", sparse_addmm_kernel);
    m.impl("_foreach_copy", foreach_copy_kernel);
    m.impl("_foreach_mm", foreach_mm_kernel);
}

// A sparse-layout tensor dispatches under the device-independent Sparse key
// rather than its device's dense key, so the single-operand reduction lives
// there alone: a dense tensor passed here reaches no kernel at all, which is
// the same outcome a dense operand gets on a computing backend.
TENSORPLAY_LIBRARY_IMPL(Sparse, MetaSparseMisc) {
    m.impl("_sparse_sum", sparse_sum_kernel);
}

} // namespace meta
} // namespace tensorplay
