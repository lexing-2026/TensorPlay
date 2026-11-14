// Shape-only kernels for reductions that report a moment or a position on the
// meta device.
//
// A moment (variance, standard deviation) reduces the operand's geometry just
// like a sum, but answers in a value type: a complex input produces a real
// moment and a whole-number input answers at single precision.  An extremum
// that names a dimension reports the winning positions in an Int64 tensor of
// the values' shape; the reduction without a dimension answers a scalar.
// Nothing reads element data: the winners are values, not metadata.

#include "MetaCommon.h"

#include "DType.h"
#include "DTypeNames.h"
#include "TypeProperties.h"

#include <cstdint>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

void require_scalar_elements(const Tensor& tensor, const char* op) {
    if (!is_scalar_dtype(tensor.dtype())) {
        TP_THROW(TypeError, op, ": unsupported dtype");
    }
}

// The value type of a moment: real-valued even for complex inputs, and at
// least single precision for whole-number inputs.
DType moment_dtype(const Tensor& self) {
    if (isComplexType(self.dtype())) return toRealValueType(self.dtype());
    return isFloatingType(self.dtype()) ? self.dtype() : DType::Float32;
}

// Defined with the mean kernels below; the paired moment forms name it.
DType mean_like_dtype(const Tensor& self, DType dtype);

// ---------------------------------------------------------------------------
// Moments.
// ---------------------------------------------------------------------------

Tensor var_kernel(const Tensor& self, int64_t /*correction*/) {
    return empty_meta({}, moment_dtype(self), self.device());
}

Tensor var_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                      int64_t /*correction*/, bool keepdim) {
    return empty_meta(reduce_out_shape(self, dim, keepdim),
                      moment_dtype(self), self.device());
}

Tensor std_kernel(const Tensor& self, int64_t /*correction*/) {
    return empty_meta({}, moment_dtype(self), self.device());
}

Tensor std_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                      int64_t /*correction*/, bool keepdim) {
    return empty_meta(reduce_out_shape(self, dim, keepdim),
                      moment_dtype(self), self.device());
}

// The paired form reports the moment and the mean, which share their shape;
// the deviation answers in the real component type, the mean keeps the
// operand's own plane.
std::tuple<Tensor, Tensor> std_mean_kernel(const Tensor& self,
                                           const std::vector<int64_t>& dim,
                                           bool /*unbiased*/, bool keepdim) {
    const std::vector<int64_t> out = reduce_out_shape(self, dim, keepdim);
    return {empty_meta(out, moment_dtype(self), self.device()),
            empty_meta(out, mean_like_dtype(self, DType::Undefined),
                       self.device())};
}

// ---------------------------------------------------------------------------
// Extrema.  The extrema without a dimension answer a scalar; with a dimension
// they answer the winning values and their positions along it.  No complex
// form exists: ordering a complex plane has no kernel.
// ---------------------------------------------------------------------------

Tensor extrema_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "extrema: not implemented for complex dtypes");
    }
    return empty_meta({}, self.dtype(), self.device());
}

std::tuple<Tensor, Tensor> extrema_dim_kernel(const Tensor& self, int64_t dim,
                                              bool keepdim) {
    dim = wrap_dim(dim, self.dim());
    if (isComplexType(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "extrema: not implemented for complex dtypes");
    }
    std::vector<int64_t> out = shape_of(self);
    if (keepdim) {
        out[static_cast<size_t>(dim)] = 1;
    } else {
        out.erase(out.begin() + dim);
    }
    return {empty_meta(out, self.dtype(), self.device()),
            empty_meta(out, DType::Int64, self.device())};
}

// ---------------------------------------------------------------------------
// Predicates: all and any answer one boolean per reduced position, and the
// whole-tensor forms answer a 0-dim boolean.
// ---------------------------------------------------------------------------

Tensor all_kernel(const Tensor& self) {
    return empty_meta({}, DType::Bool, self.device());
}

Tensor all_dim_kernel(const Tensor& self, int64_t dim, bool keepdim) {
    return empty_meta(reduce_out_shape(self, {dim}, keepdim), DType::Bool,
                      self.device());
}

Tensor any_kernel(const Tensor& self) {
    return empty_meta({}, DType::Bool, self.device());
}

Tensor any_dim_kernel(const Tensor& self, int64_t dim, bool keepdim) {
    return empty_meta(reduce_out_shape(self, {dim}, keepdim), DType::Bool,
                      self.device());
}

// ---------------------------------------------------------------------------
// logsumexp: the extents of a sum over the named axes, in the operand's value
// type; a whole-number input answers at the default floating width.
// ---------------------------------------------------------------------------

Tensor logsumexp_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                        bool keepdim) {
    if (!isFloatingType(self.dtype()) && !isIntegralType(self.dtype(), true)) {
        TP_THROW(RuntimeError, "logsumexp(): Expected floating point type, got ",
                 scalarTypeName(self.dtype()));
    }
    const int64_t nd = self.dim();
    std::vector<bool> seen(static_cast<size_t>(nd > 0 ? nd : 1), false);
    for (const int64_t d : dim) {
        const int64_t wrapped = wrap_dim(d, nd > 0 ? nd : 1);
        if (seen[static_cast<size_t>(wrapped)]) {
            TP_THROW(RuntimeError, "logsumexp(): dim ", wrapped,
                     " appears multiple times in the list of dims");
        }
        seen[static_cast<size_t>(wrapped)] = true;
    }
    const DType out_dtype =
        isIntegralType(self.dtype(), true) ? globalContext().defaultDType()
                                           : self.dtype();
    return empty_meta(reduce_out_shape(self, dim, keepdim), out_dtype,
                      self.device());
}

// ---------------------------------------------------------------------------
// Positional summaries: mode reports the most frequent value and its first
// position, kthvalue the k-th smallest value and its position.  Each answers
// a pair of same-shaped tensors that drop the reduced axis when keepdim is
// not asked for; the positions are Int64.
// ---------------------------------------------------------------------------

std::vector<int64_t> positional_out_shape(const Tensor& self, int64_t dim,
                                          bool keepdim) {
    std::vector<int64_t> out = shape_of(self);
    if (keepdim) {
        out[static_cast<size_t>(dim)] = 1;
    } else {
        out.erase(out.begin() + dim);
    }
    return out;
}

std::tuple<Tensor, Tensor> mode_kernel(const Tensor& self, int64_t dim,
                                       bool keepdim) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar mode input: ", dim);
        }
        require_scalar_elements(self, "mode");
        return {empty_meta({}, self.dtype(), self.device()),
                empty_meta({}, DType::Int64, self.device())};
    }
    dim = wrap_dim(dim, nd);
    if (self.size(dim) == 0) {
        TP_THROW(RuntimeError,
                 "mode(): Expected reduction dim ", dim,
                 " to have non-zero size");
    }
    require_scalar_elements(self, "mode");
    const std::vector<int64_t> out = positional_out_shape(self, dim, keepdim);
    return {empty_meta(out, self.dtype(), self.device()),
            empty_meta(out, DType::Int64, self.device())};
}

std::tuple<Tensor, Tensor> kthvalue_kernel(const Tensor& self, int64_t k,
                                           int64_t dim, bool keepdim) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar kthvalue input: ", dim);
        }
        if (k != 1) {
            TP_THROW(RuntimeError,
                     "kthvalue(): selected number k out of range for dim 0");
        }
        require_scalar_elements(self, "kthvalue");
        return {empty_meta({}, self.dtype(), self.device()),
                empty_meta({}, DType::Int64, self.device())};
    }
    dim = wrap_dim(dim, nd);
    if (k < 1 || k > self.size(dim)) {
        TP_THROW(RuntimeError,
                 "kthvalue(): selected number k out of range for dim ", dim);
    }
    require_scalar_elements(self, "kthvalue");
    const std::vector<int64_t> out = positional_out_shape(self, dim, keepdim);
    return {empty_meta(out, self.dtype(), self.device()),
            empty_meta(out, DType::Int64, self.device())};
}

// ---------------------------------------------------------------------------
// Order statistics: the median over the whole operand answers a scalar, the
// nan-aware form treats a whole-number operand as Int64, and the running
// extrema report a position per element beside the values.
// ---------------------------------------------------------------------------

Tensor median_flat_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "median: not implemented for complex dtypes");
    }
    require_scalar_elements(self, "median");
    return empty_meta({}, self.dtype(), self.device());
}

// The flattened form reads every element into one list: a floating operand
// keeps its width, every other operand is read through the whole-number
// reading type.
Tensor nanmedian_flat_kernel(const Tensor& self) {
    const DType out = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    return empty_meta({}, out, self.device());
}

std::tuple<Tensor, Tensor> nanmedian_dim_kernel(const Tensor& self, int64_t dim,
                                                bool keepdim) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        TP_THROW(RuntimeError,
                 "nanmedian(): expects a tensor with at least one dimension");
    }
    dim = wrap_dim(dim, nd);
    if (!isFloatingType(self.dtype())) {
        TP_THROW(RuntimeError,
                 "nanmedian(): only floating point dtypes are supported");
    }
    // The sliced form reads elements directly, so only the two widest
    // floating types have a reading loop.
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(TypeError, "nanmedian: unsupported dtype ",
                 toString(self.dtype()));
    }
    if (self.size(dim) == 0) {
        TP_THROW(RuntimeError, "nanmedian(): Expected reduction dim ", dim,
                 " to have non-zero size");
    }
    const std::vector<int64_t> out = positional_out_shape(self, dim, keepdim);
    return {empty_meta(out, self.dtype(), self.device()),
            empty_meta(out, DType::Int64, self.device())};
}

int64_t wrap_running_dim(int64_t dim, int64_t ndim, const char* op) {
    if (ndim == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError, "Dimension out of range for a scalar ", op,
                     " input: ", dim);
        }
        return 0;
    }
    return wrap_dim(dim, ndim);
}

std::tuple<Tensor, Tensor> cummax_kernel(const Tensor& self, int64_t dim) {
    wrap_running_dim(dim, self.dim(), "cummax");
    require_scalar_elements(self, "cummax");
    const std::vector<int64_t> out = shape_of(self);
    return {empty_meta(out, self.dtype(), self.device()),
            empty_meta(out, DType::Int64, self.device())};
}

std::tuple<Tensor, Tensor> cummin_kernel(const Tensor& self, int64_t dim) {
    wrap_running_dim(dim, self.dim(), "cummin");
    require_scalar_elements(self, "cummin");
    const std::vector<int64_t> out = shape_of(self);
    return {empty_meta(out, self.dtype(), self.device()),
            empty_meta(out, DType::Int64, self.device())};
}

// ---------------------------------------------------------------------------
// NaN-aware and accumulate-style reductions.
// ---------------------------------------------------------------------------

Tensor nansum_kernel(const Tensor& self, const std::vector<int64_t>& dim_in,
                     bool keepdim) {
    // A NaN passes through the accumulator untouched, which needs a
    // floating or integer lane; complex accumulation has no NaN lane here.
    if (isComplexType(self.dtype())) {
        TP_THROW(RuntimeError, "nansum does not support complex inputs");
    }
    const DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    std::vector<int64_t> dims = dim_in;
    if (dims.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i);
    }
    return empty_meta(reduce_out_shape(self, dims, keepdim), out_dt, self.device());
}

Tensor nanmean_kernel(const Tensor& self, std::optional<int64_t> dim_opt,
                      bool keepdim, std::optional<DType> dtype) {
    const DType acc_dt = dtype.value_or(DType::Undefined);
    if (!isFloatingType(self.dtype()) && !isComplexType(self.dtype())) {
        TP_THROW(TypeError,
                 "nanmean(): expected input to have floating point or complex dtype but got ",
                 toString(self.dtype()));
    }
    if (acc_dt != DType::Undefined && !isFloatingType(acc_dt) &&
        !isComplexType(acc_dt)) {
        TP_THROW(TypeError,
                 "nanmean(): could not infer output dtype. Optional dtype must be either a floating point or complex dtype. Got: ",
                 toString(acc_dt));
    }
    // The mean sums the input with the NaN-aware accumulator, so a complex
    // operand works only after the caller pins a real accumulation type.
    if (isComplexType(self.dtype()) &&
        (acc_dt == DType::Undefined || isComplexType(acc_dt))) {
        TP_THROW(RuntimeError, "nansum does not support complex inputs");
    }
    std::vector<int64_t> dims;
    if (dim_opt.has_value()) {
        dims.push_back(*dim_opt);
    } else {
        for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i);
    }
    // Narrow floating types accumulate in the single-precision lane.
    const DType out_dt = acc_dt != DType::Undefined
                             ? acc_dt
                             : (isReducedFloatingType(self.dtype()) ? DType::Float32
                                                                    : self.dtype());
    return empty_meta(reduce_out_shape(self, dims, keepdim), out_dt, self.device());
}

Tensor logcumsumexp_kernel(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        // A rank-0 tensor has one virtual dimension, addressable as -1 or 0.
        if (dim != -1 && dim != 0) {
            TP_THROW(IndexError,
                     "Dimension out of range for a scalar tensor (expected -1 or 0, but got ",
                     dim, ")");
        }
    } else if (dim < -nd || dim >= nd) {
        TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                 -nd, ", ", nd - 1, "], but got ", dim, ")");
    }
    const DType out_dt = dtype.value_or(self.dtype());
    // The scan accumulates in one of the four floating lanes; complex inputs
    // accumulate in their own lane.
    if (!isComplexType(out_dt) && out_dt != DType::Float32 &&
        out_dt != DType::Float64 && out_dt != DType::Float16 &&
        out_dt != DType::BFloat16) {
        TP_THROW(TypeError, "logcumsumexp: unsupported dtype");
    }
    return empty_meta(shape_of(self), out_dt, self.device());
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Reductions.
// ---------------------------------------------------------------------------
std::vector<int64_t> reduce_out_shape(const Tensor& self,
                                      const std::vector<int64_t>& dims,
                                      bool keepdim) {
    const int64_t nd = self.dim();
    std::vector<bool> reduced(static_cast<size_t>(nd), false);
    // An empty list names no dimension, and a reduction that names no
    // dimension is a reduction over all of them; a rank-0 operand has its one
    // virtual dimension reduced.
    if (dims.empty()) {
        if (nd == 0) {
            reduced.push_back(true);
        } else {
            for (int64_t i = 0; i < nd; ++i) {
                reduced[static_cast<size_t>(i)] = true;
            }
        }
    }
    for (int64_t d : dims) {
        int64_t dim = d;
        if (nd == 0) {
            // A 0-d tensor accepts -1 and 0 as its single (virtual) dimension.
            if (dim != 0 && dim != -1) {
                TP_THROW(IndexError, "reduction dimension ", dim,
                         " out of range for a rank-", nd, " tensor");
            }
            reduced.push_back(true);
            continue;
        }
        if (dim < 0) dim += nd;
        if (dim < 0 || dim >= nd) {
            TP_THROW(IndexError, "reduction dimension ", dim,
                     " out of range for a rank-", nd, " tensor");
        }
        reduced[static_cast<size_t>(dim)] = true;
    }
    std::vector<int64_t> out;
    out.reserve(reduced.size());
    for (size_t i = 0; i < reduced.size(); ++i) {
        if (!reduced[i]) {
            out.push_back(nd == 0 ? 0 : self.size(static_cast<int64_t>(i)));
        } else if (keepdim) {
            out.push_back(1);
        }
    }
    return out;
}

// Reductions widen narrow integers and booleans to Int64 unless a dtype is
// requested explicitly.
DType sum_like_dtype(const Tensor& self, DType dtype) {
    if (dtype != DType::Undefined) return dtype;
    if (isIntegralType(self.dtype(), /*include_bool=*/true)) return DType::Int64;
    return self.dtype();
}

// Means keep the input dtype for float and complex inputs and fall back to
// float for everything else.
DType mean_like_dtype(const Tensor& self, DType dtype) {
    if (dtype != DType::Undefined) return dtype;
    return isFloatingOrComplexType(self.dtype()) ? self.dtype() : DType::Float32;
}

Tensor sum_kernel(const Tensor& self, DType dtype) {
    return empty_meta({}, sum_like_dtype(self, dtype), self.device());
}

Tensor sum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                      bool keepdim, DType dtype) {
    return empty_meta(reduce_out_shape(self, dims, keepdim),
                      sum_like_dtype(self, dtype), self.device());
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
    return empty_meta({}, mean_like_dtype(self, dtype), self.device());
}

Tensor mean_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                       bool keepdim, DType dtype) {
    return empty_meta(reduce_out_shape(self, dims, keepdim),
                      mean_like_dtype(self, dtype), self.device());
}

Tensor prod_kernel(const Tensor& self, DType dtype) {
    return empty_meta({}, sum_like_dtype(self, dtype), self.device());
}

Tensor prod_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                       bool keepdim, DType dtype) {
    return empty_meta(reduce_out_shape(self, dims, keepdim),
                      sum_like_dtype(self, dtype), self.device());
}

// Index reductions return Int64 coordinates; a missing dim reduces
// everything to a scalar.
Tensor arg_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (isComplexType(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "argmax: not implemented for complex dtypes");
    }
    if (!dim.has_value()) {
        return empty_meta({}, DType::Int64, self.device());
    }
    return empty_meta(reduce_out_shape(self, {*dim}, keepdim), DType::Int64,
                      self.device());
}

// The clipped extrema run their own reduction loop without a complex form;
// the dtype is policed after the reduction geometry is resolved.
Tensor amax_cpu(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    const auto out = reduce_out_shape(self, dim, keepdim);
    if (isComplexType(self.dtype())) {
        TP_THROW(TypeError, "extrema: unsupported dtype ",
                 toString(self.dtype()));
    }
    return empty_meta(out, self.dtype(), self.device());
}

Tensor amin_cpu(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    const auto out = reduce_out_shape(self, dim, keepdim);
    if (isComplexType(self.dtype())) {
        TP_THROW(TypeError, "extrema: unsupported dtype ",
                 toString(self.dtype()));
    }
    return empty_meta(out, self.dtype(), self.device());
}

std::tuple<Tensor, Tensor> aminmax_cpu(const Tensor& self,
                                       const std::vector<int64_t>& dim,
                                       bool keepdim) {
    const auto out_shape = reduce_out_shape(self, dim, keepdim);
    if (isComplexType(self.dtype())) {
        TP_THROW(TypeError, "extrema: unsupported dtype ",
                 toString(self.dtype()));
    }
    return {empty_meta(out_shape, self.dtype(), self.device()),
            empty_meta(out_shape, self.dtype(), self.device())};
}

std::tuple<Tensor, Tensor> aminmax_all_kernel(const Tensor& self) {
    return aminmax_cpu(self, {}, false);
}

std::tuple<Tensor, Tensor> aminmax_dim_kernel(const Tensor& self, int64_t dim,
                                              bool keepdim) {
    return aminmax_cpu(self, {dim}, keepdim);
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Moments. The variance and the mean reduce over the same dimensions, so they
// share a shape; the deviation is a distance and answers in the real
// component type, the mean keeps the operand's own plane.
// ---------------------------------------------------------------------------
std::tuple<Tensor, Tensor> var_mean_kernel(const Tensor& self,
                                           const std::vector<int64_t>& dims,
                                           bool /*unbiased*/, bool keepdim) {
    const std::vector<int64_t> out = reduce_out_shape(self, dims, keepdim);
    return {empty_meta(out, moment_dtype(self), self.device()),
            empty_meta(out, mean_like_dtype(self, DType::Undefined),
                       self.device())};
}

// The trapezoid rule pairs neighbouring samples, so it consumes one fewer
// segment than the sampled axis has points: the plain form drops the axis,
// the cumulative form keeps it with one fewer extent.  Fewer than two
// samples leave nothing to pair -- the axis vanishes or holds nothing -- and
// a coordinate tensor shorter than the sampled axis cannot pair.  The mean
// of neighbours leaves the integral types behind, since the operation has
// no integral form to answer in.
void check_trapezoid_axis(const Tensor& y, const std::optional<Tensor>& x,
                          int64_t dim, int64_t n) {
    if (!x.has_value() || !x->defined()) {
        return;
    }
    if (x->dim() == 1 && y.dim() > 1) {
        // The coordinate vector is viewed along the sampled axis and grown
        // to the operand's extents, which only fits at the same length.
        if (x->size(0) != n) {
            TP_THROW(RuntimeError,
                     "The expanded size of the tensor must match the existing "
                     "size at existing dimensions");
        }
        return;
    }
    // Neighbouring coordinates are sliced one past the first, so the
    // coordinate extent has to reach the last paired sample, and the
    // remaining extents must broadcast against the operand's.
    if (x->size(dim) < n) {
        TP_THROW(RuntimeError, "coordinate extent ", x->size(dim),
                 " is shorter than the sampled axis of ", n);
    }
    broadcast_shapes(shape_of(y), shape_of(*x));
}

Tensor trapezoid_kernel(const Tensor& y, const std::optional<Tensor>& x,
                        const Scalar& /*dx*/, int64_t dim) {
    const int64_t nd = y.dim();
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) {
        TP_THROW(IndexError, "Dimension out of range");
    }
    const int64_t n = y.size(dim);
    if (n >= 2) {
        check_trapezoid_axis(y, x, dim, n);
    }
    std::vector<int64_t> out = shape_of(y);
    out.erase(out.begin() + dim);
    return empty_meta(out, unary_float_dtype(y.dtype()), y.device());
}

Tensor cumulative_trapezoid_kernel(const Tensor& y,
                                   const std::optional<Tensor>& x,
                                   const Scalar& /*dx*/, int64_t dim) {
    const int64_t nd = y.dim();
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) {
        TP_THROW(IndexError, "Dimension out of range");
    }
    const int64_t n = y.size(dim);
    if (n >= 2) {
        check_trapezoid_axis(y, x, dim, n);
    }
    std::vector<int64_t> out = shape_of(y);
    out[static_cast<size_t>(dim)] = n >= 2 ? n - 1 : 0;
    return empty_meta(out, unary_float_dtype(y.dtype()), y.device());
}

// The covariance answers one row per variable: a rank-one operand measures a
// single variable and collapses to a scalar, a matrix answers its variable
// count squared, and singleton extents squeeze away.  Frequencies must count
// whole observations, reliabilities must weight them fractionally.
Tensor cov_kernel(const Tensor& self, int64_t /*correction*/,
                  const std::optional<Tensor>& fweights,
                  const std::optional<Tensor>& aweights) {
    if (self.dim() > 2) {
        TP_THROW(RuntimeError,
                 "cov(): expected input to have two or fewer dimensions but got "
                 "an input with ", self.dim(), " dimensions");
    }
    TP_CHECK_NOT_IMPLEMENTED(self.dtype() != DType::Bool,
                             "cov(): bool dtype is not supported for input");
    const int64_t observations = self.dim() < 2 ? self.numel() : self.size(1);
    if (fweights.has_value()) {
        if (fweights->dim() > 1) {
            TP_THROW(RuntimeError,
                     "cov(): expected fweights to have one or fewer dimensions "
                     "but got fweights with ", fweights->dim(), " dimensions");
        }
        TP_CHECK(isIntegralType(fweights->dtype(), /*includeBool=*/false),
                 "cov(): expected fweights to have integral dtype but got "
                 "fweights with dtype ", static_cast<int>(fweights->dtype()));
        TP_CHECK(fweights->numel() == observations,
                 "cov(): expected fweights to have the same numel as there are "
                 "observations in the input but got ", fweights->numel(),
                 " != ", observations);
    }
    if (aweights.has_value()) {
        if (aweights->dim() > 1) {
            TP_THROW(RuntimeError,
                     "cov(): expected aweights to have one or fewer dimensions "
                     "but got aweights with ", aweights->dim(), " dimensions");
        }
        TP_CHECK(isFloatingType(aweights->dtype()),
                 "cov(): expected aweights to have floating point dtype but got "
                 "aweights with dtype ", static_cast<int>(aweights->dtype()));
        TP_CHECK(aweights->numel() == observations,
                 "cov(): expected aweights to have the same numel as there are "
                 "observations in the input but got ", aweights->numel(),
                 " != ", observations);
    }
    const int64_t vars = self.dim() < 2 ? 1 : self.size(0);
    // The matrix form squeezes its singleton extents away.
    std::vector<int64_t> out;
    if (vars != 1) {
        out = {vars, vars};
    }
    return empty_meta(out, unary_float_dtype(self.dtype()), self.device());
}

Tensor corrcoef_kernel(const Tensor& self) {
    if (self.dim() > 2) {
        TP_THROW(RuntimeError,
                 "corrcoef(): expected input to have two or fewer dimensions but got "
                 "an input with ", self.dim(), " dimensions");
    }
    const int64_t vars = self.dim() < 2 ? 1 : self.size(0);
    std::vector<int64_t> out;
    if (vars != 1) {
        out = {vars, vars};
    }
    return empty_meta(out, unary_float_dtype(self.dtype()), self.device());
}

// Counting non-zero entries reduces like a sum over a boolean map: no axis
// named collapses to a scalar, every named axis drops, and the count is an
// integer.
Tensor count_nonzero_kernel(const Tensor& self,
                            const std::vector<int64_t>& dim) {
    if (dim.empty()) {
        return empty_meta({}, DType::Int64, self.device());
    }
    return empty_meta(reduce_out_shape(self, dim, /*keepdim=*/false),
                      DType::Int64, self.device());
}

// A fixed bin count fixes both answers: one count per bin beside one edge
// per bin boundary.  Without an explicit range the span comes from the
// stored extrema, which a metadata-only device cannot read, so that
// validation is left to a materializing backend.
std::tuple<Tensor, Tensor> histogram_bin_ct_kernel(
    const Tensor& self, int64_t bins,
    const std::optional<std::vector<double>>& range,
    const std::optional<Tensor>& weight, bool /*density*/) {
    if (self.dtype() != DType::Float16 && self.dtype() != DType::BFloat16 &&
        self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(TypeError,
                 "histogram(): input tensor must have a floating-point dtype");
    }
    if (bins < 1) {
        TP_THROW(ValueError, "histogram(): bins must be > 0, but got ", bins,
                 " for dimension 0");
    }
    if (weight.has_value()) {
        if (weight->dtype() != self.dtype()) {
            TP_THROW(ValueError,
                     "histogramdd: if weight tensor is provided, input "
                     "tensor and weight tensor should have the same dtype");
        }
        if (weight->numel() != self.numel()) {
            TP_THROW(ValueError,
                     "histogramdd: if weight tensor is provided, it should have "
                     "the same number of elements as the input tensor");
        }
    }
    if (range.has_value()) {
        if (range->size() != 2) {
            TP_THROW(ValueError,
                     "histogramdd: for a 1-dimensional histogram range "
                     "should have 2 elements, but got ", range->size());
        }
        const double leftmost = (*range)[0];
        const double rightmost = (*range)[1];
        if (!std::isfinite(leftmost) || !std::isfinite(rightmost)) {
            TP_THROW(ValueError, "histogramdd: dimension 0's range [",
                     leftmost, ", ", rightmost, "] is not finite");
        }
        if (leftmost > rightmost) {
            TP_THROW(ValueError,
                     "histogramdd: dimension 0's lower bound ", leftmost,
                     " is larger than the upper bound ", rightmost);
        }
    }
    return {empty_meta({bins}, self.dtype(), self.device()),
            empty_meta({bins + 1}, self.dtype(), self.device())};
}
} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaReduce) {
    m.impl("var", var_kernel);
    m.impl("var.dim", var_dim_kernel);
    m.impl("std", std_kernel);
    m.impl("std.dim", std_dim_kernel);
    m.impl("std_mean", std_mean_kernel);
    m.impl("max", extrema_kernel);
    m.impl("max.dim", extrema_dim_kernel);
    m.impl("min", extrema_kernel);
    m.impl("min.dim", extrema_dim_kernel);
    m.impl("all", all_kernel);
    m.impl("all.dim", all_dim_kernel);
    m.impl("any", any_kernel);
    m.impl("any.dim", any_dim_kernel);
    m.impl("logsumexp", logsumexp_kernel);
    m.impl("mode", mode_kernel);
    m.impl("kthvalue", kthvalue_kernel);
    m.impl("median", median_flat_kernel);
    m.impl("nanmedian", nanmedian_flat_kernel);
    m.impl("nanmedian.dim", nanmedian_dim_kernel);
    m.impl("cummax", cummax_kernel);
    m.impl("cummin", cummin_kernel);
    m.impl("nansum", nansum_kernel);
    m.impl("nanmean", nanmean_kernel);
    m.impl("logcumsumexp", logcumsumexp_kernel);
    m.impl("sum", sum_kernel);
    m.impl("sum.dim_IntList", sum_dim_kernel);
    m.impl("mean", mean_kernel);
    m.impl("mean.dim", mean_dim_kernel);
    m.impl("prod", prod_kernel);
    m.impl("prod.dim_IntList", prod_dim_kernel);
    m.impl("argmax", arg_kernel);
    m.impl("argmin", arg_kernel);
    m.impl("amax", amax_cpu);
    m.impl("amin", amin_cpu);
    m.impl("aminmax", aminmax_cpu);
    m.impl("_aminmax", aminmax_all_kernel);
    m.impl("_aminmax.dim", aminmax_dim_kernel);
    m.impl("var_mean", var_mean_kernel);
    m.impl("trapezoid", trapezoid_kernel);
    m.impl("cumulative_trapezoid", cumulative_trapezoid_kernel);
    m.impl("cov", cov_kernel);
    m.impl("corrcoef", corrcoef_kernel);
    m.impl("count_nonzero", count_nonzero_kernel);
    m.impl("histogram.bin_ct", histogram_bin_ct_kernel);
}

} // namespace meta
} // namespace tensorplay
