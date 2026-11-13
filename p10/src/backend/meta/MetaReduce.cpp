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

// The paired form reports the moment and the mean, which share their shape
// and their value type.
std::tuple<Tensor, Tensor> std_mean_kernel(const Tensor& self,
                                           const std::vector<int64_t>& dim,
                                           bool /*unbiased*/, bool keepdim) {
    const std::vector<int64_t> out = reduce_out_shape(self, dim, keepdim);
    const DType dtype = moment_dtype(self);
    return {empty_meta(out, dtype, self.device()),
            empty_meta(out, dtype, self.device())};
}

// ---------------------------------------------------------------------------
// Extrema.  The extrema without a dimension answer a scalar; with a dimension
// they answer the winning values and their positions along it.
// ---------------------------------------------------------------------------

Tensor extrema_kernel(const Tensor& self) {
    return empty_meta({}, self.dtype(), self.device());
}

std::tuple<Tensor, Tensor> extrema_dim_kernel(const Tensor& self, int64_t dim,
                                              bool keepdim) {
    dim = wrap_dim(dim, self.dim());
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
    require_scalar_elements(self, "median");
    return empty_meta({}, self.dtype(), self.device());
}

Tensor nanmedian_flat_kernel(const Tensor& self) {
    require_scalar_elements(self, "nanmedian");
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
}

} // namespace meta
} // namespace tensorplay
