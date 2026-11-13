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
}

} // namespace meta
} // namespace tensorplay
