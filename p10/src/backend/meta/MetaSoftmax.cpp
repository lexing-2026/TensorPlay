// Shape-only kernels for the softmax family on the meta device.
//
// A softmax normalizes each row along one axis; the normalization changes
// values only, so the result repeats the input's sizes and -- apart from an
// explicit dtype request or a half-to-float upcast -- its element type.
// Integer inputs answer at single precision.  The backward kernels describe
// a gradient with the forward output's geometry and the forward input's
// dtype.

#include "MetaCommon.h"

#include "DType.h"
#include "TypeProperties.h"

#include <cstdint>
#include <optional>

namespace tensorplay {
namespace meta {

namespace {

// The dtype a softmax answers in: the requested one, else the input's own;
// whole-number inputs answer at single precision.
DType softmax_answer_dtype(DType self_dtype, DType asked) {
    DType out = (asked != DType::Undefined) ? asked : self_dtype;
    if (isIntegralType(out)) out = DType::Float32;
    return out;
}

// Normalize the softmax axis: a rank-0 tensor has one virtual dimension
// (0 / -1), and anything else must name a real axis.
int64_t normalize_softmax_dim(int64_t dim, int64_t ndim) {
    if (ndim == 0) {
        const int64_t d = dim < 0 ? dim + 1 : dim;
        if (d != 0) {
            TP_THROW(IndexError, "dim must be non-negative and less than input "
                     "dimensions");
        }
        return 0;
    }
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "dim must be non-negative and less than input "
                 "dimensions");
    }
    return dim;
}

// The normalization runs on real numbers only: complex planes and booleans
// have no form here and are refused after the axis is named.
void require_softmax_dtype(DType dt) {
    if (isComplexType(dt) || dt == DType::Bool) {
        TP_THROW(TypeError, "softmax: unsupported dtype");
    }
}

Tensor softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
    (void)normalize_softmax_dim(dim, self.dim());
    require_softmax_dtype(self.dtype());
    return empty_meta(shape_of(self), softmax_answer_dtype(self.dtype(), dtype),
                      self.device());
}

Tensor log_softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
    (void)normalize_softmax_dim(dim, self.dim());
    require_softmax_dtype(self.dtype());
    return empty_meta(shape_of(self), softmax_answer_dtype(self.dtype(), dtype),
                      self.device());
}

// The lower-level forward takes an explicit half-to-float flag instead of a
// dtype request; the flag lifts reduced-precision inputs to Float32.
Tensor softmax_native_kernel(const Tensor& self, int64_t dim,
                             bool half_to_float) {
    (void)normalize_softmax_dim(dim, self.dim());
    const DType out = (half_to_float &&
                       (self.dtype() == DType::Float16 ||
                        self.dtype() == DType::BFloat16))
                          ? DType::Float32
                          : self.dtype();
    return empty_meta(shape_of(self), out, self.device());
}

Tensor log_softmax_native_kernel(const Tensor& self, int64_t dim,
                                 bool half_to_float) {
    (void)normalize_softmax_dim(dim, self.dim());
    const DType out = (half_to_float &&
                       (self.dtype() == DType::Float16 ||
                        self.dtype() == DType::BFloat16))
                          ? DType::Float32
                          : self.dtype();
    return empty_meta(shape_of(self), out, self.device());
}

// The dtype a softmax backward answers in: the forward input's own type,
// except that a reduced-precision forward computes in Float32 and only comes
// back down when the incoming gradient is already Float32.
DType softmax_backward_dtype(DType input_dtype, DType grad_dtype) {
    const bool reduced = isReducedFloatingType(input_dtype);
    if (reduced && grad_dtype != DType::Float32) return DType::Float32;
    return input_dtype;
}

Tensor softmax_backward_kernel(const Tensor& grad_output, const Tensor& output,
                               int64_t dim, DType input_dtype) {
    (void)normalize_softmax_dim(dim, grad_output.dim());
    return empty_meta(shape_of(grad_output),
                      softmax_backward_dtype(input_dtype, grad_output.dtype()),
                      grad_output.device());
}

Tensor log_softmax_backward_kernel(const Tensor& grad_output,
                                   const Tensor& output, int64_t dim,
                                   DType input_dtype) {
    (void)normalize_softmax_dim(dim, grad_output.dim());
    return empty_meta(shape_of(grad_output),
                      softmax_backward_dtype(input_dtype, grad_output.dtype()),
                      grad_output.device());
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaSoftmax) {
    m.impl("softmax", softmax_kernel);
    m.impl("log_softmax", log_softmax_kernel);
    m.impl("_softmax", softmax_native_kernel);
    m.impl("_log_softmax", log_softmax_native_kernel);
    m.impl("_softmax_backward_data", softmax_backward_kernel);
    m.impl("_log_softmax_backward_data", log_softmax_backward_kernel);
}

} // namespace meta
} // namespace tensorplay