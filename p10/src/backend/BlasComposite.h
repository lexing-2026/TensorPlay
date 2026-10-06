#pragma once

// addmv, addr and addbmm for the element types the BLAS kernels do not
// serve -- complex values and whole numbers (and truth values for addr) --
// composed from the products themselves:
//
//     beta * self + alpha * product
//
// with self ignored when beta is zero, its nan and inf included.  Shared by
// the CPU and CUDA kernels.

#include "Tensor.h"
#include "TypePromotion.h"
#include "Scalar.h"
#include "Complex.h"
#include "Exception.h"

#include <string>

namespace tensorplay {
namespace blas_composite {

inline bool scalar_equals(const Scalar& s, double v) {
    if (s.isComplex()) {
        const auto c = s.to<complex<double>>();
        return c.real() == v && c.imag() == 0.0;
    }
    return s.toDouble() == v;
}

// A whole-number result takes whole-number scalars, and only a truth-value
// result takes truth values.
inline void check_scalar(DType dtype, const Scalar& s, const char* name) {
    if (s.isBoolean() && dtype != DType::Bool) {
        TP_THROW(RuntimeError, "Boolean ", name, " only supported for Boolean results.");
    }
    if (!isFloatingType(dtype) && !isComplexType(dtype) && !s.isIntegral(true)) {
        TP_THROW(RuntimeError, "For integral input tensors, argument ", name,
                 " must not be a floating point number.");
    }
}

inline Tensor scale_add(const Tensor& self, const Tensor& product, DType dtype,
                        const Scalar& beta, const Scalar& alpha) {
    const std::vector<int64_t> shape = product.shape();
    Tensor scaled = scalar_equals(alpha, 1.0) ? product : product * alpha;
    if (scalar_equals(beta, 0.0)) {
        return scaled.dtype() == dtype ? scaled : scaled.to(dtype);
    }
    const Tensor base = self.expand(shape);
    Tensor out = (scalar_equals(beta, 1.0) ? base : base * beta) + scaled;
    return out.dtype() == dtype ? out : out.to(dtype);
}

inline Tensor addmv(const Tensor& self, const Tensor& mat, const Tensor& vec,
                    const Scalar& beta, const Scalar& alpha) {
    const DType dtype = promoteTypes(promoteTypes(mat.dtype(), vec.dtype()), self.dtype());
    if (dtype == DType::Bool) {
        TP_THROW(NotImplementedError, "\"addmv\" not implemented for 'Bool'");
    }
    if (mat.dim() != 2) TP_THROW(RuntimeError, "addmv: mat must be a matrix");
    if (vec.dim() != 1) TP_THROW(RuntimeError, "addmv: vec must be a vector");
    if (vec.numel() != mat.size(1)) {
        TP_THROW(RuntimeError, "addmv: both args should have matching shapes");
    }
    check_scalar(dtype, beta, "beta");
    check_scalar(dtype, alpha, "alpha");
    return scale_add(self, mat.to(dtype).mv(vec.to(dtype)), dtype, beta, alpha);
}

inline Tensor addbmm(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                     const Scalar& beta, const Scalar& alpha) {
    const DType dtype = promoteTypes(promoteTypes(batch1.dtype(), batch2.dtype()), self.dtype());
    if (dtype == DType::Bool) {
        TP_THROW(NotImplementedError, "\"addbmm\" not implemented for 'Bool'");
    }
    if (batch1.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "Incompatible matrix sizes for bmm (",
                 batch1.size(1), "x", batch1.size(2), " and ",
                 batch2.size(1), "x", batch2.size(2), ")");
    }
    check_scalar(dtype, beta, "beta");
    check_scalar(dtype, alpha, "alpha");
    const Tensor product = batch1.to(dtype).bmm(batch2.to(dtype)).sum(0);
    return scale_add(self, product, dtype, beta, alpha);
}

inline Tensor addr(const Tensor& self, const Tensor& vec1, const Tensor& vec2,
                   const Scalar& beta, const Scalar& alpha) {
    const DType dtype = promoteTypes(promoteTypes(vec1.dtype(), vec2.dtype()), self.dtype());
    if (vec1.dim() != 1) {
        TP_THROW(RuntimeError, "addr: Expected 1-D argument vec1, but got ", vec1.dim(), "-D");
    }
    if (vec2.dim() != 1) {
        TP_THROW(RuntimeError, "addr: Expected 1-D argument vec2, but got ", vec2.dim(), "-D");
    }
    check_scalar(dtype, beta, "beta");
    check_scalar(dtype, alpha, "alpha");
    const std::vector<int64_t> shape = {vec1.size(0), vec2.size(0)};
    if (dtype == DType::Bool) {
        // (beta and self) or (alpha and vec1[i] and vec2[j]).
        const Tensor outer = vec1.to(DType::Bool).unsqueeze(1).logical_and(
            vec2.to(DType::Bool).unsqueeze(0));
        const Tensor product = alpha.to<bool>() ? outer : outer.logical_and(outer.logical_not());
        if (!beta.to<bool>()) return product;
        return self.expand(shape).to(DType::Bool).logical_or(product);
    }
    return scale_add(self, vec1.to(dtype).outer(vec2.to(dtype)), dtype, beta, alpha);
}

}  // namespace blas_composite
}  // namespace tensorplay
