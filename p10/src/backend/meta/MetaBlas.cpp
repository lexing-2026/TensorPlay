// Shape-only kernels for the matrix products and the vector/batch
// accumulators on the meta device.  Each answer carries the contraction
// geometry its operands allow -- batch extents broadcast, the contracted
// dimensions disappear -- and the promotion of the participating
// element types, with the alpha/beta scalars lifting integral
// accumulations to a floating type.
#include "MetaCommon.h"

#include "Tensor.h"
#include "TensorImpl.h"
#include "Convolution.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Utils.h"
#include "Context.h"
#include "Generator.h"
#include "Scalar.h"
#include "DType.h"
#include "DTypeNames.h"
#include "DistributionDispatch.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "../BlasComposite.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace tensorplay {
namespace meta {
namespace {
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Matrix products.
// ---------------------------------------------------------------------------
Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
    if (self.dim() != 2) TP_THROW(RuntimeError, "self must be a matrix");
    if (mat2.dim() != 2) TP_THROW(RuntimeError, "mat2 must be a matrix");
    if (self.size(1) != mat2.size(0)) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
                 self.size(0), "x", self.size(1), " and ", mat2.size(0), "x",
                 mat2.size(1), ")");
    }
    if (self.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError,
                 "expected m1 and m2 to have the same dtype, but got: ",
                 elementTypeName(self.dtype()), " != ",
                 elementTypeName(mat2.dtype()));
    }
    if (self.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError,
                 "\"addmm_impl_cpu_\" not implemented for 'Bool'");
    }
    return empty_meta({self.size(0), mat2.size(1)}, self.dtype(),
                      self.device());
}

// General N-D product shapes: vector operands fold to rows/columns, batch
// dimensions broadcast, and the matrix part follows mm.
Tensor matmul_kernel(const Tensor& self, const Tensor& other) {
    if (self.dim() < 1 || other.dim() < 1) {
        TP_THROW(RuntimeError, "matmul(): input operands must be at least 1D");
    }
    check_all_meta(self, other);
    const std::vector<int64_t> a = shape_of(self);
    const std::vector<int64_t> b = shape_of(other);
    const int64_t dim1 = self.dim();
    const int64_t dim2 = other.dim();
    std::vector<int64_t> out;

    if (dim1 == 1 && dim2 == 1) {
        if (a[0] != b[0]) {
            TP_THROW(RuntimeError, "inconsistent tensor size, expected tensor [",
                     a[0], "] and src [", b[0],
                     "] to have the same number of elements");
        }
    } else if (dim1 == 1) {
        const int64_t k = a[0];
        const int64_t other_k = b[b.size() - 2];
        if (k != other_k) {
            TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (1x", k,
                     " and ", other_k, "x", b.back(), ")");
        }
        out.assign(b.begin(), b.end() - 2);
        out.push_back(b.back());
    } else if (dim2 == 1) {
        const int64_t k = a.back();
        if (k != b[0]) {
            int64_t folded_m = a[a.size() - 2];
            for (size_t i = 0; i + 2 < a.size(); ++i) folded_m *= a[i];
            TP_THROW(RuntimeError, "size mismatch, got input (", folded_m,
                     "), mat (", folded_m, "x", k, "), vec (", b[0], ")");
        }
        out.assign(a.begin(), a.end() - 2);
        out.push_back(a[a.size() - 2]);
    } else {
        const int64_t k = a.back();
        const int64_t other_k = b[b.size() - 2];
        if (k != other_k) {
            if (dim2 == 2) {
                int64_t folded_m = a[a.size() - 2];
                for (size_t i = 0; i + 2 < a.size(); ++i) folded_m *= a[i];
                TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
                         folded_m, "x", k, " and ", other_k, "x", b.back(), ")");
            }
            std::vector<int64_t> self_batch(a.begin(), a.end() - 2);
            std::vector<int64_t> other_batch(b.begin(), b.end() - 2);
            const std::vector<int64_t> batch = broadcast_shapes(self_batch, other_batch);
            int64_t prod_batch = 1;
            for (const int64_t s : batch) prod_batch *= s;
            TP_THROW(RuntimeError,
                     "Expected size for first two dimensions of batch2 tensor to be: [",
                     prod_batch, ", ", k, "] but got: [", prod_batch, ", ", other_k, "].");
        }
        out = broadcast_shapes(
            std::vector<int64_t>(a.begin(), a.end() - 2),
            std::vector<int64_t>(b.begin(), b.end() - 2));
        out.push_back(a[a.size() - 2]);
        out.push_back(b.back());
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError,
                 "expected m1 and m2 to have the same dtype, but got: ",
                 elementTypeName(self.dtype()), " != ",
                 elementTypeName(other.dtype()));
    }
    if (self.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError,
                 "\"addmm_impl_cpu_\" not implemented for 'Bool'");
    }
    return empty_meta(out, self.dtype(), self.device());
}

// Reshape only moves metadata; a -1 slot infers from the element count.
Tensor reshape_kernel(const Tensor& self, const std::vector<int64_t>& shape) {
    int64_t known = 1;
    int64_t infer = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (infer >= 0) {
                TP_THROW(RuntimeError, "only one dimension can be inferred");
            }
            infer = static_cast<int64_t>(i);
        } else {
            known *= shape[i];
        }
    }
    std::vector<int64_t> out = shape;
    if (infer >= 0) {
        if (known == 0 || self.numel() % known != 0) {
            TP_THROW(RuntimeError, "shape '", self.numel(),
                     "' is invalid for input of size ", self.numel());
        }
        out[static_cast<size_t>(infer)] = self.numel() / known;
    }
    int64_t total = 1;
    for (const int64_t d : out) total *= d;
    if (total != self.numel()) {
        TP_THROW(RuntimeError, "shape '", total,
                 "' is invalid for input of size ", self.numel());
    }
    return empty_meta(out, self.dtype(), self.device());
}

Tensor bmm_kernel(const Tensor& self, const Tensor& batch2) {
    if (self.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (self.size(0) != batch2.size(0) || self.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError,
                 "Expected size for first two dimensions of batch2 tensor to "
                 "be: [", self.size(0), ", ", self.size(2), "] but got: [",
                 batch2.size(0), ", ", batch2.size(1), "].");
    }
    if (self.dtype() != batch2.dtype()) {
        TP_THROW(RuntimeError, "expected scalar type ",
                 scalarTypeName(self.dtype()), " but found ",
                 scalarTypeName(batch2.dtype()));
    }
    if (self.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError,
                 "\"addmm_impl_cpu_\" not implemented for 'Bool'");
    }
    return empty_meta({self.size(0), self.size(1), batch2.size(2)},
                      self.dtype(), self.device());
}

// addmm broadcasts the bias input against the product of the two matrices.
// Every operand shares one dtype; a whole-number product is blended at single
// precision.  A zero weight leaves the bias unread.
std::string matmul_operand_desc(const Tensor& t) {
    if (t.dim() != 2) return std::to_string(t.dim()) + "D";
    return std::to_string(t.size(0)) + "x" + std::to_string(t.size(1));
}

void check_addmm_operands(const Tensor& input, const Tensor& mat1,
                          const Tensor& mat2) {
    if (mat1.dim() != 2 || mat2.dim() != 2) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
                 matmul_operand_desc(mat1), " and ",
                 matmul_operand_desc(mat2), ")");
    }
    if (mat1.size(1) != mat2.size(0)) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
                 mat1.size(0), "x", mat1.size(1), " and ", mat2.size(0), "x",
                 mat2.size(1), ")");
    }
    if (mat1.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError,
                 "mat1 and mat2 must have the same dtype, but got ",
                 scalarTypeName(mat1.dtype()), " and ",
                 scalarTypeName(mat2.dtype()));
    }
    if (input.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError,
                 "self and mat2 must have the same dtype, but got ",
                 scalarTypeName(input.dtype()), " and ",
                 scalarTypeName(mat2.dtype()));
    }
    if (mat2.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError,
                 "\"addmm_impl_cpu_\" not implemented for 'Bool'");
    }
}

bool scalar_is_zero(const Scalar& s) {
    if (s.isComplex()) {
        return s.to<complex<double>>() == complex<double>(0.0, 0.0);
    }
    return s.toDouble() == 0.0;
}

void require_expandable_to(const Tensor& self,
                           const std::vector<int64_t>& target,
                           const char* op) {
    if (self.dim() > static_cast<int64_t>(target.size()) ||
        broadcast_shapes(shape_of(self), target) != target) {
        TP_THROW(RuntimeError,
                 op, ": the addend's shape cannot be expanded to the result's "
                 "shape");
    }
}

Tensor addmm_kernel(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
                    const Scalar& beta, const Scalar& /*alpha*/) {
    check_addmm_operands(input, mat1, mat2);
    const std::vector<int64_t> target{mat1.size(0), mat2.size(1)};
    if (!scalar_is_zero(beta)) {
        require_expandable_to(input, target, "addmm");
    }
    const DType out = isIntegralType(mat2.dtype(), /*include_bool=*/true)
                          ? DType::Float32
                          : mat2.dtype();
    return empty_meta(target, out, mat1.device());
}

// The in-place form lands the product back on the bias, which must be able to
// hold the whole result.
Tensor& addmm_inplace_kernel(Tensor& self, const Tensor& mat1,
                             const Tensor& mat2, const Scalar& beta,
                             const Scalar& alpha) {
    const Tensor result = addmm_kernel(self, mat1, mat2, beta, alpha);
    require_expandable_to(result, shape_of(self), "addmm_");
    return self;
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Vector products and the batch accumulators.  dot and mv require equal value
// types, inner contracts the last axes and stacks the leading ones, and the
// accumulators seed a broadcast addend with the product.
// ---------------------------------------------------------------------------
void check_matmul_value_dtype(DType dtype, const char* op) {
    switch (dtype) {
        case DType::Bool:
        case DType::UInt16:
        case DType::UInt32:
        case DType::UInt64:
        case DType::ComplexHalf:
        case DType::BComplex32:
            TP_THROW(NotImplementedError, op, ": unsupported dtype ",
                     scalarTypeName(dtype));
        default:
            return;
    }
}

Tensor dot_kernel(const Tensor& self, const Tensor& other) {
    if (self.dim() != 1 || other.dim() != 1) {
        TP_THROW(RuntimeError, "1D tensors expected, but got ", self.dim(),
                 "D and ", other.dim(), "D tensors");
    }
    if (self.size(0) != other.size(0)) {
        TP_THROW(RuntimeError, "inconsistent tensor size, expected tensor [",
                 self.size(0), "] and src [", other.size(0),
                 "] to have the same number of elements, but got ",
                 self.size(0), " and ", other.size(0),
                 " elements respectively");
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError,
                 "dot : expected both vectors to have same dtype, but found ",
                 scalarTypeName(self.dtype()), " and ",
                 scalarTypeName(other.dtype()));
    }
    check_matmul_value_dtype(self.dtype(), "dot");
    return empty_meta({}, self.dtype(), self.device());
}

Tensor mv_kernel(const Tensor& self, const Tensor& vec) {
    if (self.dim() != 2 || vec.dim() != 1) {
        TP_THROW(RuntimeError, "vector + matrix @ vector expected, got ", 1,
                 ", ", self.dim(), ", ", vec.dim());
    }
    if (self.size(1) != vec.size(0)) {
        TP_THROW(RuntimeError, "size mismatch, got input (", self.size(0),
                 "), mat (", self.size(0), "x", self.size(1), "), vec (",
                 vec.size(0), ")");
    }
    if (self.dtype() != vec.dtype()) {
        // The leading slot is the accumulator's; a matrix-vector product
        // carries the vector's type there.
        TP_THROW(RuntimeError,
                 "addmv input tensors must have the same dtype, but got ",
                 scalarTypeName(vec.dtype()), ", ",
                 scalarTypeName(self.dtype()), ", and ",
                 scalarTypeName(vec.dtype()));
    }
    check_matmul_value_dtype(self.dtype(), "mv");
    return empty_meta({self.size(0)}, self.dtype(), self.device());
}

Tensor inner_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    if (self.dim() == 0 || other.dim() == 0) {
        return empty_meta(broadcast2(self, other),
                          promoteTypes(self.dtype(), other.dtype()),
                          self.device());
    }
    if (self.size(-1) != other.size(-1)) {
        TP_THROW(RuntimeError,
                 "inner() the last dimension must match on both input tensors "
                 "but got shapes ",
                 Size(shape_of(self)).toString(), " and ",
                 Size(shape_of(other)).toString());
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError,
                 "expected m1 and m2 to have the same dtype, but got: ",
                 elementTypeName(self.dtype()), " != ",
                 elementTypeName(other.dtype()));
    }
    if (self.dim() == 1 && other.dim() == 1) {
        return dot_kernel(self, other);
    }
    check_matmul_value_dtype(self.dtype(), "inner");
    const std::vector<int64_t> self_shape = shape_of(self);
    const std::vector<int64_t> other_shape = shape_of(other);
    std::vector<int64_t> out(self_shape.begin(), self_shape.end() - 1);
    out.insert(out.end(), other_shape.begin(), other_shape.end() - 1);
    return empty_meta(out, self.dtype(), self.device());
}

Tensor ger_kernel(const Tensor& self, const Tensor& vec2) {
    if (self.dim() != 1 || vec2.dim() != 1) {
        TP_THROW(RuntimeError, "1D tensors expected, but got ", self.dim(),
                 "D and ", vec2.dim(), "D tensors");
    }
    return empty_meta({self.numel(), vec2.numel()},
                      promoteTypes(self.dtype(), vec2.dtype()),
                      self.device());
}

Tensor baddbmm_kernel(const Tensor& self, const Tensor& batch1,
                      const Tensor& batch2, const Scalar& beta,
                      const Scalar& /*alpha*/) {
    if (batch1.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError,
                 "Expected size for first two dimensions of batch2 tensor to "
                 "be: [", batch1.size(0), ", ", batch1.size(2), "] but got: [",
                 batch2.size(0), ", ", batch2.size(1), "].");
    }
    if (self.dtype() != batch1.dtype() || batch1.dtype() != batch2.dtype()) {
        TP_THROW(RuntimeError, "Input dtypes must be the same, got: input ",
                 elementTypeName(self.dtype()), ", batch1: ",
                 elementTypeName(batch1.dtype()), ", batch2: ",
                 elementTypeName(batch2.dtype()));
    }
    check_matmul_value_dtype(self.dtype(), "baddbmm");
    const std::vector<int64_t> target{batch1.size(0), batch1.size(1),
                                      batch2.size(2)};
    // A zero weight leaves the addend unread, so only a weighted seed has to
    // reach the product's shape.
    if (!scalar_is_zero(beta)) {
        require_expandable_to(self, target, "baddbmm");
    }
    // The product is blended back through an add whose weight carries single
    // precision, which lifts a whole-number result to a real type.
    const DType out =
        add_like_dtype(self.dtype(), self.dtype(), Scalar(1.0));
    return empty_meta(target, out, self.device());
}

Tensor addbmm_kernel(const Tensor& self, const Tensor& batch1,
                     const Tensor& batch2, const Scalar& beta,
                     const Scalar& alpha) {
    if (batch1.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "Incompatible matrix sizes for bmm (",
                 batch1.size(1), "x", batch1.size(2), " and ",
                 batch2.size(1), "x", batch2.size(2), ")");
    }
    const DType out = promoteTypes(promoteTypes(batch1.dtype(), batch2.dtype()),
                                   self.dtype());
    if (out == DType::Bool) {
        TP_THROW(NotImplementedError, "\"addbmm\" not implemented for 'Bool'");
    }
    if (!isFloatingType(out) && !isComplexType(out)) {
        if (!beta.isIntegral(/*includeBool=*/true)) {
            TP_THROW(RuntimeError,
                     "For integral input tensors, argument beta must not be a "
                     "floating point number.");
        }
        if (!alpha.isIntegral(/*includeBool=*/true)) {
            TP_THROW(RuntimeError,
                     "For integral input tensors, argument alpha must not be a "
                     "floating point number.");
        }
    }
    const std::vector<int64_t> target{batch1.size(1), batch2.size(2)};
    // A zero weight on a whole-number addend leaves it unread, and that route
    // skips the broadcast check.
    if (!(scalar_is_zero(beta) && !isFloatingType(out))) {
        require_expandable_to(self, target, "addbmm");
    }
    return empty_meta(target, out, self.device());
}

Tensor& baddbmm_inplace_kernel(Tensor& self, const Tensor& batch1,
                               const Tensor& batch2, const Scalar& /*beta*/,
                               const Scalar& /*alpha*/) {
    baddbmm_kernel(self, batch1, batch2, Scalar(1), Scalar(1));
    return self;
}

Tensor& addbmm_inplace_kernel(Tensor& self, const Tensor& batch1,
                              const Tensor& batch2, const Scalar& /*beta*/,
                              const Scalar& /*alpha*/) {
    addbmm_kernel(self, batch1, batch2, Scalar(1), Scalar(1));
    return self;
}

// addr and addmv blend a broadcast addend with a product under one shared
// scalar reading: a truth-value weight needs a truth-value result, and a
// whole-number result takes only whole-number weights.  The floating route
// answers at the promotion of all three operands; other value types compose
// through broadcast arithmetic, and a zero weight leaves the addend unread.
bool all_floating(const Tensor& a, const Tensor& b, const Tensor& c) {
    return isFloatingType(a.dtype()) && isFloatingType(b.dtype()) &&
           isFloatingType(c.dtype());
}

Tensor addr_kernel(const Tensor& self, const Tensor& vec1, const Tensor& vec2,
                   const Scalar& beta, const Scalar& alpha) {
    const DType dtype =
        promoteTypes(promoteTypes(vec1.dtype(), vec2.dtype()), self.dtype());
    const bool native = all_floating(self, vec1, vec2);
    if (!native) {
        if (vec1.dim() != 1) {
            TP_THROW(RuntimeError, "addr: Expected 1-D argument vec1, but got ",
                     vec1.dim(), "-D");
        }
        if (vec2.dim() != 1) {
            TP_THROW(RuntimeError, "addr: Expected 1-D argument vec2, but got ",
                     vec2.dim(), "-D");
        }
    }
    blas_composite::check_scalar(dtype, beta, "beta");
    blas_composite::check_scalar(dtype, alpha, "alpha");
    const std::vector<int64_t> target{vec1.numel(), vec2.numel()};
    // The floating product reaches for the addend whatever its weight; the
    // composed product only reads it when the weight is nonzero.
    if (native || !scalar_is_zero(beta)) {
        require_expandable_to(self, target, "addr");
    }
    return empty_meta(target, dtype, self.device());
}

Tensor& addr_inplace_kernel(Tensor& self, const Tensor& vec1,
                            const Tensor& vec2, const Scalar& beta,
                            const Scalar& alpha) {
    const Tensor result = addr_kernel(self, vec1, vec2, beta, alpha);
    require_expandable_to(result, shape_of(self), "addr_");
    return self;
}

Tensor addmv_kernel(const Tensor& self, const Tensor& mat, const Tensor& vec,
                    const Scalar& beta, const Scalar& alpha) {
    const DType dtype =
        promoteTypes(promoteTypes(mat.dtype(), vec.dtype()), self.dtype());
    const bool native = all_floating(self, mat, vec);
    if (!native && dtype == DType::Bool) {
        TP_THROW(NotImplementedError, "\"addmv\" not implemented for 'Bool'");
    }
    if (mat.dim() != 2) TP_THROW(RuntimeError, "addmv: mat must be a matrix");
    if (vec.dim() != 1) TP_THROW(RuntimeError, "addmv: vec must be a vector");
    if (vec.numel() != mat.size(1)) {
        TP_THROW(RuntimeError, "addmv: both args should have matching shapes");
    }
    if (!native) {
        blas_composite::check_scalar(dtype, beta, "beta");
        blas_composite::check_scalar(dtype, alpha, "alpha");
    }
    if (!scalar_is_zero(beta)) {
        require_expandable_to(self, {mat.size(0)}, "addmv");
    }
    return empty_meta({mat.size(0)}, dtype, self.device());
}

Tensor& addmv_inplace_kernel(Tensor& self, const Tensor& mat, const Tensor& vec,
                             const Scalar& beta, const Scalar& alpha) {
    const Tensor result = addmv_kernel(self, mat, vec, beta, alpha);
    require_expandable_to(result, shape_of(self), "addmv_");
    return self;
}

// The ternary blend of a product or quotient into a scaled addend.  Division
// lifts a whole-number result to single precision and refuses an all-integer
// quotient outright; the in-place forms land back on the addend, which has to
// already hold the three-way broadcast.
Tensor addcmul_kernel(const Tensor& self, const Tensor& tensor1,
                      const Tensor& tensor2, const Scalar& /*value*/) {
    return empty_meta(broadcast_shapes(shape_of(self), shape_of(tensor1),
                                       shape_of(tensor2)),
                      promoteTypes(promoteTypes(self.dtype(), tensor1.dtype()),
                                   tensor2.dtype()),
                      self.device());
}

Tensor& addcmul_inplace_kernel(Tensor& self, const Tensor& tensor1,
                               const Tensor& tensor2, const Scalar& /*value*/) {
    if (broadcast_shapes(shape_of(self), shape_of(tensor1),
                         shape_of(tensor2)) != shape_of(self)) {
        TP_THROW(RuntimeError, "addcmul_: output shape does not match self");
    }
    return self;
}

void check_addcdiv_dtypes(const Tensor& tensor1, const Tensor& tensor2) {
    if (isIntegralType(tensor1.dtype(), /*include_bool=*/true) &&
        isIntegralType(tensor2.dtype(), /*include_bool=*/true)) {
        TP_THROW(RuntimeError, "Integer division with addcdiv is not supported");
    }
}

Tensor addcdiv_kernel(const Tensor& self, const Tensor& tensor1,
                      const Tensor& tensor2, const Scalar& /*value*/) {
    check_addcdiv_dtypes(tensor1, tensor2);
    DType out = promoteTypes(promoteTypes(self.dtype(), tensor1.dtype()),
                             tensor2.dtype());
    if (isIntegralType(out)) out = DType::Float32;
    return empty_meta(broadcast_shapes(shape_of(self), shape_of(tensor1),
                                       shape_of(tensor2)),
                      out, self.device());
}

Tensor& addcdiv_inplace_kernel(Tensor& self, const Tensor& tensor1,
                               const Tensor& tensor2, const Scalar& /*value*/) {
    check_addcdiv_dtypes(tensor1, tensor2);
    if (broadcast_shapes(shape_of(self), shape_of(tensor1),
                         shape_of(tensor2)) != shape_of(self)) {
        TP_THROW(RuntimeError, "addcdiv_: output shape does not match self");
    }
    return self;
}
} // anonymous namespace
TENSORPLAY_LIBRARY_IMPL(Meta, MetaBlas) {
    m.impl("mm", mm_kernel);
    m.impl("matmul", matmul_kernel);
    m.impl("reshape", reshape_kernel);
    m.impl("bmm", bmm_kernel);
    m.impl("addmm", addmm_kernel);
    m.impl("addmm_", addmm_inplace_kernel);
    m.impl("dot", dot_kernel);
    m.impl("mv", mv_kernel);
    m.impl("inner", inner_kernel);
    m.impl("ger", ger_kernel);
    m.impl("baddbmm", baddbmm_kernel);
    m.impl("baddbmm_", baddbmm_inplace_kernel);
    m.impl("addbmm", addbmm_kernel);
    m.impl("addbmm_", addbmm_inplace_kernel);
    m.impl("addr", addr_kernel);
    m.impl("addr_", addr_inplace_kernel);
    m.impl("addmv", addmv_kernel);
    m.impl("addmv_", addmv_inplace_kernel);
    m.impl("addcmul", addcmul_kernel);
    m.impl("addcmul_", addcmul_inplace_kernel);
    m.impl("addcdiv", addcdiv_kernel);
    m.impl("addcdiv_", addcdiv_inplace_kernel);
}

} // namespace meta
} // namespace tensorplay
