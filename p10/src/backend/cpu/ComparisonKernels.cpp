#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "Exception.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace tensorplay {
namespace cpu {

// Helper for comparison ops
template<typename Op>
Tensor comparison_kernel_impl(const Tensor& self, const Tensor& other, Op op) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    
    // Result is always Bool
    Tensor result = Tensor::empty(out_shape, DType::Bool, self.device());
    
    // For comparison, we usually don't promote types to a common type for the operation, 
    // but C++ requires it. PyTorch promotes to common type before comparison.
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    
    Tensor self_casted = (self.dtype() == common_dtype) ? self : self.to(common_dtype);
    Tensor other_casted = (other.dtype() == common_dtype) ? other : other.to(common_dtype);

    Tensor self_expanded = self_casted.expand(out_shape);
    Tensor other_expanded = other_casted.expand(out_shape);
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        apply_op_recursive<bool, ctype>(result.data_ptr<bool>(), result.strides(), \
                                 self_expanded, self_expanded.strides(), \
                                 other_expanded, other_expanded.strides(), \
                                 0, 0, 0, 0, out_shape, op); \
        break; \
    }

    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "comparison: unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

Tensor eq_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a == b; });
}

Tensor ne_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a != b; });
}

Tensor lt_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a < b; });
}

Tensor le_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a <= b; });
}

Tensor gt_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a > b; });
}

Tensor ge_tensor_kernel(const Tensor& self, const Tensor& other) {
    return comparison_kernel_impl(self, other, [](auto a, auto b) { return a >= b; });
}

// Scalar versions
Tensor eq_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return eq_tensor_kernel(self, other_t);
}

Tensor ne_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return ne_tensor_kernel(self, other_t);
}

Tensor lt_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return lt_tensor_kernel(self, other_t);
}

Tensor le_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return le_tensor_kernel(self, other_t);
}

Tensor gt_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return gt_tensor_kernel(self, other_t);
}

Tensor ge_scalar_kernel(const Tensor& self, Scalar other) {
    Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
    return ge_tensor_kernel(self, other_t);
}

TENSORPLAY_LIBRARY_IMPL(CPU, ComparisonKernels) {
    m.impl("eq.Tensor", eq_tensor_kernel);
    m.impl("eq.Scalar", eq_scalar_kernel);
    m.impl("ne.Tensor", ne_tensor_kernel);
    m.impl("ne.Scalar", ne_scalar_kernel);
    m.impl("lt.Tensor", lt_tensor_kernel);
    m.impl("lt.Scalar", lt_scalar_kernel);
    m.impl("le.Tensor", le_tensor_kernel);
    m.impl("le.Scalar", le_scalar_kernel);
    m.impl("gt.Tensor", gt_tensor_kernel);
    m.impl("gt.Scalar", gt_scalar_kernel);
    m.impl("ge.Tensor", ge_tensor_kernel);
    m.impl("ge.Scalar", ge_scalar_kernel);
}

} // namespace cpu
} // namespace tensorplay
