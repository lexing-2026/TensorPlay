#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "Exception.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <type_traits>

namespace tensorplay {
namespace cpu {

// ATen alignment: wrapped Python scalars participate weakly in type promotion
// (result_type(int_tensor, 2.5) == Float32), and the scalar must never be
// truncated into the tensor's dtype before comparing.
static DType result_type_with_scalar(const Tensor& t, const Scalar& s) {
    DType td = t.dtype();
    if (s.dtype() == DType::Bool) return td;
    if (isFloatingType(s.dtype())) {
        if (isFloatingType(td)) return td;   // half/bf16 stay reduced
        return DType::Float32;
    }
    // int scalar: float tensors keep their dtype; integral tensors keep theirs
    return td;
}

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

// Scalar versions: promote the scalar with weak-scalar rules instead of
// casting it into self.dtype() (which truncated e.g. eq(2.5) on int tensors)
#define DEFINE_CMP_SCALAR_KERNEL(NAME) \
Tensor NAME##_scalar_kernel(const Tensor& self, Scalar other) { \
    DType common = result_type_with_scalar(self, other); \
    Tensor other_t = Tensor::full({}, other, common, self.device()); \
    return NAME##_tensor_kernel(self.to(common), other_t); \
}

DEFINE_CMP_SCALAR_KERNEL(eq)
DEFINE_CMP_SCALAR_KERNEL(ne)
DEFINE_CMP_SCALAR_KERNEL(lt)
DEFINE_CMP_SCALAR_KERNEL(le)
DEFINE_CMP_SCALAR_KERNEL(gt)
DEFINE_CMP_SCALAR_KERNEL(ge)
#undef DEFINE_CMP_SCALAR_KERNEL

template <typename Op>
Tensor where_kernel_impl(const Tensor& condition, const Tensor& self,
                         const Tensor& other, Op op) {
    if (condition.dtype() != DType::Bool) {
        TP_THROW(TypeError, "where condition must be a boolean tensor");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(condition.shape()),
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    Tensor self_casted = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor other_casted = other.dtype() == common_dtype ? other : other.to(common_dtype);
    auto condition_strides = broadcast_strides(condition, out_shape);
    auto self_strides = broadcast_strides(self_casted, out_shape);
    auto other_strides = broadcast_strides(other_casted, out_shape);

    #define WHERE_CASE(ctype, name) \
        case DType::name: { \
            apply_ternary_op_recursive_mixed<ctype, bool, ctype>( \
                result.data_ptr<ctype>(), result.strides(), condition, condition_strides, \
                self_casted, self_strides, other_casted, other_strides, \
                0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(WHERE_CASE)
        case DType::ComplexFloat: {
            apply_ternary_op_recursive_mixed<std::complex<float>, bool, std::complex<float>>( \
                result.data_ptr<std::complex<float>>(), result.strides(), condition, condition_strides, \
                self_casted, self_strides, other_casted, other_strides, \
                0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
        case DType::ComplexDouble: {
            apply_ternary_op_recursive_mixed<std::complex<double>, bool, std::complex<double>>( \
                result.data_ptr<std::complex<double>>(), result.strides(), condition, condition_strides, \
                self_casted, self_strides, other_casted, other_strides, \
                0, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
        default: TP_THROW(TypeError, "where: unsupported dtype");
    }
    #undef WHERE_CASE
    return result;
}

Tensor where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other) {
    return where_kernel_impl(condition, self, other,
        [](bool select_self, auto a, auto b) { return select_self ? a : b; });
}

Tensor where_scalar_self_cpu(const Tensor& condition, Scalar self, const Tensor& other) {
    DType common_dtype = result_type(self, other.dtype());
    Tensor self_tensor = Tensor::full({}, self, common_dtype, other.device());
    return where_cpu(condition, self_tensor, other);
}

Tensor where_scalar_other_cpu(const Tensor& condition, const Tensor& self, Scalar other) {
    DType common_dtype = result_type(other, self.dtype());
    Tensor other_tensor = Tensor::full({}, other, common_dtype, self.device());
    return where_cpu(condition, self, other_tensor);
}

static DType where_scalar_dtype(const Scalar& self, const Scalar& other) {
    if (self.isComplex() || other.isComplex()) {
        return promoteTypes(self.dtype(), other.dtype());
    }
    if (self.isFloatingPoint() || other.isFloatingPoint()) {
        return self.dtype() == DType::Float64 || other.dtype() == DType::Float64
            ? DType::Float64 : DType::Float32;
    }
    // Python integer scalars use the default integral result type in Torch.
    return DType::Int64;
}

Tensor where_scalar_scalar_cpu(const Tensor& condition, Scalar self, Scalar other) {
    DType common_dtype = where_scalar_dtype(self, other);
    Tensor self_tensor = Tensor::full({}, self, common_dtype, condition.device());
    Tensor other_tensor = Tensor::full({}, other, common_dtype, condition.device());
    return where_cpu(condition, self_tensor, other_tensor);
}

template <typename Op>
Tensor maximum_minimum_kernel_impl(const Tensor& self, const Tensor& other, Op op) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    if (isComplexType(common_dtype)) {
        TP_THROW(RuntimeError, "maximum/minimum is not implemented for complex tensors");
    }
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
    auto a_strides = broadcast_strides(a, out_shape);
    auto b_strides = broadcast_strides(b, out_shape);
    #define MAXMIN_CASE(ctype, name) \
        case DType::name: { \
            apply_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                a, a_strides, b, b_strides, 0, 0, 0, 0, out_shape, op); \
            break; \
        }
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(MAXMIN_CASE)
        default: TP_THROW(TypeError, "maximum/minimum: unsupported dtype");
    }
    #undef MAXMIN_CASE
    return result;
}

struct MaximumOp {
    template <typename T>
    T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(a)) return a;
            if (std::isnan(b)) return b;
        }
        return a < b ? b : a;
    }
};

struct MinimumOp {
    template <typename T>
    T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(a)) return a;
            if (std::isnan(b)) return b;
        }
        return a < b ? a : b;
    }
};

Tensor maximum_cpu(const Tensor& self, const Tensor& other) {
    return maximum_minimum_kernel_impl(self, other, MaximumOp());
}

Tensor minimum_cpu(const Tensor& self, const Tensor& other) {
    return maximum_minimum_kernel_impl(self, other, MinimumOp());
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
    m.impl("where.self", where_cpu);
    m.impl("where.ScalarSelf", where_scalar_self_cpu);
    m.impl("where.ScalarOther", where_scalar_other_cpu);
    m.impl("where.Scalar", where_scalar_scalar_cpu);
    m.impl("maximum", maximum_cpu);
    m.impl("minimum", minimum_cpu);
}

} // namespace cpu
} // namespace tensorplay
