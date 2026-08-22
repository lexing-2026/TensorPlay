#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cpu {
namespace {

std::vector<Scalar> unpack_packed_scalar_tensor(const Tensor& scalars,
                                                size_t expected_length) {
    if (scalars.device().type() != DeviceType::CPU) {
        TP_THROW(RuntimeError, "Expected scalars to be on CPU, got " +
            scalars.device().toString() + " instead.");
    }
    if (!scalars.is_contiguous()) {
        TP_THROW(RuntimeError, "Expected scalars to be contiguous.");
    }
    if (scalars.dim() != 1) {
        TP_THROW(RuntimeError,
            "Expected packed scalar Tensor to be of dimension 1. Got " +
            std::to_string(scalars.dim()) + " instead.");
    }
    if (scalars.size(0) != static_cast<int64_t>(expected_length)) {
        TP_THROW(RuntimeError,
            "Expected length of scalars to match input of length " +
            std::to_string(expected_length) + " but got " +
            std::to_string(scalars.size(0)) + " instead.");
    }
    std::vector<Scalar> result;
    result.reserve(expected_length);
    for (size_t i = 0; i < expected_length; ++i) {
        result.push_back(scalars.select(0, static_cast<int64_t>(i)).item());
    }
    return result;
}

template <typename F>
std::vector<Tensor> map_tensors(const std::vector<Tensor>& self, F&& fn) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (const Tensor& value : self) {
        result.push_back(fn(value));
    }
    return result;
}

template <typename F>
std::vector<Tensor> map_tensors_inplace(std::vector<Tensor> self, F&& fn) {
    for (Tensor& value : self) {
        fn(value);
    }
    return self;
}

template <typename F>
std::vector<Tensor> map_tensor_lists(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& other,
                                     F&& fn) {
    if (self.size() != other.size()) {
        TP_THROW(ValueError, "foreach tensor list arguments must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
        result.push_back(fn(self[i], other[i]));
    }
    return result;
}

template <typename F>
std::vector<Tensor> map_tensor_lists_inplace(std::vector<Tensor> self,
                                             const std::vector<Tensor>& other,
                                             F&& fn) {
    if (self.size() != other.size()) {
        TP_THROW(ValueError, "foreach tensor list arguments must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) {
        fn(self[i], other[i]);
    }
    return self;
}

template <typename F>
std::vector<Tensor> map_scalar_lists(const std::vector<Tensor>& self,
                                     const std::vector<Scalar>& scalars,
                                     F&& fn) {
    if (self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor and scalar lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
        result.push_back(fn(self[i], scalars[i]));
    }
    return result;
}

template <typename F>
std::vector<Tensor> map_scalar_lists_inplace(std::vector<Tensor> self,
                                             const std::vector<Scalar>& scalars,
                                             F&& fn) {
    if (self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor and scalar lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) {
        fn(self[i], scalars[i]);
    }
    return self;
}

template <typename F>
std::vector<Tensor> map_ternary_lists(const std::vector<Tensor>& self,
                                      const std::vector<Tensor>& tensor1,
                                      const std::vector<Tensor>& tensor2,
                                      F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size()) {
        TP_THROW(ValueError, "foreach ternary tensor lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
        result.push_back(fn(self[i], tensor1[i], tensor2[i]));
    }
    return result;
}

template <typename F>
std::vector<Tensor> map_ternary_lists_inplace(std::vector<Tensor> self,
                                              const std::vector<Tensor>& tensor1,
                                              const std::vector<Tensor>& tensor2,
                                              F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size()) {
        TP_THROW(ValueError, "foreach ternary tensor lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) {
        fn(self[i], tensor1[i], tensor2[i]);
    }
    return self;
}

template <typename F>
std::vector<Tensor> map_ternary_scalar_lists(const std::vector<Tensor>& self,
                                             const std::vector<Tensor>& tensor1,
                                             const std::vector<Tensor>& tensor2,
                                             const std::vector<Scalar>& scalars,
                                             F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size() ||
        self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach ternary tensor/scalar lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
        result.push_back(fn(self[i], tensor1[i], tensor2[i], scalars[i]));
    }
    return result;
}

template <typename F>
std::vector<Tensor> map_ternary_scalar_lists_inplace(std::vector<Tensor> self,
                                                     const std::vector<Tensor>& tensor1,
                                                     const std::vector<Tensor>& tensor2,
                                                     const std::vector<Scalar>& scalars,
                                                     F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size() ||
        self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach ternary tensor/scalar lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) {
        fn(self[i], tensor1[i], tensor2[i], scalars[i]);
    }
    return self;
}

template <typename F>
std::vector<Tensor> map_pair_scalar_lists(const std::vector<Tensor>& self,
                                          const std::vector<Tensor>& other,
                                          const std::vector<Scalar>& scalars,
                                          F&& fn) {
    if (self.size() != other.size() || self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor/scalar lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) {
        result.push_back(fn(self[i], other[i], scalars[i]));
    }
    return result;
}

template <typename F>
std::vector<Tensor> map_pair_scalar_lists_inplace(std::vector<Tensor> self,
                                                  const std::vector<Tensor>& other,
                                                  const std::vector<Scalar>& scalars,
                                                  F&& fn) {
    if (self.size() != other.size() || self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor/scalar lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) {
        fn(self[i], other[i], scalars[i]);
    }
    return self;
}

std::vector<Tensor> foreach_add_scalar_cpu(const std::vector<Tensor>& self, Scalar scalar) {
    return map_tensors(self, [&](const Tensor& value) { return value.add(scalar); });
}
std::vector<Tensor> foreach_add_list_cpu(const std::vector<Tensor>& self, const std::vector<Tensor>& other, Scalar alpha) {
    return map_tensor_lists(self, other, [&](const Tensor& value, const Tensor& rhs) { return value.add(rhs, alpha); });
}
std::vector<Tensor> foreach_add_scalar_list_cpu(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return map_scalar_lists(self, scalars, [&](const Tensor& value, Scalar scalar) { return value.add(scalar); });
}
std::vector<Tensor> foreach_add_tensor_cpu(const std::vector<Tensor>& self, const Tensor& other, Scalar alpha) {
    return map_tensors(self, [&](const Tensor& value) { return value.add(other, alpha); });
}
void foreach_add_scalar_inplace_cpu(std::vector<Tensor> self, Scalar scalar) {
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.add_(scalar); });
}
void foreach_add_list_inplace_cpu(std::vector<Tensor> self, const std::vector<Tensor>& other, Scalar alpha) {
    map_tensor_lists_inplace(std::move(self), other, [&](Tensor& value, const Tensor& rhs) { value.add_(rhs, alpha); });
}
void foreach_add_scalar_list_inplace_cpu(std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    map_scalar_lists_inplace(std::move(self), scalars, [&](Tensor& value, Scalar scalar) { value.add_(scalar); });
}
void foreach_add_tensor_inplace_cpu(std::vector<Tensor> self, const Tensor& other, Scalar alpha) {
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.add_(other, alpha); });
}

#define DEFINE_FOREACH_OP(NAME, METHOD) \
std::vector<Tensor> foreach_##NAME##_scalar_cpu(const std::vector<Tensor>& self, Scalar scalar) { \
    return map_tensors(self, [&](const Tensor& value) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_list_cpu(const std::vector<Tensor>& self, const std::vector<Tensor>& other) { \
    return map_tensor_lists(self, other, [&](const Tensor& value, const Tensor& rhs) { return value.METHOD(rhs); }); \
} \
std::vector<Tensor> foreach_##NAME##_scalar_list_cpu(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) { \
    return map_scalar_lists(self, scalars, [&](const Tensor& value, Scalar scalar) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_tensor_cpu(const std::vector<Tensor>& self, const Tensor& other) { \
    return map_tensors(self, [&](const Tensor& value) { return value.METHOD(other); }); \
} \
void foreach_##NAME##_tensor_inplace_cpu(std::vector<Tensor> self, const Tensor& other) { \
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.copy_(value.METHOD(other)); }); \
} \
void foreach_##NAME##_scalar_inplace_cpu(std::vector<Tensor> self, Scalar scalar) { \
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.copy_(value.METHOD(scalar)); }); \
} \
void foreach_##NAME##_list_inplace_cpu(std::vector<Tensor> self, const std::vector<Tensor>& other) { \
    map_tensor_lists_inplace(std::move(self), other, [&](Tensor& value, const Tensor& rhs) { value.copy_(value.METHOD(rhs)); }); \
} \
void foreach_##NAME##_scalar_list_inplace_cpu(std::vector<Tensor> self, const std::vector<Scalar>& scalars) { \
    map_scalar_lists_inplace(std::move(self), scalars, [&](Tensor& value, Scalar scalar) { value.copy_(value.METHOD(scalar)); }); \
}

DEFINE_FOREACH_OP(mul, mul)
DEFINE_FOREACH_OP(div, div)
#undef DEFINE_FOREACH_OP

// sub's out variants call the base kernels with an alpha (mirroring the
// hand-written add kernels above), so spell them out instead of using
// DEFINE_FOREACH_OP.
std::vector<Tensor> foreach_sub_scalar_cpu(const std::vector<Tensor>& self, Scalar scalar) {
    return map_tensors(self, [&](const Tensor& value) { return value.sub(scalar); });
}
std::vector<Tensor> foreach_sub_list_cpu(const std::vector<Tensor>& self, const std::vector<Tensor>& other, Scalar alpha) {
    return map_tensor_lists(self, other, [&](const Tensor& value, const Tensor& rhs) { return value.sub(rhs, alpha); });
}
std::vector<Tensor> foreach_sub_scalar_list_cpu(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return map_scalar_lists(self, scalars, [&](const Tensor& value, Scalar scalar) { return value.sub(scalar); });
}
std::vector<Tensor> foreach_sub_tensor_cpu(const std::vector<Tensor>& self, const Tensor& other, Scalar alpha) {
    return map_tensors(self, [&](const Tensor& value) { return value.sub(other, alpha); });
}
void foreach_sub_tensor_inplace_cpu(std::vector<Tensor> self, const Tensor& other) {
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.copy_(value.sub(other)); });
}
void foreach_sub_scalar_inplace_cpu(std::vector<Tensor> self, Scalar scalar) {
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.copy_(value.sub(scalar)); });
}
void foreach_sub_list_inplace_cpu(std::vector<Tensor> self, const std::vector<Tensor>& other) {
    map_tensor_lists_inplace(std::move(self), other, [&](Tensor& value, const Tensor& rhs) { value.copy_(value.sub(rhs)); });
}
void foreach_sub_scalar_list_inplace_cpu(std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    map_scalar_lists_inplace(std::move(self), scalars, [&](Tensor& value, Scalar scalar) { value.copy_(value.sub(scalar)); });
}

#define DEFINE_FOREACH_UNARY(NAME, METHOD) \
std::vector<Tensor> foreach_##NAME##_cpu(const std::vector<Tensor>& self) { \
    return map_tensors(self, [&](const Tensor& value) { return value.METHOD(); }); \
} \
void foreach_##NAME##_inplace_cpu(std::vector<Tensor> self) { \
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.copy_(value.METHOD()); }); \
}

DEFINE_FOREACH_UNARY(sqrt, sqrt)
DEFINE_FOREACH_UNARY(rsqrt, rsqrt)
DEFINE_FOREACH_UNARY(neg, neg)
DEFINE_FOREACH_UNARY(abs, abs)
DEFINE_FOREACH_UNARY(sign, sign)
DEFINE_FOREACH_UNARY(acos, acos)
DEFINE_FOREACH_UNARY(asin, asin)
DEFINE_FOREACH_UNARY(atan, atan)
DEFINE_FOREACH_UNARY(ceil, ceil)
DEFINE_FOREACH_UNARY(cos, cos)
DEFINE_FOREACH_UNARY(cosh, cosh)
DEFINE_FOREACH_UNARY(erf, erf)
DEFINE_FOREACH_UNARY(erfc, erfc)
DEFINE_FOREACH_UNARY(exp, exp)
DEFINE_FOREACH_UNARY(expm1, expm1)
DEFINE_FOREACH_UNARY(floor, floor)
DEFINE_FOREACH_UNARY(frac, frac)
DEFINE_FOREACH_UNARY(lgamma, lgamma)
DEFINE_FOREACH_UNARY(log, log)
DEFINE_FOREACH_UNARY(log10, log10)
DEFINE_FOREACH_UNARY(log1p, log1p)
DEFINE_FOREACH_UNARY(log2, log2)
DEFINE_FOREACH_UNARY(round, round)
DEFINE_FOREACH_UNARY(sigmoid, sigmoid)
DEFINE_FOREACH_UNARY(sin, sin)
DEFINE_FOREACH_UNARY(sinh, sinh)
DEFINE_FOREACH_UNARY(tan, tan)
DEFINE_FOREACH_UNARY(tanh, tanh)
DEFINE_FOREACH_UNARY(trunc, trunc)
#undef DEFINE_FOREACH_UNARY

std::vector<Tensor> foreach_max_cpu(const std::vector<Tensor>& self) {
    return map_tensors(self, [](const Tensor& value) { return value.max(); });
}

std::vector<Tensor> foreach_norm_cpu(const std::vector<Tensor>& self,
                                     Scalar ord,
                                     const std::optional<DType>& dtype) {
    return map_tensors(self, [&](const Tensor& value) {
        Tensor input = dtype.has_value() ? value.to(*dtype) : value;
        return input.norm(ord.toDouble());
    });
}

std::vector<Tensor> foreach_powsum_cpu(const std::vector<Tensor>& self,
                                       Scalar ord,
                                       const std::optional<DType>& dtype) {
    return map_tensors(self, [&](const Tensor& value) {
        Tensor input = dtype.has_value() ? value.to(*dtype) : value;
        return input.abs().pow(ord).sum();
    });
}

std::vector<Tensor> foreach_clone_cpu(
        const std::vector<Tensor>& self,
        const std::optional<int64_t>& /*memory_format*/) {
    return map_tensors(self, [](const Tensor& value) { return value.clone(); });
}

std::vector<Tensor> foreach_copy_cpu(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& src,
                                     bool /*non_blocking*/) {
    return map_tensor_lists(self, src,
        [](const Tensor& /*destination*/, const Tensor& source) {
            return source.clone();
        });
}

std::vector<Tensor> foreach_mm_cpu(const std::vector<Tensor>& self,
                                   const std::vector<Tensor>& mat2) {
    return map_tensor_lists(self, mat2,
        [](const Tensor& lhs, const Tensor& rhs) { return lhs.mm(rhs); });
}

std::vector<Tensor> foreach_reciprocal_cpu(const std::vector<Tensor>& self) {
    return map_tensors(self, [&](const Tensor& value) {
        return value.pow(Scalar(-1));
    });
}
void foreach_reciprocal_inplace_cpu(std::vector<Tensor> self) {
    map_tensors_inplace(std::move(self), [&](Tensor& value) {
        value.copy_(value.pow(Scalar(-1)));
    });
}

std::vector<Tensor> foreach_addcmul_scalar_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, Scalar value) {
    return map_ternary_lists(self, tensor1, tensor2,
        [&](const Tensor& x, const Tensor& a, const Tensor& b) {
            return x.addcmul(a, b, value);
        });
}
void foreach_addcmul_scalar_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, Scalar value) {
    map_ternary_lists_inplace(std::move(self), tensor1, tensor2,
        [&](Tensor& x, const Tensor& a, const Tensor& b) { x.addcmul_(a, b, value); });
}
std::vector<Tensor> foreach_addcdiv_scalar_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, Scalar value) {
    return map_ternary_lists(self, tensor1, tensor2,
        [&](const Tensor& x, const Tensor& a, const Tensor& b) {
            return x.addcdiv(a, b, value);
        });
}
void foreach_addcdiv_scalar_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, Scalar value) {
    map_ternary_lists_inplace(std::move(self), tensor1, tensor2,
        [&](Tensor& x, const Tensor& a, const Tensor& b) { x.addcdiv_(a, b, value); });
}

std::vector<Tensor> foreach_addcmul_scalar_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    return map_ternary_scalar_lists(self, tensor1, tensor2, scalars,
        [&](const Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            return x.addcmul(a, b, value);
        });
}
void foreach_addcmul_scalar_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    map_ternary_scalar_lists_inplace(std::move(self), tensor1, tensor2, scalars,
        [&](Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            x.addcmul_(a, b, value);
        });
}
std::vector<Tensor> foreach_addcmul_tensor_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    const auto scalar_values = unpack_packed_scalar_tensor(scalars, self.size());
    return map_ternary_scalar_lists(self, tensor1, tensor2, scalar_values,
        [&](const Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            return x.addcmul(a, b, value);
        });
}
void foreach_addcmul_tensor_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    const auto scalar_values = unpack_packed_scalar_tensor(scalars, self.size());
    map_ternary_scalar_lists_inplace(std::move(self), tensor1, tensor2,
        scalar_values,
        [&](Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            x.addcmul_(a, b, value);
        });
}

std::vector<Tensor> foreach_addcdiv_scalar_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    return map_ternary_scalar_lists(self, tensor1, tensor2, scalars,
        [&](const Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            return x.addcdiv(a, b, value);
        });
}
void foreach_addcdiv_scalar_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    map_ternary_scalar_lists_inplace(std::move(self), tensor1, tensor2, scalars,
        [&](Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            x.addcdiv_(a, b, value);
        });
}
std::vector<Tensor> foreach_addcdiv_tensor_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    const auto scalar_values = unpack_packed_scalar_tensor(scalars, self.size());
    return map_ternary_scalar_lists(self, tensor1, tensor2, scalar_values,
        [&](const Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            return x.addcdiv(a, b, value);
        });
}
void foreach_addcdiv_tensor_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    const auto scalar_values = unpack_packed_scalar_tensor(scalars, self.size());
    map_ternary_scalar_lists_inplace(std::move(self), tensor1, tensor2,
        scalar_values,
        [&](Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            x.addcdiv_(a, b, value);
        });
}

std::vector<Tensor> foreach_lerp_scalar_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end, Scalar weight) {
    return map_tensor_lists(self, end,
        [&](const Tensor& x, const Tensor& y) { return x.lerp(y, weight); });
}
std::vector<Tensor> foreach_lerp_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        const std::vector<Tensor>& weight) {
    if (self.size() != end.size() || self.size() != weight.size()) {
        TP_THROW(ValueError, "foreach lerp lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) result.push_back(self[i].lerp(end[i], weight[i]));
    return result;
}
void foreach_lerp_scalar_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& end, Scalar weight) {
    map_tensor_lists_inplace(std::move(self), end,
        [&](Tensor& x, const Tensor& y) { x.copy_(x.lerp(y, weight)); });
}
void foreach_lerp_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& end,
        const std::vector<Tensor>& weight) {
    if (self.size() != end.size() || self.size() != weight.size()) {
        TP_THROW(ValueError, "foreach lerp lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) self[i].copy_(self[i].lerp(end[i], weight[i]));
}
std::vector<Tensor> foreach_lerp_scalar_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weight) {
    return map_pair_scalar_lists(self, end, weight,
        [&](const Tensor& x, const Tensor& y, Scalar w) { return x.lerp(y, w); });
}
void foreach_lerp_scalar_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weight) {
    map_pair_scalar_lists_inplace(std::move(self), end, weight,
        [&](Tensor& x, const Tensor& y, Scalar w) { x.copy_(x.lerp(y, w)); });
}

std::vector<Tensor> foreach_pow_scalar_cpu(const std::vector<Tensor>& self, Scalar exponent) {
    return map_tensors(self, [&](const Tensor& value) { return value.pow(exponent); });
}
std::vector<Tensor> foreach_pow_scalar_tensor_cpu(
        Scalar self, const std::vector<Tensor>& exponent) {
    return map_tensors(exponent, [&](const Tensor& value) {
        Tensor base = Tensor::full({}, self, value.dtype(), value.device());
        return base.pow(value);
    });
}
std::vector<Tensor> foreach_pow_tensor_tensor_cpu(
        const Tensor& self, const std::vector<Tensor>& exponent) {
    return map_tensors(exponent, [&](const Tensor& value) {
        return self.pow(value);
    });
}
std::vector<Tensor> foreach_pow_list_cpu(const std::vector<Tensor>& self, const std::vector<Tensor>& exponent) {
    return map_tensor_lists(self, exponent, [&](const Tensor& value, const Tensor& rhs) { return value.pow(rhs); });
}
void foreach_pow_scalar_inplace_cpu(std::vector<Tensor> self, Scalar exponent) {
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.copy_(value.pow(exponent)); });
}
void foreach_pow_list_inplace_cpu(std::vector<Tensor> self, const std::vector<Tensor>& exponent) {
    map_tensor_lists_inplace(std::move(self), exponent, [&](Tensor& value, const Tensor& rhs) { value.copy_(value.pow(rhs)); });
}
std::vector<Tensor> foreach_pow_scalar_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Scalar>& exponent) {
    return map_scalar_lists(self, exponent,
        [&](const Tensor& value, Scalar rhs) { return value.pow(rhs); });
}
void foreach_pow_scalar_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Scalar>& exponent) {
    map_scalar_lists_inplace(std::move(self), exponent,
        [&](Tensor& value, Scalar rhs) { value.copy_(value.pow(rhs)); });
}

std::vector<Tensor> foreach_clamp_min_scalar_cpu(const std::vector<Tensor>& self, Scalar scalar) {
    return map_tensors(self, [&](const Tensor& value) { return value.clamp(scalar, std::nullopt); });
}
std::vector<Tensor> foreach_clamp_max_scalar_cpu(const std::vector<Tensor>& self, Scalar scalar) {
    return map_tensors(self, [&](const Tensor& value) { return value.clamp(std::nullopt, scalar); });
}
void foreach_clamp_min_scalar_inplace_cpu(std::vector<Tensor> self, Scalar scalar) {
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.copy_(value.clamp(scalar, std::nullopt)); });
}
void foreach_clamp_max_scalar_inplace_cpu(std::vector<Tensor> self, Scalar scalar) {
    map_tensors_inplace(std::move(self), [&](Tensor& value) { value.copy_(value.clamp(std::nullopt, scalar)); });
}

std::vector<Tensor> foreach_clamp_min_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return map_tensor_lists(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::maximum(value, rhs); });
}
void foreach_clamp_min_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    map_tensor_lists_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::maximum(value, rhs)); });
}
std::vector<Tensor> foreach_clamp_max_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return map_tensor_lists(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::minimum(value, rhs); });
}
void foreach_clamp_max_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    map_tensor_lists_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::minimum(value, rhs)); });
}
std::vector<Tensor> foreach_clamp_min_scalar_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return map_scalar_lists(self, scalars,
        [&](const Tensor& value, Scalar rhs) { return value.clamp(rhs, std::nullopt); });
}
void foreach_clamp_min_scalar_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    map_scalar_lists_inplace(std::move(self), scalars,
        [&](Tensor& value, Scalar rhs) { value.copy_(value.clamp(rhs, std::nullopt)); });
}
std::vector<Tensor> foreach_clamp_max_scalar_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return map_scalar_lists(self, scalars,
        [&](const Tensor& value, Scalar rhs) { return value.clamp(std::nullopt, rhs); });
}
void foreach_clamp_max_scalar_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    map_scalar_lists_inplace(std::move(self), scalars,
        [&](Tensor& value, Scalar rhs) { value.copy_(value.clamp(std::nullopt, rhs)); });
}

std::vector<Tensor> foreach_maximum_scalar_cpu(const std::vector<Tensor>& self, Scalar scalar) {
    return foreach_clamp_min_scalar_cpu(self, scalar);
}
std::vector<Tensor> foreach_minimum_scalar_cpu(const std::vector<Tensor>& self, Scalar scalar) {
    return foreach_clamp_max_scalar_cpu(self, scalar);
}
void foreach_maximum_scalar_inplace_cpu(std::vector<Tensor> self, Scalar scalar) {
    foreach_clamp_min_scalar_inplace_cpu(std::move(self), scalar);
}
void foreach_minimum_scalar_inplace_cpu(std::vector<Tensor> self, Scalar scalar) {
    foreach_clamp_max_scalar_inplace_cpu(std::move(self), scalar);
}

std::vector<Tensor> foreach_maximum_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return map_tensor_lists(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::maximum(value, rhs); });
}
void foreach_maximum_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    map_tensor_lists_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::maximum(value, rhs)); });
}
std::vector<Tensor> foreach_maximum_scalar_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_clamp_min_scalar_list_cpu(self, scalars);
}
void foreach_maximum_scalar_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_clamp_min_scalar_list_inplace_cpu(std::move(self), scalars);
}
std::vector<Tensor> foreach_minimum_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return map_tensor_lists(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::minimum(value, rhs); });
}
void foreach_minimum_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    map_tensor_lists_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::minimum(value, rhs)); });
}
std::vector<Tensor> foreach_minimum_scalar_list_cpu(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_clamp_max_scalar_list_cpu(self, scalars);
}
void foreach_minimum_scalar_list_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_clamp_max_scalar_list_inplace_cpu(std::move(self), scalars);
}

void foreach_copy_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& src, bool non_blocking) {
    map_tensor_lists_inplace(std::move(self), src,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(rhs, non_blocking); });
}
std::vector<Tensor> foreach_zero_cpu(const std::vector<Tensor>& self) {
    return map_tensors(self, [](const Tensor& value) {
        return value.clone().zero_();
    });
}
void foreach_zero_inplace_cpu(std::vector<Tensor> self) {
    for (Tensor& value : self) value.zero_();
}

// The ATen out= overloads write the complete result list into caller-owned
// tensors.  Compute the list first so aliasing between `out` and an input has
// the same non-destructive behavior as the native foreach kernels, then copy
// each result into its corresponding output handle.
void copy_foreach_out_cpu(std::vector<Tensor> result,
                          std::vector<Tensor> out,
                          const char* op_name) {
    if (result.size() != out.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": output list must have the same length as the input list");
    }
    for (size_t i = 0; i < result.size(); ++i) {
        out[i].copy_(result[i]);
    }
}

#define DEFINE_FOREACH_UNARY_OUT(NAME) \
void foreach_##NAME##_out_cpu(const std::vector<Tensor>& self, \
                              std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_cpu(self), std::move(out), \
                         "_foreach_" #NAME ".out"); \
}
DEFINE_FOREACH_UNARY_OUT(sqrt)
DEFINE_FOREACH_UNARY_OUT(rsqrt)
DEFINE_FOREACH_UNARY_OUT(neg)
DEFINE_FOREACH_UNARY_OUT(abs)
DEFINE_FOREACH_UNARY_OUT(reciprocal)
DEFINE_FOREACH_UNARY_OUT(sign)
DEFINE_FOREACH_UNARY_OUT(acos)
DEFINE_FOREACH_UNARY_OUT(asin)
DEFINE_FOREACH_UNARY_OUT(atan)
DEFINE_FOREACH_UNARY_OUT(ceil)
DEFINE_FOREACH_UNARY_OUT(cos)
DEFINE_FOREACH_UNARY_OUT(cosh)
DEFINE_FOREACH_UNARY_OUT(erf)
DEFINE_FOREACH_UNARY_OUT(erfc)
DEFINE_FOREACH_UNARY_OUT(exp)
DEFINE_FOREACH_UNARY_OUT(expm1)
DEFINE_FOREACH_UNARY_OUT(floor)
DEFINE_FOREACH_UNARY_OUT(frac)
DEFINE_FOREACH_UNARY_OUT(lgamma)
DEFINE_FOREACH_UNARY_OUT(log)
DEFINE_FOREACH_UNARY_OUT(log10)
DEFINE_FOREACH_UNARY_OUT(log1p)
DEFINE_FOREACH_UNARY_OUT(log2)
DEFINE_FOREACH_UNARY_OUT(round)
DEFINE_FOREACH_UNARY_OUT(sigmoid)
DEFINE_FOREACH_UNARY_OUT(sin)
DEFINE_FOREACH_UNARY_OUT(sinh)
DEFINE_FOREACH_UNARY_OUT(tan)
DEFINE_FOREACH_UNARY_OUT(tanh)
DEFINE_FOREACH_UNARY_OUT(trunc)
#undef DEFINE_FOREACH_UNARY_OUT

#define DEFINE_FOREACH_ADD_SUB_OUT(NAME) \
void foreach_##NAME##_scalar_out_cpu(const std::vector<Tensor>& self, Scalar scalar, \
                                     std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_scalar_cpu(self, scalar), std::move(out), \
                         "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_list_out_cpu(const std::vector<Tensor>& self, \
                                   const std::vector<Tensor>& other, Scalar alpha, \
                                   std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_list_cpu(self, other, alpha), std::move(out), \
                         "_foreach_" #NAME ".List_out"); \
} \
void foreach_##NAME##_scalar_list_out_cpu(const std::vector<Tensor>& self, \
                                          const std::vector<Scalar>& scalars, \
                                          std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_scalar_list_cpu(self, scalars), std::move(out), \
                         "_foreach_" #NAME ".ScalarList_out"); \
} \
void foreach_##NAME##_tensor_out_cpu(const std::vector<Tensor>& self, const Tensor& other, \
                                     Scalar alpha, std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_tensor_cpu(self, other, alpha), std::move(out), \
                         "_foreach_" #NAME ".Tensor_out"); \
}
DEFINE_FOREACH_ADD_SUB_OUT(add)
DEFINE_FOREACH_ADD_SUB_OUT(sub)
#undef DEFINE_FOREACH_ADD_SUB_OUT

#define DEFINE_FOREACH_MUL_DIV_OUT(NAME) \
void foreach_##NAME##_scalar_out_cpu(const std::vector<Tensor>& self, Scalar scalar, \
                                     std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_scalar_cpu(self, scalar), std::move(out), \
                         "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_list_out_cpu(const std::vector<Tensor>& self, \
                                   const std::vector<Tensor>& other, \
                                   std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_list_cpu(self, other), std::move(out), \
                         "_foreach_" #NAME ".List_out"); \
} \
void foreach_##NAME##_scalar_list_out_cpu(const std::vector<Tensor>& self, \
                                          const std::vector<Scalar>& scalars, \
                                          std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_scalar_list_cpu(self, scalars), std::move(out), \
                         "_foreach_" #NAME ".ScalarList_out"); \
} \
void foreach_##NAME##_tensor_out_cpu(const std::vector<Tensor>& self, const Tensor& other, \
                                     std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_tensor_cpu(self, other), std::move(out), \
                         "_foreach_" #NAME ".Tensor_out"); \
}
DEFINE_FOREACH_MUL_DIV_OUT(mul)
DEFINE_FOREACH_MUL_DIV_OUT(div)
#undef DEFINE_FOREACH_MUL_DIV_OUT

#define DEFINE_FOREACH_TERNARY_OUT(NAME) \
void foreach_##NAME##_scalar_out_cpu(const std::vector<Tensor>& self, \
                                     const std::vector<Tensor>& tensor1, \
                                     const std::vector<Tensor>& tensor2, Scalar value, \
                                     std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_scalar_cpu(self, tensor1, tensor2, value), \
                         std::move(out), "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_scalar_list_out_cpu(const std::vector<Tensor>& self, \
                                          const std::vector<Tensor>& tensor1, \
                                          const std::vector<Tensor>& tensor2, \
                                          const std::vector<Scalar>& scalars, \
                                          std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_scalar_list_cpu(self, tensor1, tensor2, scalars), \
                         std::move(out), "_foreach_" #NAME ".ScalarList_out"); \
} \
void foreach_##NAME##_tensor_out_cpu(const std::vector<Tensor>& self, \
                                     const std::vector<Tensor>& tensor1, \
                                     const std::vector<Tensor>& tensor2, const Tensor& scalars, \
                                     std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_tensor_cpu(self, tensor1, tensor2, scalars), \
                         std::move(out), "_foreach_" #NAME ".Tensor_out"); \
}
DEFINE_FOREACH_TERNARY_OUT(addcmul)
DEFINE_FOREACH_TERNARY_OUT(addcdiv)
#undef DEFINE_FOREACH_TERNARY_OUT

void foreach_lerp_scalar_out_cpu(const std::vector<Tensor>& self,
                                  const std::vector<Tensor>& end,
                                  Scalar weight, std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_lerp_scalar_cpu(self, end, weight), std::move(out),
                         "_foreach_lerp.Scalar_out");
}
void foreach_lerp_list_out_cpu(const std::vector<Tensor>& self,
                               const std::vector<Tensor>& end,
                               const std::vector<Tensor>& weight,
                               std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_lerp_list_cpu(self, end, weight), std::move(out),
                         "_foreach_lerp.List_out");
}
void foreach_lerp_scalar_list_out_cpu(const std::vector<Tensor>& self,
                                      const std::vector<Tensor>& end,
                                      const std::vector<Scalar>& weight,
                                      std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_lerp_scalar_list_cpu(self, end, weight), std::move(out),
                         "_foreach_lerp.ScalarList_out");
}

#define DEFINE_FOREACH_CLAMP_OUT(NAME) \
void foreach_##NAME##_scalar_out_cpu(const std::vector<Tensor>& self, Scalar scalar, \
                                     std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_scalar_cpu(self, scalar), std::move(out), \
                         "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_list_out_cpu(const std::vector<Tensor>& self, \
                                   const std::vector<Tensor>& other, \
                                   std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_list_cpu(self, other), std::move(out), \
                         "_foreach_" #NAME ".List_out"); \
} \
void foreach_##NAME##_scalar_list_out_cpu(const std::vector<Tensor>& self, \
                                          const std::vector<Scalar>& scalars, \
                                          std::vector<Tensor> out) { \
    copy_foreach_out_cpu(foreach_##NAME##_scalar_list_cpu(self, scalars), std::move(out), \
                         "_foreach_" #NAME ".ScalarList_out"); \
}
DEFINE_FOREACH_CLAMP_OUT(clamp_min)
DEFINE_FOREACH_CLAMP_OUT(clamp_max)
DEFINE_FOREACH_CLAMP_OUT(maximum)
DEFINE_FOREACH_CLAMP_OUT(minimum)
#undef DEFINE_FOREACH_CLAMP_OUT

void foreach_clone_out_cpu(const std::vector<Tensor>& self,
                           const std::optional<int64_t>& memory_format,
                           std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_clone_cpu(self, memory_format), std::move(out),
                         "_foreach_clone.out");
}
void foreach_copy_out_cpu(const std::vector<Tensor>& self,
                          const std::vector<Tensor>& src,
                          bool non_blocking, std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_copy_cpu(self, src, non_blocking), std::move(out),
                         "_foreach_copy.out");
}
void foreach_max_out_cpu(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_max_cpu(self), std::move(out), "_foreach_max.out");
}
void foreach_norm_out_cpu(const std::vector<Tensor>& self, Scalar ord,
                          const std::optional<DType>& dtype,
                          std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_norm_cpu(self, ord, dtype), std::move(out),
                         "_foreach_norm.Scalar_out");
}
void foreach_pow_scalar_out_cpu(const std::vector<Tensor>& self, Scalar exponent,
                                std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_pow_scalar_cpu(self, exponent), std::move(out),
                         "_foreach_pow.Scalar_out");
}
void foreach_pow_list_out_cpu(const std::vector<Tensor>& self,
                              const std::vector<Tensor>& exponent,
                              std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_pow_list_cpu(self, exponent), std::move(out),
                         "_foreach_pow.List_out");
}
void foreach_pow_scalar_list_out_cpu(const std::vector<Tensor>& self,
                                     const std::vector<Scalar>& exponent,
                                     std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_pow_scalar_list_cpu(self, exponent), std::move(out),
                         "_foreach_pow.ScalarList_out");
}
void foreach_powsum_out_cpu(const std::vector<Tensor>& self, Scalar ord,
                            const std::optional<DType>& dtype,
                            std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_powsum_cpu(self, ord, dtype), std::move(out),
                         "_foreach_powsum.Scalar_out");
}
void foreach_zero_out_cpu(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    copy_foreach_out_cpu(foreach_zero_cpu(self), std::move(out), "_foreach_zero.out");
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CPU, ForeachKernels) {
    m.impl("_foreach_add.Scalar", foreach_add_scalar_cpu);
    m.impl("_foreach_add.List", foreach_add_list_cpu);
    m.impl("_foreach_add.ScalarList", foreach_add_scalar_list_cpu);
    m.impl("_foreach_add.Tensor", foreach_add_tensor_cpu);
    m.impl("_foreach_add_.Scalar", foreach_add_scalar_inplace_cpu);
    m.impl("_foreach_add_.List", foreach_add_list_inplace_cpu);
    m.impl("_foreach_add_.ScalarList", foreach_add_scalar_list_inplace_cpu);
    m.impl("_foreach_add_.Tensor", foreach_add_tensor_inplace_cpu);

#define REGISTER_FOREACH_BINARY(NAME) \
    m.impl("_foreach_" #NAME ".Scalar", foreach_##NAME##_scalar_cpu); \
    m.impl("_foreach_" #NAME ".List", foreach_##NAME##_list_cpu); \
    m.impl("_foreach_" #NAME ".ScalarList", foreach_##NAME##_scalar_list_cpu); \
    m.impl("_foreach_" #NAME ".Tensor", foreach_##NAME##_tensor_cpu); \
    m.impl("_foreach_" #NAME "_.Scalar", foreach_##NAME##_scalar_inplace_cpu); \
    m.impl("_foreach_" #NAME "_.List", foreach_##NAME##_list_inplace_cpu); \
    m.impl("_foreach_" #NAME "_.ScalarList", foreach_##NAME##_scalar_list_inplace_cpu); \
    m.impl("_foreach_" #NAME "_.Tensor", foreach_##NAME##_tensor_inplace_cpu);
    REGISTER_FOREACH_BINARY(sub)
    REGISTER_FOREACH_BINARY(mul)
    REGISTER_FOREACH_BINARY(div)
#undef REGISTER_FOREACH_BINARY

    m.impl("_foreach_add.List_out", foreach_add_list_out_cpu);
    m.impl("_foreach_add.ScalarList_out", foreach_add_scalar_list_out_cpu);
    m.impl("_foreach_add.Scalar_out", foreach_add_scalar_out_cpu);
    m.impl("_foreach_add.Tensor_out", foreach_add_tensor_out_cpu);
    m.impl("_foreach_sub.List_out", foreach_sub_list_out_cpu);
    m.impl("_foreach_sub.ScalarList_out", foreach_sub_scalar_list_out_cpu);
    m.impl("_foreach_sub.Scalar_out", foreach_sub_scalar_out_cpu);
    m.impl("_foreach_mul.List_out", foreach_mul_list_out_cpu);
    m.impl("_foreach_mul.ScalarList_out", foreach_mul_scalar_list_out_cpu);
    m.impl("_foreach_mul.Scalar_out", foreach_mul_scalar_out_cpu);
    m.impl("_foreach_mul.Tensor_out", foreach_mul_tensor_out_cpu);
    m.impl("_foreach_div.List_out", foreach_div_list_out_cpu);
    m.impl("_foreach_div.ScalarList_out", foreach_div_scalar_list_out_cpu);
    m.impl("_foreach_div.Scalar_out", foreach_div_scalar_out_cpu);
    m.impl("_foreach_div.Tensor_out", foreach_div_tensor_out_cpu);

#define REGISTER_FOREACH_UNARY(NAME) \
    m.impl("_foreach_" #NAME, foreach_##NAME##_cpu); \
    m.impl("_foreach_" #NAME "_", foreach_##NAME##_inplace_cpu); \
    m.impl("_foreach_" #NAME ".out", foreach_##NAME##_out_cpu);
    REGISTER_FOREACH_UNARY(sqrt)
    REGISTER_FOREACH_UNARY(rsqrt)
    REGISTER_FOREACH_UNARY(neg)
    REGISTER_FOREACH_UNARY(abs)
    REGISTER_FOREACH_UNARY(reciprocal)
    REGISTER_FOREACH_UNARY(sign)
    REGISTER_FOREACH_UNARY(acos)
    REGISTER_FOREACH_UNARY(asin)
    REGISTER_FOREACH_UNARY(atan)
    REGISTER_FOREACH_UNARY(ceil)
    REGISTER_FOREACH_UNARY(cos)
    REGISTER_FOREACH_UNARY(cosh)
    REGISTER_FOREACH_UNARY(erf)
    REGISTER_FOREACH_UNARY(erfc)
    REGISTER_FOREACH_UNARY(exp)
    REGISTER_FOREACH_UNARY(expm1)
    REGISTER_FOREACH_UNARY(floor)
    REGISTER_FOREACH_UNARY(frac)
    REGISTER_FOREACH_UNARY(lgamma)
    REGISTER_FOREACH_UNARY(log)
    REGISTER_FOREACH_UNARY(log10)
    REGISTER_FOREACH_UNARY(log1p)
    REGISTER_FOREACH_UNARY(log2)
    REGISTER_FOREACH_UNARY(round)
    REGISTER_FOREACH_UNARY(sigmoid)
    REGISTER_FOREACH_UNARY(sin)
    REGISTER_FOREACH_UNARY(sinh)
    REGISTER_FOREACH_UNARY(tan)
    REGISTER_FOREACH_UNARY(tanh)
    REGISTER_FOREACH_UNARY(trunc)
#undef REGISTER_FOREACH_UNARY

    m.impl("_foreach_addcmul.Scalar", foreach_addcmul_scalar_cpu);
    m.impl("_foreach_addcmul_.Scalar", foreach_addcmul_scalar_inplace_cpu);
    m.impl("_foreach_addcmul.ScalarList", foreach_addcmul_scalar_list_cpu);
    m.impl("_foreach_addcmul_.ScalarList", foreach_addcmul_scalar_list_inplace_cpu);
    m.impl("_foreach_addcmul.Tensor", foreach_addcmul_tensor_cpu);
    m.impl("_foreach_addcmul_.Tensor", foreach_addcmul_tensor_inplace_cpu);
    m.impl("_foreach_addcmul.Scalar_out", foreach_addcmul_scalar_out_cpu);
    m.impl("_foreach_addcmul.ScalarList_out", foreach_addcmul_scalar_list_out_cpu);
    m.impl("_foreach_addcmul.Tensor_out", foreach_addcmul_tensor_out_cpu);
    m.impl("_foreach_addcdiv.Scalar", foreach_addcdiv_scalar_cpu);
    m.impl("_foreach_addcdiv_.Scalar", foreach_addcdiv_scalar_inplace_cpu);
    m.impl("_foreach_addcdiv.ScalarList", foreach_addcdiv_scalar_list_cpu);
    m.impl("_foreach_addcdiv_.ScalarList", foreach_addcdiv_scalar_list_inplace_cpu);
    m.impl("_foreach_addcdiv.Tensor", foreach_addcdiv_tensor_cpu);
    m.impl("_foreach_addcdiv_.Tensor", foreach_addcdiv_tensor_inplace_cpu);
    m.impl("_foreach_addcdiv.Scalar_out", foreach_addcdiv_scalar_out_cpu);
    m.impl("_foreach_addcdiv.ScalarList_out", foreach_addcdiv_scalar_list_out_cpu);
    m.impl("_foreach_addcdiv.Tensor_out", foreach_addcdiv_tensor_out_cpu);
    m.impl("_foreach_lerp.Scalar", foreach_lerp_scalar_cpu);
    m.impl("_foreach_lerp.List", foreach_lerp_list_cpu);
    m.impl("_foreach_lerp_.Scalar", foreach_lerp_scalar_inplace_cpu);
    m.impl("_foreach_lerp_.List", foreach_lerp_list_inplace_cpu);
    m.impl("_foreach_lerp.ScalarList", foreach_lerp_scalar_list_cpu);
    m.impl("_foreach_lerp_.ScalarList", foreach_lerp_scalar_list_inplace_cpu);
    m.impl("_foreach_lerp.Scalar_out", foreach_lerp_scalar_out_cpu);
    m.impl("_foreach_lerp.List_out", foreach_lerp_list_out_cpu);
    m.impl("_foreach_lerp.ScalarList_out", foreach_lerp_scalar_list_out_cpu);
    m.impl("_foreach_pow.Scalar", foreach_pow_scalar_cpu);
    m.impl("_foreach_pow.ScalarAndTensor", foreach_pow_scalar_tensor_cpu);
    m.impl("_foreach_pow.TensorAndTensor", foreach_pow_tensor_tensor_cpu);
    m.impl("_foreach_pow.List", foreach_pow_list_cpu);
    m.impl("_foreach_pow_.Scalar", foreach_pow_scalar_inplace_cpu);
    m.impl("_foreach_pow_.List", foreach_pow_list_inplace_cpu);
    m.impl("_foreach_pow.ScalarList", foreach_pow_scalar_list_cpu);
    m.impl("_foreach_pow_.ScalarList", foreach_pow_scalar_list_inplace_cpu);
    m.impl("_foreach_clamp_min.Scalar", foreach_clamp_min_scalar_cpu);
    m.impl("_foreach_clamp_max.Scalar", foreach_clamp_max_scalar_cpu);
    m.impl("_foreach_clamp_min_.Scalar", foreach_clamp_min_scalar_inplace_cpu);
    m.impl("_foreach_clamp_max_.Scalar", foreach_clamp_max_scalar_inplace_cpu);
    m.impl("_foreach_clamp_min.List", foreach_clamp_min_list_cpu);
    m.impl("_foreach_clamp_min_.List", foreach_clamp_min_list_inplace_cpu);
    m.impl("_foreach_clamp_min.ScalarList", foreach_clamp_min_scalar_list_cpu);
    m.impl("_foreach_clamp_min_.ScalarList", foreach_clamp_min_scalar_list_inplace_cpu);
    m.impl("_foreach_clamp_max.List", foreach_clamp_max_list_cpu);
    m.impl("_foreach_clamp_max_.List", foreach_clamp_max_list_inplace_cpu);
    m.impl("_foreach_clamp_max.ScalarList", foreach_clamp_max_scalar_list_cpu);
    m.impl("_foreach_clamp_max_.ScalarList", foreach_clamp_max_scalar_list_inplace_cpu);
    m.impl("_foreach_maximum.Scalar", foreach_maximum_scalar_cpu);
    m.impl("_foreach_minimum.Scalar", foreach_minimum_scalar_cpu);
    m.impl("_foreach_maximum_.Scalar", foreach_maximum_scalar_inplace_cpu);
    m.impl("_foreach_minimum_.Scalar", foreach_minimum_scalar_inplace_cpu);
    m.impl("_foreach_maximum.List", foreach_maximum_list_cpu);
    m.impl("_foreach_maximum_.List", foreach_maximum_list_inplace_cpu);
    m.impl("_foreach_maximum.ScalarList", foreach_maximum_scalar_list_cpu);
    m.impl("_foreach_maximum_.ScalarList", foreach_maximum_scalar_list_inplace_cpu);
    m.impl("_foreach_minimum.List", foreach_minimum_list_cpu);
    m.impl("_foreach_minimum_.List", foreach_minimum_list_inplace_cpu);
    m.impl("_foreach_minimum.ScalarList", foreach_minimum_scalar_list_cpu);
    m.impl("_foreach_minimum_.ScalarList", foreach_minimum_scalar_list_inplace_cpu);
    m.impl("_foreach_clamp_min.Scalar_out", foreach_clamp_min_scalar_out_cpu);
    m.impl("_foreach_clamp_min.List_out", foreach_clamp_min_list_out_cpu);
    m.impl("_foreach_clamp_min.ScalarList_out", foreach_clamp_min_scalar_list_out_cpu);
    m.impl("_foreach_clamp_max.Scalar_out", foreach_clamp_max_scalar_out_cpu);
    m.impl("_foreach_clamp_max.List_out", foreach_clamp_max_list_out_cpu);
    m.impl("_foreach_clamp_max.ScalarList_out", foreach_clamp_max_scalar_list_out_cpu);
    m.impl("_foreach_maximum.Scalar_out", foreach_maximum_scalar_out_cpu);
    m.impl("_foreach_maximum.List_out", foreach_maximum_list_out_cpu);
    m.impl("_foreach_maximum.ScalarList_out", foreach_maximum_scalar_list_out_cpu);
    m.impl("_foreach_minimum.Scalar_out", foreach_minimum_scalar_out_cpu);
    m.impl("_foreach_minimum.List_out", foreach_minimum_list_out_cpu);
    m.impl("_foreach_minimum.ScalarList_out", foreach_minimum_scalar_list_out_cpu);
    m.impl("_foreach_copy_", foreach_copy_inplace_cpu);
    m.impl("_foreach_zero", foreach_zero_cpu);
    m.impl("_foreach_zero_", foreach_zero_inplace_cpu);
    m.impl("_foreach_max", foreach_max_cpu);
    m.impl("_foreach_norm.Scalar", foreach_norm_cpu);
    m.impl("_foreach_powsum.Scalar", foreach_powsum_cpu);
    m.impl("_foreach_clone", foreach_clone_cpu);
    m.impl("_foreach_copy", foreach_copy_cpu);
    m.impl("_foreach_mm", foreach_mm_cpu);
    m.impl("_foreach_clone.out", foreach_clone_out_cpu);
    m.impl("_foreach_copy.out", foreach_copy_out_cpu);
    m.impl("_foreach_max.out", foreach_max_out_cpu);
    m.impl("_foreach_norm.Scalar_out", foreach_norm_out_cpu);
    m.impl("_foreach_pow.Scalar_out", foreach_pow_scalar_out_cpu);
    m.impl("_foreach_pow.List_out", foreach_pow_list_out_cpu);
    m.impl("_foreach_pow.ScalarList_out", foreach_pow_scalar_list_out_cpu);
    m.impl("_foreach_powsum.Scalar_out", foreach_powsum_out_cpu);
    m.impl("_foreach_zero.out", foreach_zero_out_cpu);
}

} // namespace cpu
} // namespace tensorplay
