#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include <optional>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cpu {
namespace {

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

DEFINE_FOREACH_OP(sub, sub)
DEFINE_FOREACH_OP(mul, mul)
DEFINE_FOREACH_OP(div, div)
#undef DEFINE_FOREACH_OP

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
#undef DEFINE_FOREACH_UNARY

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
    const Scalar value = scalars.item();
    return foreach_addcmul_scalar_cpu(self, tensor1, tensor2, value);
}
void foreach_addcmul_tensor_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    foreach_addcmul_scalar_inplace_cpu(std::move(self), tensor1, tensor2, scalars.item());
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
    return foreach_addcdiv_scalar_cpu(self, tensor1, tensor2, scalars.item());
}
void foreach_addcdiv_tensor_inplace_cpu(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    foreach_addcdiv_scalar_inplace_cpu(std::move(self), tensor1, tensor2, scalars.item());
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
void foreach_zero_inplace_cpu(std::vector<Tensor> self) {
    for (Tensor& value : self) value.zero_();
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

#define REGISTER_FOREACH_UNARY(NAME) \
    m.impl("_foreach_" #NAME, foreach_##NAME##_cpu); \
    m.impl("_foreach_" #NAME "_", foreach_##NAME##_inplace_cpu);
    REGISTER_FOREACH_UNARY(sqrt)
    REGISTER_FOREACH_UNARY(rsqrt)
    REGISTER_FOREACH_UNARY(neg)
    REGISTER_FOREACH_UNARY(abs)
    REGISTER_FOREACH_UNARY(reciprocal)
    REGISTER_FOREACH_UNARY(sign)
#undef REGISTER_FOREACH_UNARY

    m.impl("_foreach_addcmul.Scalar", foreach_addcmul_scalar_cpu);
    m.impl("_foreach_addcmul_.Scalar", foreach_addcmul_scalar_inplace_cpu);
    m.impl("_foreach_addcmul.ScalarList", foreach_addcmul_scalar_list_cpu);
    m.impl("_foreach_addcmul_.ScalarList", foreach_addcmul_scalar_list_inplace_cpu);
    m.impl("_foreach_addcmul.Tensor", foreach_addcmul_tensor_cpu);
    m.impl("_foreach_addcmul_.Tensor", foreach_addcmul_tensor_inplace_cpu);
    m.impl("_foreach_addcdiv.Scalar", foreach_addcdiv_scalar_cpu);
    m.impl("_foreach_addcdiv_.Scalar", foreach_addcdiv_scalar_inplace_cpu);
    m.impl("_foreach_addcdiv.ScalarList", foreach_addcdiv_scalar_list_cpu);
    m.impl("_foreach_addcdiv_.ScalarList", foreach_addcdiv_scalar_list_inplace_cpu);
    m.impl("_foreach_addcdiv.Tensor", foreach_addcdiv_tensor_cpu);
    m.impl("_foreach_addcdiv_.Tensor", foreach_addcdiv_tensor_inplace_cpu);
    m.impl("_foreach_lerp.Scalar", foreach_lerp_scalar_cpu);
    m.impl("_foreach_lerp.List", foreach_lerp_list_cpu);
    m.impl("_foreach_lerp_.Scalar", foreach_lerp_scalar_inplace_cpu);
    m.impl("_foreach_lerp_.List", foreach_lerp_list_inplace_cpu);
    m.impl("_foreach_lerp.ScalarList", foreach_lerp_scalar_list_cpu);
    m.impl("_foreach_lerp_.ScalarList", foreach_lerp_scalar_list_inplace_cpu);
    m.impl("_foreach_pow.Scalar", foreach_pow_scalar_cpu);
    m.impl("_foreach_pow.ScalarAndTensor", foreach_pow_scalar_tensor_cpu);
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
    m.impl("_foreach_copy_", foreach_copy_inplace_cpu);
    m.impl("_foreach_zero_", foreach_zero_inplace_cpu);
}

} // namespace cpu
} // namespace tensorplay
