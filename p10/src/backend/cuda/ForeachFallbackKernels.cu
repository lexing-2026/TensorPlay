#include "CUDARuntime.h"
#include "Exception.h"
#include "ForeachKernels.h"
#include "ForeachMultiTensor.cuh"

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

template <typename F>
std::vector<Tensor> foreach_map(const std::vector<Tensor>& self, F&& fn) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (const Tensor& value : self) result.push_back(fn(value));
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_inplace(std::vector<Tensor> self, F&& fn) {
    for (Tensor& value : self) fn(value);
    return self;
}

template <typename F>
std::vector<Tensor> foreach_map_pair(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& other, F&& fn) {
    if (self.size() != other.size()) {
        TP_THROW(ValueError, "foreach tensor list arguments must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) result.push_back(fn(self[i], other[i]));
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_pair_inplace(std::vector<Tensor> self,
                                             const std::vector<Tensor>& other,
                                             F&& fn) {
    if (self.size() != other.size()) {
        TP_THROW(ValueError, "foreach tensor list arguments must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) fn(self[i], other[i]);
    return self;
}

template <typename F>
std::vector<Tensor> foreach_map_scalars(const std::vector<Tensor>& self,
                                        const std::vector<Scalar>& scalars, F&& fn) {
    if (self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor and scalar lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) result.push_back(fn(self[i], scalars[i]));
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_scalars_inplace(std::vector<Tensor> self,
                                                const std::vector<Scalar>& scalars,
                                                F&& fn) {
    if (self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor and scalar lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) fn(self[i], scalars[i]);
    return self;
}

template <typename F>
std::vector<Tensor> foreach_map_ternary(const std::vector<Tensor>& self,
                                        const std::vector<Tensor>& tensor1,
                                        const std::vector<Tensor>& tensor2, F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size()) {
        TP_THROW(ValueError, "foreach ternary tensor lists must have the same length");
    }
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) result.push_back(fn(self[i], tensor1[i], tensor2[i]));
    return result;
}

template <typename F>
std::vector<Tensor> foreach_map_ternary_inplace(std::vector<Tensor> self,
                                                const std::vector<Tensor>& tensor1,
                                                const std::vector<Tensor>& tensor2, F&& fn) {
    if (self.size() != tensor1.size() || self.size() != tensor2.size()) {
        TP_THROW(ValueError, "foreach ternary tensor lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) fn(self[i], tensor1[i], tensor2[i]);
    return self;
}

template <typename F>
std::vector<Tensor> foreach_map_ternary_scalar_lists(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars,
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
std::vector<Tensor> foreach_map_ternary_scalar_lists_inplace(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars,
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
std::vector<Tensor> foreach_map_pair_scalars(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other,
        const std::vector<Scalar>& scalars, F&& fn) {
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
std::vector<Tensor> foreach_map_pair_scalars_inplace(
        std::vector<Tensor> self, const std::vector<Tensor>& other,
        const std::vector<Scalar>& scalars, F&& fn) {
    if (self.size() != other.size() || self.size() != scalars.size()) {
        TP_THROW(ValueError, "foreach tensor/scalar lists must have the same length");
    }
    for (size_t i = 0; i < self.size(); ++i) {
        fn(self[i], other[i], scalars[i]);
    }
    return self;
}

#define DEFINE_FOREACH_ADD_SUB(NAME, METHOD) \
std::vector<Tensor> foreach_##NAME##_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& other, Scalar alpha) { \
    return foreach_map_pair(self, other, [&](const Tensor& value, const Tensor& rhs) { return value.METHOD(rhs, alpha); }); \
} \
std::vector<Tensor> foreach_##NAME##_scalar_list_cuda(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) { \
    return foreach_map_scalars(self, scalars, [&](const Tensor& value, Scalar scalar) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_tensor_cuda(const std::vector<Tensor>& self, const Tensor& other, Scalar alpha) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(other, alpha); }); \
} \
void foreach_##NAME##_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.METHOD##_(scalar); }); \
} \
void foreach_##NAME##_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& other, Scalar alpha) { \
    foreach_map_pair_inplace(self, other, [&](Tensor& value, const Tensor& rhs) { value.METHOD##_(rhs, alpha); }); \
} \
void foreach_##NAME##_scalar_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Scalar>& scalars) { \
    foreach_map_scalars_inplace(self, scalars, [&](Tensor& value, Scalar scalar) { value.METHOD##_(scalar); }); \
} \
void foreach_##NAME##_tensor_inplace_cuda(std::vector<Tensor> self, const Tensor& other, Scalar alpha) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.METHOD##_(other, alpha); }); \
}

DEFINE_FOREACH_ADD_SUB(sub, sub)
#undef DEFINE_FOREACH_ADD_SUB

std::vector<Tensor> foreach_add_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) {
    return foreach_map(self, [&](const Tensor& value) { return value.add(scalar); });
}
std::vector<Tensor> foreach_add_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& other, Scalar alpha) {
    return foreach_map_pair(self, other, [&](const Tensor& value, const Tensor& rhs) { return value.add(rhs, alpha); });
}
std::vector<Tensor> foreach_add_scalar_list_cuda(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_map_scalars(self, scalars, [&](const Tensor& value, Scalar scalar) { return value.add(scalar); });
}
std::vector<Tensor> foreach_add_tensor_cuda(const std::vector<Tensor>& self, const Tensor& other, Scalar alpha) {
    return foreach_map(self, [&](const Tensor& value) { return value.add(other, alpha); });
}
void foreach_add_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) {
    foreach_map_inplace(self, [&](Tensor& value) { value.add_(scalar); });
}
void foreach_add_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& other, Scalar alpha) {
    foreach_map_pair_inplace(self, other, [&](Tensor& value, const Tensor& rhs) { value.add_(rhs, alpha); });
}
void foreach_add_scalar_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_map_scalars_inplace(self, scalars, [&](Tensor& value, Scalar scalar) { value.add_(scalar); });
}
void foreach_add_tensor_inplace_cuda(std::vector<Tensor> self, const Tensor& other, Scalar alpha) {
    foreach_map_inplace(self, [&](Tensor& value) { value.add_(other, alpha); });
}

#define DEFINE_FOREACH_MUL_DIV(NAME, METHOD) \
std::vector<Tensor> foreach_##NAME##_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& other) { \
    return foreach_map_pair(self, other, [&](const Tensor& value, const Tensor& rhs) { return value.METHOD(rhs); }); \
} \
std::vector<Tensor> foreach_##NAME##_scalar_list_cuda(const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) { \
    return foreach_map_scalars(self, scalars, [&](const Tensor& value, Scalar scalar) { return value.METHOD(scalar); }); \
} \
std::vector<Tensor> foreach_##NAME##_tensor_cuda(const std::vector<Tensor>& self, const Tensor& other) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(other); }); \
} \
void foreach_##NAME##_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.METHOD##_(scalar); }); \
} \
void foreach_##NAME##_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& other) { \
    foreach_map_pair_inplace(self, other, [&](Tensor& value, const Tensor& rhs) { value.METHOD##_(rhs); }); \
} \
void foreach_##NAME##_scalar_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Scalar>& scalars) { \
    foreach_map_scalars_inplace(self, scalars, [&](Tensor& value, Scalar scalar) { value.METHOD##_(scalar); }); \
} \
void foreach_##NAME##_tensor_inplace_cuda(std::vector<Tensor> self, const Tensor& other) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.METHOD##_(other); }); \
}

DEFINE_FOREACH_MUL_DIV(mul, mul)
DEFINE_FOREACH_MUL_DIV(div, div)
#undef DEFINE_FOREACH_MUL_DIV

#define DEFINE_FOREACH_UNARY(NAME, METHOD) \
std::vector<Tensor> foreach_##NAME##_cuda(const std::vector<Tensor>& self) { \
    return foreach_map(self, [&](const Tensor& value) { return value.METHOD(); }); \
} \
void foreach_##NAME##_inplace_cuda(std::vector<Tensor> self) { \
    foreach_map_inplace(self, [&](Tensor& value) { value.copy_(value.METHOD()); }); \
}
DEFINE_FOREACH_UNARY(sqrt, sqrt)
DEFINE_FOREACH_UNARY(rsqrt, rsqrt)
DEFINE_FOREACH_UNARY(neg, neg)
DEFINE_FOREACH_UNARY(abs, abs)
DEFINE_FOREACH_UNARY(sign, sign)
#undef DEFINE_FOREACH_UNARY

std::vector<Tensor> foreach_reciprocal_cuda(const std::vector<Tensor>& self) {
    return foreach_map(self, [&](const Tensor& value) {
        return value.pow(Scalar(-1));
    });
}
void foreach_reciprocal_inplace_cuda(std::vector<Tensor> self) {
    foreach_map_inplace(self, [&](Tensor& value) {
        value.copy_(value.pow(Scalar(-1)));
    });
}

std::vector<Tensor> foreach_addcmul_scalar_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, Scalar value) {
    return foreach_map_ternary(self, tensor1, tensor2, [&](const Tensor& x, const Tensor& a, const Tensor& b) { return x.addcmul(a, b, value); });
}
void foreach_addcmul_scalar_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, Scalar value) {
    foreach_map_ternary_inplace(self, tensor1, tensor2, [&](Tensor& x, const Tensor& a, const Tensor& b) { x.addcmul_(a, b, value); });
}
std::vector<Tensor> foreach_addcdiv_scalar_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, Scalar value) {
    return foreach_map_ternary(self, tensor1, tensor2, [&](const Tensor& x, const Tensor& a, const Tensor& b) { return x.addcdiv(a, b, value); });
}
void foreach_addcdiv_scalar_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& tensor1, const std::vector<Tensor>& tensor2, Scalar value) {
    foreach_map_ternary_inplace(self, tensor1, tensor2, [&](Tensor& x, const Tensor& a, const Tensor& b) { x.addcdiv_(a, b, value); });
}

std::vector<Tensor> foreach_addcmul_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    return foreach_map_ternary_scalar_lists(self, tensor1, tensor2, scalars,
        [&](const Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            return x.addcmul(a, b, value);
        });
}
void foreach_addcmul_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    foreach_map_ternary_scalar_lists_inplace(std::move(self), tensor1, tensor2, scalars,
        [&](Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            x.addcmul_(a, b, value);
        });
}
std::vector<Tensor> foreach_addcmul_tensor_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    return foreach_addcmul_scalar_cuda(self, tensor1, tensor2, scalars.item());
}
void foreach_addcmul_tensor_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    foreach_addcmul_scalar_inplace_cuda(std::move(self), tensor1, tensor2, scalars.item());
}

std::vector<Tensor> foreach_addcdiv_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    return foreach_map_ternary_scalar_lists(self, tensor1, tensor2, scalars,
        [&](const Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            return x.addcdiv(a, b, value);
        });
}
void foreach_addcdiv_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {
    foreach_map_ternary_scalar_lists_inplace(std::move(self), tensor1, tensor2, scalars,
        [&](Tensor& x, const Tensor& a, const Tensor& b, Scalar value) {
            x.addcdiv_(a, b, value);
        });
}
std::vector<Tensor> foreach_addcdiv_tensor_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    return foreach_addcdiv_scalar_cuda(self, tensor1, tensor2, scalars.item());
}
void foreach_addcdiv_tensor_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const Tensor& scalars) {
    foreach_addcdiv_scalar_inplace_cuda(std::move(self), tensor1, tensor2, scalars.item());
}

std::vector<Tensor> foreach_lerp_scalar_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& end, Scalar weight) {
    return foreach_map_pair(self, end, [&](const Tensor& x, const Tensor& y) { return x.lerp(y, weight); });
}
std::vector<Tensor> foreach_lerp_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& end, const std::vector<Tensor>& weight) {
    if (self.size() != end.size() || self.size() != weight.size()) TP_THROW(ValueError, "foreach lerp lists must have the same length");
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) result.push_back(self[i].lerp(end[i], weight[i]));
    return result;
}
void foreach_lerp_scalar_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& end, Scalar weight) {
    foreach_map_pair_inplace(self, end, [&](Tensor& x, const Tensor& y) { x.copy_(x.lerp(y, weight)); });
}
void foreach_lerp_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& end, const std::vector<Tensor>& weight) {
    if (self.size() != end.size() || self.size() != weight.size()) TP_THROW(ValueError, "foreach lerp lists must have the same length");
    for (size_t i = 0; i < self.size(); ++i) self[i].copy_(self[i].lerp(end[i], weight[i]));
}
std::vector<Tensor> foreach_lerp_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weight) {
    return foreach_map_pair_scalars(self, end, weight,
        [&](const Tensor& x, const Tensor& y, Scalar w) { return x.lerp(y, w); });
}
void foreach_lerp_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weight) {
    foreach_map_pair_scalars_inplace(std::move(self), end, weight,
        [&](Tensor& x, const Tensor& y, Scalar w) { x.copy_(x.lerp(y, w)); });
}

std::vector<Tensor> foreach_pow_scalar_cuda(const std::vector<Tensor>& self, Scalar exponent) {
    return foreach_map(self, [&](const Tensor& value) { return value.pow(exponent); });
}
std::vector<Tensor> foreach_pow_scalar_tensor_cuda(
        Scalar self, const std::vector<Tensor>& exponent) {
    return foreach_map(exponent, [&](const Tensor& value) {
        Tensor base = Tensor::full({}, self, value.dtype(), value.device());
        return base.pow(value);
    });
}
// cpu foreach_pow_tensor_tensor_cpu: one base tensor, per-element exponents
// -- out[i] = self ** exponent[i] (broadcast via the dispatcher pow op).
std::vector<Tensor> foreach_pow_tensor_tensor_cuda(const Tensor& self,
                                                  const std::vector<Tensor>& exponent) {
    return foreach_map(exponent, [&](const Tensor& value) { return self.pow(value); });
}
std::vector<Tensor> foreach_pow_list_cuda(const std::vector<Tensor>& self, const std::vector<Tensor>& exponent) {
    return foreach_map_pair(self, exponent, [&](const Tensor& value, const Tensor& rhs) { return value.pow(rhs); });
}
void foreach_pow_scalar_inplace_cuda(std::vector<Tensor> self, Scalar exponent) {
    foreach_map_inplace(self, [&](Tensor& value) { value.copy_(value.pow(exponent)); });
}
void foreach_pow_list_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& exponent) {
    foreach_map_pair_inplace(self, exponent, [&](Tensor& value, const Tensor& rhs) { value.copy_(value.pow(rhs)); });
}
std::vector<Tensor> foreach_pow_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& exponent) {
    return foreach_map_scalars(self, exponent,
        [&](const Tensor& value, Scalar rhs) { return value.pow(rhs); });
}
void foreach_pow_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& exponent) {
    foreach_map_scalars_inplace(std::move(self), exponent,
        [&](Tensor& value, Scalar rhs) { value.copy_(value.pow(rhs)); });
}

std::vector<Tensor> foreach_clamp_min_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) {
    return foreach_map(self, [&](const Tensor& value) { return value.clamp(scalar, std::nullopt); });
}
std::vector<Tensor> foreach_clamp_max_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) {
    return foreach_map(self, [&](const Tensor& value) { return value.clamp(std::nullopt, scalar); });
}
void foreach_clamp_min_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) {
    foreach_map_inplace(self, [&](Tensor& value) { value.copy_(value.clamp(scalar, std::nullopt)); });
}
void foreach_clamp_max_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) {
    foreach_map_inplace(self, [&](Tensor& value) { value.copy_(value.clamp(std::nullopt, scalar)); });
}
std::vector<Tensor> foreach_clamp_min_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return foreach_map_pair(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::maximum(value, rhs); });
}
void foreach_clamp_min_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    foreach_map_pair_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::maximum(value, rhs)); });
}
std::vector<Tensor> foreach_clamp_max_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return foreach_map_pair(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::minimum(value, rhs); });
}
void foreach_clamp_max_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    foreach_map_pair_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::minimum(value, rhs)); });
}
std::vector<Tensor> foreach_clamp_min_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_map_scalars(self, scalars,
        [&](const Tensor& value, Scalar rhs) { return value.clamp(rhs, std::nullopt); });
}
void foreach_clamp_min_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_map_scalars_inplace(std::move(self), scalars,
        [&](Tensor& value, Scalar rhs) { value.copy_(value.clamp(rhs, std::nullopt)); });
}
std::vector<Tensor> foreach_clamp_max_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_map_scalars(self, scalars,
        [&](const Tensor& value, Scalar rhs) { return value.clamp(std::nullopt, rhs); });
}
void foreach_clamp_max_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_map_scalars_inplace(std::move(self), scalars,
        [&](Tensor& value, Scalar rhs) { value.copy_(value.clamp(std::nullopt, rhs)); });
}
std::vector<Tensor> foreach_maximum_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) { return foreach_clamp_min_scalar_cuda(self, scalar); }
std::vector<Tensor> foreach_minimum_scalar_cuda(const std::vector<Tensor>& self, Scalar scalar) { return foreach_clamp_max_scalar_cuda(self, scalar); }
void foreach_maximum_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) { foreach_clamp_min_scalar_inplace_cuda(self, scalar); }
void foreach_minimum_scalar_inplace_cuda(std::vector<Tensor> self, Scalar scalar) { foreach_clamp_max_scalar_inplace_cuda(self, scalar); }
std::vector<Tensor> foreach_maximum_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return foreach_map_pair(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::maximum(value, rhs); });
}
void foreach_maximum_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    foreach_map_pair_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::maximum(value, rhs)); });
}
std::vector<Tensor> foreach_maximum_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_clamp_min_scalar_list_cuda(self, scalars);
}
void foreach_maximum_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_clamp_min_scalar_list_inplace_cuda(std::move(self), scalars);
}
std::vector<Tensor> foreach_minimum_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {
    return foreach_map_pair(self, other,
        [&](const Tensor& value, const Tensor& rhs) { return Tensor::minimum(value, rhs); });
}
void foreach_minimum_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& other) {
    foreach_map_pair_inplace(std::move(self), other,
        [&](Tensor& value, const Tensor& rhs) { value.copy_(Tensor::minimum(value, rhs)); });
}
std::vector<Tensor> foreach_minimum_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {
    return foreach_clamp_max_scalar_list_cuda(self, scalars);
}
void foreach_minimum_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {
    foreach_clamp_max_scalar_list_inplace_cuda(std::move(self), scalars);
}
template <typename M>
struct CopyListOp {
    __device__ M operator()(M* values) const { return values[1]; }
};

void foreach_copy_inplace_cuda(std::vector<Tensor> self, const std::vector<Tensor>& src, bool non_blocking) {
    // Fused multi-tensor copy (one launch for the whole list) for the common
    // case: matching dtypes/shapes, contiguous, supported float dtype — the
    // DDP reducer's bucket copy-in/copy-back and optimizer buffer syncs all
    // hit this path.  Falls back to per-tensor copy_ otherwise.
    if (!self.empty() && foreach_mta::eligible_pair(self, src)) {
        const DType dt = self[0].dtype();
        if (dt == DType::Float32) {
            foreach_mta::launch<2, 0, float, float, CopyListOp<float>>(
                {&self, &src}, CopyListOp<float>{}, "_foreach_copy_");
            for (Tensor& value : self) value.unsafeGetTensorImpl()->bump_version();
            return;
        }
        if (dt == DType::Float64) {
            foreach_mta::launch<2, 0, double, double, CopyListOp<double>>(
                {&self, &src}, CopyListOp<double>{}, "_foreach_copy_");
            for (Tensor& value : self) value.unsafeGetTensorImpl()->bump_version();
            return;
        }
        if (dt == DType::Float16) {
            foreach_mta::launch<2, 0, Half, float, CopyListOp<float>>(
                {&self, &src}, CopyListOp<float>{}, "_foreach_copy_");
            for (Tensor& value : self) value.unsafeGetTensorImpl()->bump_version();
            return;
        }
        if (dt == DType::BFloat16) {
            foreach_mta::launch<2, 0, BFloat16, float, CopyListOp<float>>(
                {&self, &src}, CopyListOp<float>{}, "_foreach_copy_");
            for (Tensor& value : self) value.unsafeGetTensorImpl()->bump_version();
            return;
        }
    }
    foreach_map_pair_inplace(self, src, [&](Tensor& value, const Tensor& rhs) { value.copy_(rhs, non_blocking); });
}
void foreach_zero_inplace_cuda(std::vector<Tensor> self) {
    for (Tensor& value : self) value.zero_();
}

}  // namespace cuda
}  // namespace tensorplay
