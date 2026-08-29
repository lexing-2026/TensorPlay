#pragma once

#include "Tensor.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace cuda {

#define TP_DECLARE_FOREACH_ADD_SUB(NAME)                                      \
std::vector<Tensor> foreach_##NAME##_scalar_cuda(                             \
        const std::vector<Tensor>& self, Scalar scalar);                     \
std::vector<Tensor> foreach_##NAME##_list_cuda(                               \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other,    \
        Scalar alpha);                                                        \
std::vector<Tensor> foreach_##NAME##_scalar_list_cuda(                        \
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars); \
std::vector<Tensor> foreach_##NAME##_tensor_cuda(                             \
        const std::vector<Tensor>& self, const Tensor& other, Scalar alpha);  \
void foreach_##NAME##_scalar_inplace_cuda(                                    \
        std::vector<Tensor> self, Scalar scalar);                             \
void foreach_##NAME##_list_inplace_cuda(                                      \
        std::vector<Tensor> self, const std::vector<Tensor>& other,           \
        Scalar alpha);                                                        \
void foreach_##NAME##_scalar_list_inplace_cuda(                               \
        std::vector<Tensor> self, const std::vector<Scalar>& scalars);        \
void foreach_##NAME##_tensor_inplace_cuda(                                    \
        std::vector<Tensor> self, const Tensor& other, Scalar alpha);

TP_DECLARE_FOREACH_ADD_SUB(add)
TP_DECLARE_FOREACH_ADD_SUB(sub)
#undef TP_DECLARE_FOREACH_ADD_SUB

#define TP_DECLARE_FOREACH_MUL_DIV(NAME)                                      \
std::vector<Tensor> foreach_##NAME##_scalar_cuda(                             \
        const std::vector<Tensor>& self, Scalar scalar);                     \
std::vector<Tensor> foreach_##NAME##_list_cuda(                               \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other);   \
std::vector<Tensor> foreach_##NAME##_scalar_list_cuda(                        \
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars); \
std::vector<Tensor> foreach_##NAME##_tensor_cuda(                             \
        const std::vector<Tensor>& self, const Tensor& other);                \
void foreach_##NAME##_scalar_inplace_cuda(                                    \
        std::vector<Tensor> self, Scalar scalar);                             \
void foreach_##NAME##_list_inplace_cuda(                                      \
        std::vector<Tensor> self, const std::vector<Tensor>& other);          \
void foreach_##NAME##_scalar_list_inplace_cuda(                               \
        std::vector<Tensor> self, const std::vector<Scalar>& scalars);        \
void foreach_##NAME##_tensor_inplace_cuda(                                    \
        std::vector<Tensor> self, const Tensor& other);

TP_DECLARE_FOREACH_MUL_DIV(mul)
TP_DECLARE_FOREACH_MUL_DIV(div)
#undef TP_DECLARE_FOREACH_MUL_DIV

#define TP_DECLARE_FOREACH_UNARY(NAME)                                        \
std::vector<Tensor> foreach_##NAME##_cuda(                                    \
        const std::vector<Tensor>& self);                                    \
void foreach_##NAME##_inplace_cuda(std::vector<Tensor> self);

TP_DECLARE_FOREACH_UNARY(sqrt)
TP_DECLARE_FOREACH_UNARY(rsqrt)
TP_DECLARE_FOREACH_UNARY(neg)
TP_DECLARE_FOREACH_UNARY(abs)
TP_DECLARE_FOREACH_UNARY(sign)
#undef TP_DECLARE_FOREACH_UNARY

std::vector<Tensor> foreach_reciprocal_cuda(const std::vector<Tensor>& self);
void foreach_reciprocal_inplace_cuda(std::vector<Tensor> self);

#define TP_DECLARE_FOREACH_TERNARY(NAME)                                      \
std::vector<Tensor> foreach_##NAME##_scalar_cuda(                             \
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,  \
        const std::vector<Tensor>& tensor2, Scalar value);                    \
void foreach_##NAME##_scalar_inplace_cuda(                                    \
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,         \
        const std::vector<Tensor>& tensor2, Scalar value);                    \
std::vector<Tensor> foreach_##NAME##_scalar_list_cuda(                        \
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,  \
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& values);\
void foreach_##NAME##_scalar_list_inplace_cuda(                               \
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,         \
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& values);\
std::vector<Tensor> foreach_##NAME##_tensor_cuda(                             \
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,  \
        const std::vector<Tensor>& tensor2, const Tensor& value);             \
void foreach_##NAME##_tensor_inplace_cuda(                                    \
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,         \
        const std::vector<Tensor>& tensor2, const Tensor& value);

TP_DECLARE_FOREACH_TERNARY(addcmul)
TP_DECLARE_FOREACH_TERNARY(addcdiv)
#undef TP_DECLARE_FOREACH_TERNARY

std::vector<Tensor> foreach_lerp_scalar_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        Scalar weight);
std::vector<Tensor> foreach_lerp_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        const std::vector<Tensor>& weight);
void foreach_lerp_scalar_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& end, Scalar weight);
void foreach_lerp_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& end,
        const std::vector<Tensor>& weight);
std::vector<Tensor> foreach_lerp_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weight);
void foreach_lerp_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weight);

std::vector<Tensor> foreach_pow_scalar_cuda(
        const std::vector<Tensor>& self, Scalar exponent);
std::vector<Tensor> foreach_pow_scalar_tensor_cuda(
        Scalar self, const std::vector<Tensor>& exponent);
std::vector<Tensor> foreach_pow_tensor_tensor_cuda(
        const Tensor& self, const std::vector<Tensor>& exponent);
std::vector<Tensor> foreach_pow_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& exponent);
void foreach_pow_scalar_inplace_cuda(std::vector<Tensor> self, Scalar exponent);
void foreach_pow_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& exponent);
std::vector<Tensor> foreach_pow_scalar_list_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& exponent);
void foreach_pow_scalar_list_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Scalar>& exponent);

#define TP_DECLARE_FOREACH_CLAMP(NAME)                                       \
std::vector<Tensor> foreach_##NAME##_scalar_cuda(                            \
        const std::vector<Tensor>& self, Scalar scalar);                    \
void foreach_##NAME##_scalar_inplace_cuda(                                   \
        std::vector<Tensor> self, Scalar scalar);                            \
std::vector<Tensor> foreach_##NAME##_list_cuda(                              \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other);  \
void foreach_##NAME##_list_inplace_cuda(                                     \
        std::vector<Tensor> self, const std::vector<Tensor>& other);         \
std::vector<Tensor> foreach_##NAME##_scalar_list_cuda(                       \
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars);\
void foreach_##NAME##_scalar_list_inplace_cuda(                              \
        std::vector<Tensor> self, const std::vector<Scalar>& scalars);

TP_DECLARE_FOREACH_CLAMP(clamp_min)
TP_DECLARE_FOREACH_CLAMP(clamp_max)
TP_DECLARE_FOREACH_CLAMP(maximum)
TP_DECLARE_FOREACH_CLAMP(minimum)
#undef TP_DECLARE_FOREACH_CLAMP

void foreach_copy_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& src,
        bool non_blocking);
void foreach_zero_inplace_cuda(std::vector<Tensor> self);

std::vector<Tensor> foreach_max_cuda(const std::vector<Tensor>& self);
std::vector<Tensor> foreach_zero_cuda(const std::vector<Tensor>& self);
std::vector<Tensor> foreach_clone_cuda(
        const std::vector<Tensor>& self,
        const std::optional<int64_t>& memory_format);
std::vector<Tensor> foreach_copy_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& src,
        bool non_blocking);
std::vector<Tensor> foreach_mm_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& mat2);
std::vector<Tensor> foreach_norm_cuda(
        const std::vector<Tensor>& self, Scalar ord,
        const std::optional<DType>& dtype);
std::vector<Tensor> foreach_powsum_cuda(
        const std::vector<Tensor>& self, Scalar ord,
        const std::optional<DType>& dtype);

#define TP_DECLARE_FOREACH_UNARY_OUT(NAME)                                    \
void foreach_##NAME##_out_cuda(                                               \
        const std::vector<Tensor>& self, std::vector<Tensor> out);

TP_DECLARE_FOREACH_UNARY_OUT(sqrt)
TP_DECLARE_FOREACH_UNARY_OUT(rsqrt)
TP_DECLARE_FOREACH_UNARY_OUT(neg)
TP_DECLARE_FOREACH_UNARY_OUT(abs)
TP_DECLARE_FOREACH_UNARY_OUT(sign)
TP_DECLARE_FOREACH_UNARY_OUT(reciprocal)
TP_DECLARE_FOREACH_UNARY_OUT(acos)
TP_DECLARE_FOREACH_UNARY_OUT(asin)
TP_DECLARE_FOREACH_UNARY_OUT(atan)
TP_DECLARE_FOREACH_UNARY_OUT(ceil)
TP_DECLARE_FOREACH_UNARY_OUT(cos)
TP_DECLARE_FOREACH_UNARY_OUT(cosh)
TP_DECLARE_FOREACH_UNARY_OUT(erf)
TP_DECLARE_FOREACH_UNARY_OUT(erfc)
TP_DECLARE_FOREACH_UNARY_OUT(exp)
TP_DECLARE_FOREACH_UNARY_OUT(expm1)
TP_DECLARE_FOREACH_UNARY_OUT(floor)
TP_DECLARE_FOREACH_UNARY_OUT(frac)
TP_DECLARE_FOREACH_UNARY_OUT(lgamma)
TP_DECLARE_FOREACH_UNARY_OUT(log)
TP_DECLARE_FOREACH_UNARY_OUT(log10)
TP_DECLARE_FOREACH_UNARY_OUT(log1p)
TP_DECLARE_FOREACH_UNARY_OUT(log2)
TP_DECLARE_FOREACH_UNARY_OUT(round)
TP_DECLARE_FOREACH_UNARY_OUT(sigmoid)
TP_DECLARE_FOREACH_UNARY_OUT(sin)
TP_DECLARE_FOREACH_UNARY_OUT(sinh)
TP_DECLARE_FOREACH_UNARY_OUT(tan)
TP_DECLARE_FOREACH_UNARY_OUT(tanh)
TP_DECLARE_FOREACH_UNARY_OUT(trunc)
#undef TP_DECLARE_FOREACH_UNARY_OUT

#define TP_DECLARE_FOREACH_ADD_SUB_OUT(NAME)                                  \
void foreach_##NAME##_scalar_out_cuda(                                       \
        const std::vector<Tensor>& self, Scalar scalar,                      \
        std::vector<Tensor> out);                                             \
void foreach_##NAME##_list_out_cuda(                                         \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other,    \
        Scalar alpha, std::vector<Tensor> out);                               \
void foreach_##NAME##_scalar_list_out_cuda(                                  \
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars,  \
        std::vector<Tensor> out);                                             \
void foreach_##NAME##_tensor_out_cuda(                                       \
        const std::vector<Tensor>& self, const Tensor& other, Scalar alpha,   \
        std::vector<Tensor> out);

TP_DECLARE_FOREACH_ADD_SUB_OUT(add)
TP_DECLARE_FOREACH_ADD_SUB_OUT(sub)
#undef TP_DECLARE_FOREACH_ADD_SUB_OUT

#define TP_DECLARE_FOREACH_MUL_DIV_OUT(NAME)                                  \
void foreach_##NAME##_scalar_out_cuda(                                       \
        const std::vector<Tensor>& self, Scalar scalar,                      \
        std::vector<Tensor> out);                                             \
void foreach_##NAME##_list_out_cuda(                                         \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other,    \
        std::vector<Tensor> out);                                             \
void foreach_##NAME##_scalar_list_out_cuda(                                  \
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars,  \
        std::vector<Tensor> out);                                             \
void foreach_##NAME##_tensor_out_cuda(                                       \
        const std::vector<Tensor>& self, const Tensor& other,                 \
        std::vector<Tensor> out);

TP_DECLARE_FOREACH_MUL_DIV_OUT(mul)
TP_DECLARE_FOREACH_MUL_DIV_OUT(div)
#undef TP_DECLARE_FOREACH_MUL_DIV_OUT

#define TP_DECLARE_FOREACH_CLAMP_OUT(NAME)                                    \
void foreach_##NAME##_scalar_out_cuda(                                       \
        const std::vector<Tensor>& self, Scalar scalar,                      \
        std::vector<Tensor> out);                                             \
void foreach_##NAME##_list_out_cuda(                                         \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other,    \
        std::vector<Tensor> out);                                             \
void foreach_##NAME##_scalar_list_out_cuda(                                  \
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars,  \
        std::vector<Tensor> out);

TP_DECLARE_FOREACH_CLAMP_OUT(clamp_max)
TP_DECLARE_FOREACH_CLAMP_OUT(clamp_min)
TP_DECLARE_FOREACH_CLAMP_OUT(maximum)
TP_DECLARE_FOREACH_CLAMP_OUT(minimum)
#undef TP_DECLARE_FOREACH_CLAMP_OUT

void foreach_lerp_scalar_out_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        Scalar weight, std::vector<Tensor> out);
void foreach_lerp_list_out_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        const std::vector<Tensor>& weight, std::vector<Tensor> out);
void foreach_lerp_scalar_list_out_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weights, std::vector<Tensor> out);

void foreach_pow_scalar_out_cuda(
        const std::vector<Tensor>& self, Scalar exponent,
        std::vector<Tensor> out);
void foreach_pow_list_out_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& exponent,
        std::vector<Tensor> out);
void foreach_pow_scalar_list_out_cuda(
        const std::vector<Tensor>& self, const std::vector<Scalar>& exponents,
        std::vector<Tensor> out);

#define TP_DECLARE_FOREACH_TERNARY_OUT(NAME)                                  \
void foreach_##NAME##_scalar_out_cuda(                                       \
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,  \
        const std::vector<Tensor>& tensor2, Scalar value,                     \
        std::vector<Tensor> out);                                             \
void foreach_##NAME##_scalar_list_out_cuda(                                  \
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,  \
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& values, \
        std::vector<Tensor> out);                                             \
void foreach_##NAME##_tensor_out_cuda(                                       \
        const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,  \
        const std::vector<Tensor>& tensor2, Scalar value,                     \
        std::vector<Tensor> out);

TP_DECLARE_FOREACH_TERNARY_OUT(addcmul)
TP_DECLARE_FOREACH_TERNARY_OUT(addcdiv)
#undef TP_DECLARE_FOREACH_TERNARY_OUT

void foreach_max_out_cuda(const std::vector<Tensor>& self,
                          std::vector<Tensor> out);
void foreach_zero_out_cuda(const std::vector<Tensor>& self,
                           std::vector<Tensor> out);
void foreach_clone_out_cuda(
        const std::vector<Tensor>& self,
        const std::optional<int64_t>& memory_format, std::vector<Tensor> out);
void foreach_copy_out_cuda(
        const std::vector<Tensor>& self, const std::vector<Tensor>& src,
        bool non_blocking, std::vector<Tensor> out);
void foreach_norm_out_cuda(
        const std::vector<Tensor>& self, Scalar ord,
        const std::optional<DType>& dtype, std::vector<Tensor> out);
void foreach_powsum_out_cuda(
        const std::vector<Tensor>& self, Scalar ord,
        const std::optional<DType>& dtype, std::vector<Tensor> out);

#define TP_DECLARE_MTA_UNARY(NAME)                                            \
void foreach_##NAME##_mta_inplace_cuda(std::vector<Tensor> self);             \
std::vector<Tensor> foreach_##NAME##_mta_ret_cuda(                             \
        const std::vector<Tensor>& self);

TP_DECLARE_MTA_UNARY(sqrt)
TP_DECLARE_MTA_UNARY(rsqrt)
TP_DECLARE_MTA_UNARY(neg)
TP_DECLARE_MTA_UNARY(abs)
TP_DECLARE_MTA_UNARY(sign)
TP_DECLARE_MTA_UNARY(reciprocal)
TP_DECLARE_MTA_UNARY(acos)
TP_DECLARE_MTA_UNARY(asin)
TP_DECLARE_MTA_UNARY(atan)
TP_DECLARE_MTA_UNARY(ceil)
TP_DECLARE_MTA_UNARY(cos)
TP_DECLARE_MTA_UNARY(cosh)
TP_DECLARE_MTA_UNARY(erf)
TP_DECLARE_MTA_UNARY(erfc)
TP_DECLARE_MTA_UNARY(exp)
TP_DECLARE_MTA_UNARY(expm1)
TP_DECLARE_MTA_UNARY(floor)
TP_DECLARE_MTA_UNARY(frac)
TP_DECLARE_MTA_UNARY(lgamma)
TP_DECLARE_MTA_UNARY(log)
TP_DECLARE_MTA_UNARY(log10)
TP_DECLARE_MTA_UNARY(log1p)
TP_DECLARE_MTA_UNARY(log2)
TP_DECLARE_MTA_UNARY(round)
TP_DECLARE_MTA_UNARY(sigmoid)
TP_DECLARE_MTA_UNARY(sin)
TP_DECLARE_MTA_UNARY(sinh)
TP_DECLARE_MTA_UNARY(tan)
TP_DECLARE_MTA_UNARY(tanh)
TP_DECLARE_MTA_UNARY(trunc)
#undef TP_DECLARE_MTA_UNARY

void foreach_zero_mta_inplace_cuda(std::vector<Tensor> self);
std::vector<Tensor> foreach_zero_mta_ret_cuda(
        const std::vector<Tensor>& self);

#define TP_DECLARE_MTA_SCALAR(NAME)                                           \
void foreach_##NAME##_scalar_mta_inplace_cuda(                                \
        std::vector<Tensor> self, Scalar scalar);                             \
std::vector<Tensor> foreach_##NAME##_scalar_mta_ret_cuda(                     \
        const std::vector<Tensor>& self, Scalar scalar);

TP_DECLARE_MTA_SCALAR(add)
TP_DECLARE_MTA_SCALAR(sub)
TP_DECLARE_MTA_SCALAR(mul)
TP_DECLARE_MTA_SCALAR(div)
TP_DECLARE_MTA_SCALAR(pow)
TP_DECLARE_MTA_SCALAR(clamp_min)
TP_DECLARE_MTA_SCALAR(clamp_max)
TP_DECLARE_MTA_SCALAR(maximum)
TP_DECLARE_MTA_SCALAR(minimum)
#undef TP_DECLARE_MTA_SCALAR

#define TP_DECLARE_MTA_SCALAR_LIST(NAME)                                      \
void foreach_##NAME##_scalar_list_mta_inplace_cuda(                           \
        std::vector<Tensor> self, const std::vector<Scalar>& scalars);        \
std::vector<Tensor> foreach_##NAME##_scalar_list_mta_ret_cuda(                \
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars);

TP_DECLARE_MTA_SCALAR_LIST(add)
TP_DECLARE_MTA_SCALAR_LIST(sub)
TP_DECLARE_MTA_SCALAR_LIST(mul)
TP_DECLARE_MTA_SCALAR_LIST(div)
TP_DECLARE_MTA_SCALAR_LIST(pow)
TP_DECLARE_MTA_SCALAR_LIST(clamp_min)
TP_DECLARE_MTA_SCALAR_LIST(clamp_max)
TP_DECLARE_MTA_SCALAR_LIST(maximum)
TP_DECLARE_MTA_SCALAR_LIST(minimum)
#undef TP_DECLARE_MTA_SCALAR_LIST

#define TP_DECLARE_MTA_LIST(NAME)                                             \
void foreach_##NAME##_list_mta_inplace_cuda(                                  \
        std::vector<Tensor> self, const std::vector<Tensor>& other);          \
std::vector<Tensor> foreach_##NAME##_list_mta_ret_cuda(                       \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other);

TP_DECLARE_MTA_LIST(mul)
TP_DECLARE_MTA_LIST(div)
TP_DECLARE_MTA_LIST(pow)
TP_DECLARE_MTA_LIST(clamp_min)
TP_DECLARE_MTA_LIST(clamp_max)
TP_DECLARE_MTA_LIST(maximum)
TP_DECLARE_MTA_LIST(minimum)
#undef TP_DECLARE_MTA_LIST

#define TP_DECLARE_MTA_LIST_ALPHA(NAME)                                       \
void foreach_##NAME##_list_mta_inplace_cuda(                                  \
        std::vector<Tensor> self, const std::vector<Tensor>& other,           \
        Scalar alpha);                                                        \
std::vector<Tensor> foreach_##NAME##_list_mta_ret_cuda(                       \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other,    \
        Scalar alpha);

TP_DECLARE_MTA_LIST_ALPHA(add)
TP_DECLARE_MTA_LIST_ALPHA(sub)
#undef TP_DECLARE_MTA_LIST_ALPHA

void foreach_lerp_scalar_mta_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& end,
        Scalar weight);
void foreach_lerp_scalar_list_mta_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weights);
void foreach_addcmul_scalar_mta_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, Scalar value);
void foreach_addcmul_scalar_list_mta_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars);
void foreach_addcdiv_scalar_mta_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, Scalar value);
void foreach_addcdiv_scalar_list_mta_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& tensor1,
        const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars);

}  // namespace cuda
}  // namespace tensorplay
