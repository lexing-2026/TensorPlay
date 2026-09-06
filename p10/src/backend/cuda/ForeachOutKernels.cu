#include "Exception.h"
#include "ForeachKernels.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {
// ---------------------------------------------------------------------------
// Remaining functional foreach operations and the _out variant family.
// The _out variants compute functionally, then copy each result into the
// matching output handle.
// ---------------------------------------------------------------------------

static void copy_foreach_out_cuda(std::vector<Tensor> result,
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

#define DEFINE_FOREACH_EXTRA_UNARY(NAME) \
std::vector<Tensor> foreach_##NAME##_cuda(const std::vector<Tensor>& self) { \
    std::vector<Tensor> out; \
    out.reserve(self.size()); \
    for (const auto& value : self) out.push_back(value.NAME()); \
    return out; \
} \
void foreach_##NAME##_inplace_cuda(std::vector<Tensor> self) { \
    for (auto& value : self) value.copy_(value.NAME()); \
}
DEFINE_FOREACH_EXTRA_UNARY(acos)
DEFINE_FOREACH_EXTRA_UNARY(asin)
DEFINE_FOREACH_EXTRA_UNARY(atan)
DEFINE_FOREACH_EXTRA_UNARY(ceil)
DEFINE_FOREACH_EXTRA_UNARY(cos)
DEFINE_FOREACH_EXTRA_UNARY(cosh)
DEFINE_FOREACH_EXTRA_UNARY(erf)
DEFINE_FOREACH_EXTRA_UNARY(erfc)
DEFINE_FOREACH_EXTRA_UNARY(exp)
DEFINE_FOREACH_EXTRA_UNARY(expm1)
DEFINE_FOREACH_EXTRA_UNARY(floor)
DEFINE_FOREACH_EXTRA_UNARY(frac)
DEFINE_FOREACH_EXTRA_UNARY(lgamma)
DEFINE_FOREACH_EXTRA_UNARY(log)
DEFINE_FOREACH_EXTRA_UNARY(log10)
DEFINE_FOREACH_EXTRA_UNARY(log1p)
DEFINE_FOREACH_EXTRA_UNARY(log2)
DEFINE_FOREACH_EXTRA_UNARY(round)
DEFINE_FOREACH_EXTRA_UNARY(sigmoid)
DEFINE_FOREACH_EXTRA_UNARY(sin)
DEFINE_FOREACH_EXTRA_UNARY(sinh)
DEFINE_FOREACH_EXTRA_UNARY(tanh)
DEFINE_FOREACH_EXTRA_UNARY(tan)
DEFINE_FOREACH_EXTRA_UNARY(trunc)
#undef DEFINE_FOREACH_EXTRA_UNARY

std::vector<Tensor> foreach_max_cuda(const std::vector<Tensor>& self) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& value : self) out.push_back(value.max());
    return out;
}

std::vector<Tensor> foreach_zero_cuda(const std::vector<Tensor>& self) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& value : self) out.push_back(Tensor::zeros_like(value));
    return out;
}

std::vector<Tensor> foreach_clone_cuda(const std::vector<Tensor>& self,
                                       std::optional<int64_t> /*memory_format*/) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& value : self) out.push_back(value.clone());
    return out;
}

std::vector<Tensor> foreach_copy_cuda(const std::vector<Tensor>& self,
                                      const std::vector<Tensor>& src,
                                      bool /*non_blocking*/) {
    if (self.size() != src.size()) {
        TP_THROW(ValueError, "_foreach_copy: list sizes must match");
    }
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) out.push_back(src[i].clone());
    return out;
}

std::vector<Tensor> foreach_mm_cuda(const std::vector<Tensor>& self,
                                    const std::vector<Tensor>& mat2) {
    if (self.size() != mat2.size()) {
        TP_THROW(ValueError, "_foreach_mm: list sizes must match");
    }
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i) out.push_back(self[i].mm(mat2[i]));
    return out;
}

std::vector<Tensor> foreach_norm_cuda(const std::vector<Tensor>& self,
                                      Scalar ord,
                                      std::optional<DType> dtype) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& value : self) {
        Tensor input = dtype.has_value() ? value.to(*dtype) : value;
        out.push_back(input.norm(ord.toDouble()));
    }
    return out;
}

std::vector<Tensor> foreach_powsum_cuda(const std::vector<Tensor>& self,
                                        Scalar ord,
                                        std::optional<DType> dtype) {
    std::vector<Tensor> out;
    out.reserve(self.size());
    for (const auto& value : self) {
        Tensor input = dtype.has_value() ? value.to(*dtype) : value;
        out.push_back(input.abs().pow(ord).sum());
    }
    return out;
}

#define DEFINE_FOREACH_UNARY_OUT_CUDA(NAME) \
void foreach_##NAME##_out_cuda(const std::vector<Tensor>& self, \
                               std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_cuda(self), std::move(out), \
                          "_foreach_" #NAME ".out"); \
}
DEFINE_FOREACH_UNARY_OUT_CUDA(sqrt)
DEFINE_FOREACH_UNARY_OUT_CUDA(rsqrt)
DEFINE_FOREACH_UNARY_OUT_CUDA(neg)
DEFINE_FOREACH_UNARY_OUT_CUDA(abs)
DEFINE_FOREACH_UNARY_OUT_CUDA(sign)
DEFINE_FOREACH_UNARY_OUT_CUDA(reciprocal)
DEFINE_FOREACH_UNARY_OUT_CUDA(acos)
DEFINE_FOREACH_UNARY_OUT_CUDA(asin)
DEFINE_FOREACH_UNARY_OUT_CUDA(atan)
DEFINE_FOREACH_UNARY_OUT_CUDA(ceil)
DEFINE_FOREACH_UNARY_OUT_CUDA(cos)
DEFINE_FOREACH_UNARY_OUT_CUDA(cosh)
DEFINE_FOREACH_UNARY_OUT_CUDA(erf)
DEFINE_FOREACH_UNARY_OUT_CUDA(erfc)
DEFINE_FOREACH_UNARY_OUT_CUDA(exp)
DEFINE_FOREACH_UNARY_OUT_CUDA(expm1)
DEFINE_FOREACH_UNARY_OUT_CUDA(floor)
DEFINE_FOREACH_UNARY_OUT_CUDA(frac)
DEFINE_FOREACH_UNARY_OUT_CUDA(lgamma)
DEFINE_FOREACH_UNARY_OUT_CUDA(log)
DEFINE_FOREACH_UNARY_OUT_CUDA(log10)
DEFINE_FOREACH_UNARY_OUT_CUDA(log1p)
DEFINE_FOREACH_UNARY_OUT_CUDA(log2)
DEFINE_FOREACH_UNARY_OUT_CUDA(round)
DEFINE_FOREACH_UNARY_OUT_CUDA(sigmoid)
DEFINE_FOREACH_UNARY_OUT_CUDA(sin)
DEFINE_FOREACH_UNARY_OUT_CUDA(sinh)
DEFINE_FOREACH_UNARY_OUT_CUDA(tan)
DEFINE_FOREACH_UNARY_OUT_CUDA(tanh)
DEFINE_FOREACH_UNARY_OUT_CUDA(trunc)
#undef DEFINE_FOREACH_UNARY_OUT_CUDA

#define DEFINE_FOREACH_ADDSUB_OUT_CUDA(NAME) \
void foreach_##NAME##_scalar_out_cuda(const std::vector<Tensor>& self, Scalar scalar, \
                                      std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_cuda(self, scalar), std::move(out), \
                          "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_list_out_cuda(const std::vector<Tensor>& self, \
                                    const std::vector<Tensor>& other, Scalar alpha, \
                                    std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_list_cuda(self, other, alpha), std::move(out), \
                          "_foreach_" #NAME ".List_out"); \
} \
void foreach_##NAME##_scalar_list_out_cuda(const std::vector<Tensor>& self, \
                                           const std::vector<Scalar>& scalars, \
                                           std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_list_cuda(self, scalars), std::move(out), \
                          "_foreach_" #NAME ".ScalarList_out"); \
} \
void foreach_##NAME##_tensor_out_cuda(const std::vector<Tensor>& self, const Tensor& other, \
                                      Scalar alpha, std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_tensor_cuda(self, other, alpha), std::move(out), \
                          "_foreach_" #NAME ".Tensor_out"); \
}
DEFINE_FOREACH_ADDSUB_OUT_CUDA(add)
DEFINE_FOREACH_ADDSUB_OUT_CUDA(sub)
#undef DEFINE_FOREACH_ADDSUB_OUT_CUDA

#define DEFINE_FOREACH_MULDIV_OUT_CUDA(NAME) \
void foreach_##NAME##_scalar_out_cuda(const std::vector<Tensor>& self, Scalar scalar, \
                                      std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_cuda(self, scalar), std::move(out), \
                          "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_list_out_cuda(const std::vector<Tensor>& self, \
                                    const std::vector<Tensor>& other, \
                                    std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_list_cuda(self, other), std::move(out), \
                          "_foreach_" #NAME ".List_out"); \
} \
void foreach_##NAME##_scalar_list_out_cuda(const std::vector<Tensor>& self, \
                                           const std::vector<Scalar>& scalars, \
                                           std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_list_cuda(self, scalars), std::move(out), \
                          "_foreach_" #NAME ".ScalarList_out"); \
} \
void foreach_##NAME##_tensor_out_cuda(const std::vector<Tensor>& self, const Tensor& other, \
                                      std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_tensor_cuda(self, other), std::move(out), \
                          "_foreach_" #NAME ".Tensor_out"); \
}
DEFINE_FOREACH_MULDIV_OUT_CUDA(mul)
DEFINE_FOREACH_MULDIV_OUT_CUDA(div)
#undef DEFINE_FOREACH_MULDIV_OUT_CUDA

#define DEFINE_FOREACH_CLAMP_OUT_CUDA(NAME) \
void foreach_##NAME##_scalar_out_cuda(const std::vector<Tensor>& self, Scalar scalar, \
                                      std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_cuda(self, scalar), std::move(out), \
                          "_foreach_" #NAME ".Scalar_out"); \
} \
void foreach_##NAME##_list_out_cuda(const std::vector<Tensor>& self, \
                                    const std::vector<Tensor>& other, \
                                    std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_list_cuda(self, other), std::move(out), \
                          "_foreach_" #NAME ".List_out"); \
} \
void foreach_##NAME##_scalar_list_out_cuda(const std::vector<Tensor>& self, \
                                           const std::vector<Scalar>& scalars, \
                                           std::vector<Tensor> out) { \
    copy_foreach_out_cuda(foreach_##NAME##_scalar_list_cuda(self, scalars), std::move(out), \
                          "_foreach_" #NAME ".ScalarList_out"); \
}
DEFINE_FOREACH_CLAMP_OUT_CUDA(clamp_max)
DEFINE_FOREACH_CLAMP_OUT_CUDA(clamp_min)
DEFINE_FOREACH_CLAMP_OUT_CUDA(maximum)
DEFINE_FOREACH_CLAMP_OUT_CUDA(minimum)
#undef DEFINE_FOREACH_CLAMP_OUT_CUDA

// lerp overloads have differing weight types; write them out explicitly.
void foreach_lerp_scalar_out_cuda(const std::vector<Tensor>& self,
                                  const std::vector<Tensor>& end,
                                  Scalar weight,
                                  std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_lerp_scalar_cuda(self, end, weight), std::move(out),
                          "_foreach_lerp.Scalar_out");
}
void foreach_lerp_list_out_cuda(const std::vector<Tensor>& self,
                                const std::vector<Tensor>& end,
                                const std::vector<Tensor>& weight,
                                std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_lerp_list_cuda(self, end, weight), std::move(out),
                          "_foreach_lerp.List_out");
}
void foreach_lerp_scalar_list_out_cuda(const std::vector<Tensor>& self,
                                       const std::vector<Tensor>& end,
                                       const std::vector<Scalar>& weights,
                                       std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_lerp_scalar_list_cuda(self, end, weights), std::move(out),
                          "_foreach_lerp.ScalarList_out");
}

void foreach_pow_scalar_out_cuda(const std::vector<Tensor>& self, Scalar exponent,
                                 std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_pow_scalar_cuda(self, exponent), std::move(out),
                          "_foreach_pow.Scalar_out");
}
void foreach_pow_list_out_cuda(const std::vector<Tensor>& self,
                               const std::vector<Tensor>& exponent,
                               std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_pow_list_cuda(self, exponent), std::move(out),
                          "_foreach_pow.List_out");
}
void foreach_pow_scalar_list_out_cuda(const std::vector<Tensor>& self,
                                      const std::vector<Scalar>& exponents,
                                      std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_pow_scalar_list_cuda(self, exponents), std::move(out),
                          "_foreach_pow.ScalarList_out");
}

void foreach_addcmul_scalar_out_cuda(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& tensor1,
                                     const std::vector<Tensor>& tensor2, Scalar value,
                                     std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcmul(tensor1[i], tensor2[i], value));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcmul.Scalar_out");
}
void foreach_addcmul_scalar_list_out_cuda(const std::vector<Tensor>& self,
                                          const std::vector<Tensor>& tensor1,
                                          const std::vector<Tensor>& tensor2,
                                          const std::vector<Scalar>& scalars,
                                          std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcmul(tensor1[i], tensor2[i], scalars[i]));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcmul.ScalarList_out");
}
void foreach_addcmul_tensor_out_cuda(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& tensor1,
                                     const std::vector<Tensor>& tensor2, Scalar value,
                                     std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcmul(tensor1[i], tensor2[i], value));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcmul.Tensor_out");
}
void foreach_addcdiv_scalar_out_cuda(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& tensor1,
                                     const std::vector<Tensor>& tensor2, Scalar value,
                                     std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcdiv(tensor1[i], tensor2[i], value));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcdiv.Scalar_out");
}
void foreach_addcdiv_scalar_list_out_cuda(const std::vector<Tensor>& self,
                                          const std::vector<Tensor>& tensor1,
                                          const std::vector<Tensor>& tensor2,
                                          const std::vector<Scalar>& scalars,
                                          std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcdiv(tensor1[i], tensor2[i], scalars[i]));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcdiv.ScalarList_out");
}
void foreach_addcdiv_tensor_out_cuda(const std::vector<Tensor>& self,
                                     const std::vector<Tensor>& tensor1,
                                     const std::vector<Tensor>& tensor2, Scalar value,
                                     std::vector<Tensor> out) {
    std::vector<Tensor> result;
    result.reserve(self.size());
    for (size_t i = 0; i < self.size(); ++i)
        result.push_back(self[i].addcdiv(tensor1[i], tensor2[i], value));
    copy_foreach_out_cuda(std::move(result), std::move(out), "_foreach_addcdiv.Tensor_out");
}

void foreach_max_out_cuda(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_max_cuda(self), std::move(out), "_foreach_max.out");
}
void foreach_zero_out_cuda(const std::vector<Tensor>& self, std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_zero_cuda(self), std::move(out), "_foreach_zero.out");
}
void foreach_clone_out_cuda(const std::vector<Tensor>& self,
                            std::optional<int64_t> memory_format,
                            std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_clone_cuda(self, memory_format), std::move(out),
                          "_foreach_clone.out");
}
void foreach_copy_out_cuda(const std::vector<Tensor>& self,
                           const std::vector<Tensor>& src, bool non_blocking,
                           std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_copy_cuda(self, src, non_blocking), std::move(out),
                          "_foreach_copy.out");
}
void foreach_norm_out_cuda(const std::vector<Tensor>& self, Scalar ord,
                           std::optional<DType> dtype, std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_norm_cuda(self, ord, dtype), std::move(out),
                          "_foreach_norm.Scalar_out");
}
void foreach_powsum_out_cuda(const std::vector<Tensor>& self, Scalar ord,
                             std::optional<DType> dtype, std::vector<Tensor> out) {
    copy_foreach_out_cuda(foreach_powsum_cuda(self, ord, dtype), std::move(out),
                          "_foreach_powsum.Scalar_out");
}

}  // namespace cuda
}  // namespace tensorplay
