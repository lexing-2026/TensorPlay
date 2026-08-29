#include "Exception.h"
#include "ForeachKernels.h"
#include "ForeachMultiTensor.cuh"

#include <array>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {
// ------------------------------------------------------------------
// Fused multi-tensor fast paths for the hot optimizer foreach ops.
//
// The per-tensor foreach_map_* implementations pay one kernel launch per
// tensor per op; a transformer-like group of 100+ small tensors benefits from
// one grouped launch.  These wrappers route
// eligible fp16/bf16/fp32/fp64 groups through foreach_mta::launch -- one
// launch walks chunks from every tensor -- and fall back to the
// per-tensor implementations otherwise.
// ------------------------------------------------------------------

namespace {

bool mta_ready(const std::vector<Tensor>& xs) {
    return !xs.empty() && xs.front().defined() &&
        xs.front().device().is_cuda();
}

template <typename M>
std::vector<M> mta_scalar_values(const std::vector<Scalar>& scalars) {
    std::vector<M> values;
    values.reserve(scalars.size());
    for (const Scalar& scalar : scalars) {
        values.push_back(scalar.to<M>());
    }
    return values;
}

void mta_bump(std::vector<Tensor>& xs) {
    for (Tensor& t : xs) t.unsafeGetTensorImpl()->bump_version();
}

std::vector<Tensor> foreach_alloc_like_cuda(const std::vector<Tensor>& xs) {
    std::vector<Tensor> out;
    out.reserve(xs.size());
    for (const Tensor& t : xs) out.push_back(Tensor::empty_like(t));
    return out;
}


}  // namespace

#define TP_MTA_UNARY_INPLACE(NAME, FUNCTOR)                                   \
void foreach_##NAME##_mta_inplace_cuda(std::vector<Tensor> self) {            \
    if (!mta_ready(self) ||                                                   \
        !foreach_mta::eligible_list(self)) {                                  \
        foreach_##NAME##_inplace_cuda(std::move(self));                       \
        return;                                                               \
    }                                                                         \
    const bool launched = foreach_mta::dispatch_dtype(                        \
        self[0].dtype(), [&]<typename T, typename M>() {                      \
            foreach_mta::launch<1, 0, T, M>(                                  \
                std::array<const std::vector<Tensor>*, 1>{&self},             \
                FUNCTOR<M>{}, "_foreach_" #NAME "_.cuda");                    \
        });                                                                   \
    if (!launched) {                                                          \
        foreach_##NAME##_inplace_cuda(std::move(self));                       \
        return;                                                               \
    }                                                                         \
    mta_bump(self);                                                           \
}

TP_MTA_UNARY_INPLACE(sqrt, foreach_mta::UnarySqrt)
TP_MTA_UNARY_INPLACE(rsqrt, foreach_mta::UnaryRsqrt)
TP_MTA_UNARY_INPLACE(neg, foreach_mta::UnaryNeg)
TP_MTA_UNARY_INPLACE(abs, foreach_mta::UnaryAbs)
TP_MTA_UNARY_INPLACE(sign, foreach_mta::UnarySign)
TP_MTA_UNARY_INPLACE(reciprocal, foreach_mta::UnaryReciprocal)
TP_MTA_UNARY_INPLACE(acos, foreach_mta::UnaryAcos)
TP_MTA_UNARY_INPLACE(asin, foreach_mta::UnaryAsin)
TP_MTA_UNARY_INPLACE(atan, foreach_mta::UnaryAtan)
TP_MTA_UNARY_INPLACE(ceil, foreach_mta::UnaryCeil)
TP_MTA_UNARY_INPLACE(cos, foreach_mta::UnaryCos)
TP_MTA_UNARY_INPLACE(cosh, foreach_mta::UnaryCosh)
TP_MTA_UNARY_INPLACE(erf, foreach_mta::UnaryErf)
TP_MTA_UNARY_INPLACE(erfc, foreach_mta::UnaryErfc)
TP_MTA_UNARY_INPLACE(exp, foreach_mta::UnaryExp)
TP_MTA_UNARY_INPLACE(expm1, foreach_mta::UnaryExpm1)
TP_MTA_UNARY_INPLACE(floor, foreach_mta::UnaryFloor)
TP_MTA_UNARY_INPLACE(frac, foreach_mta::UnaryFrac)
TP_MTA_UNARY_INPLACE(lgamma, foreach_mta::UnaryLgamma)
TP_MTA_UNARY_INPLACE(log, foreach_mta::UnaryLog)
TP_MTA_UNARY_INPLACE(log10, foreach_mta::UnaryLog10)
TP_MTA_UNARY_INPLACE(log1p, foreach_mta::UnaryLog1p)
TP_MTA_UNARY_INPLACE(log2, foreach_mta::UnaryLog2)
TP_MTA_UNARY_INPLACE(round, foreach_mta::UnaryRound)
TP_MTA_UNARY_INPLACE(sigmoid, foreach_mta::UnarySigmoid)
TP_MTA_UNARY_INPLACE(sin, foreach_mta::UnarySin)
TP_MTA_UNARY_INPLACE(sinh, foreach_mta::UnarySinh)
TP_MTA_UNARY_INPLACE(tan, foreach_mta::UnaryTan)
TP_MTA_UNARY_INPLACE(tanh, foreach_mta::UnaryTanh)
TP_MTA_UNARY_INPLACE(trunc, foreach_mta::UnaryTrunc)
#undef TP_MTA_UNARY_INPLACE

void foreach_zero_mta_inplace_cuda(std::vector<Tensor> self) {
    if (!mta_ready(self) || !foreach_mta::eligible_list(self)) {
        foreach_zero_inplace_cuda(std::move(self));
        return;
    }
    const bool launched = foreach_mta::dispatch_dtype(
        self[0].dtype(), [&]<typename T, typename M>() {
            foreach_mta::launch<1, 0, T, M>(
                std::array<const std::vector<Tensor>*, 1>{&self},
                foreach_mta::UnaryZero<M>{}, "_foreach_zero_.cuda");
        });
    if (!launched) {
        foreach_zero_inplace_cuda(std::move(self));
        return;
    }
    mta_bump(self);
}

#define TP_MTA_SCALAR_INPLACE(NAME, FUNCTOR)                                  \
void foreach_##NAME##_scalar_mta_inplace_cuda(std::vector<Tensor> self,       \
                                              Scalar s) {                     \
    if (!mta_ready(self) || !foreach_mta::eligible_list(self)) {              \
        foreach_##NAME##_scalar_inplace_cuda(std::move(self), s);             \
        return;                                                               \
    }                                                                         \
    const bool launched = foreach_mta::dispatch_dtype(                        \
        self[0].dtype(), [&]<typename T, typename M>() {                      \
            foreach_mta::launch<1, 0, T, M>(                                  \
                std::array<const std::vector<Tensor>*, 1>{&self},             \
                FUNCTOR<M>{s.to<M>()}, "_foreach_" #NAME "_.cuda");           \
        });                                                                   \
    if (!launched) {                                                          \
        foreach_##NAME##_scalar_inplace_cuda(std::move(self), s);             \
        return;                                                               \
    }                                                                         \
    mta_bump(self);                                                           \
}

TP_MTA_SCALAR_INPLACE(add, foreach_mta::BinaryAddScalar)
TP_MTA_SCALAR_INPLACE(sub, foreach_mta::BinarySubScalar)
TP_MTA_SCALAR_INPLACE(mul, foreach_mta::BinaryMulScalar)
TP_MTA_SCALAR_INPLACE(div, foreach_mta::BinaryDivScalar)
TP_MTA_SCALAR_INPLACE(pow, foreach_mta::UnaryPow)
TP_MTA_SCALAR_INPLACE(clamp_min, foreach_mta::BinaryMaximum)
TP_MTA_SCALAR_INPLACE(clamp_max, foreach_mta::BinaryMinimum)
TP_MTA_SCALAR_INPLACE(maximum, foreach_mta::BinaryMaximum)
TP_MTA_SCALAR_INPLACE(minimum, foreach_mta::BinaryMinimum)
#undef TP_MTA_SCALAR_INPLACE

#define TP_MTA_SCALAR_LIST_INPLACE(NAME, FUNCTOR)                             \
void foreach_##NAME##_scalar_list_mta_inplace_cuda(                           \
        std::vector<Tensor> self, const std::vector<Scalar>& scalars) {       \
    if (!mta_ready(self) || !foreach_mta::eligible_list(self) ||              \
        self.size() != scalars.size()) {                                      \
        foreach_##NAME##_scalar_list_inplace_cuda(std::move(self), scalars);   \
        return;                                                               \
    }                                                                          \
    const bool launched = foreach_mta::dispatch_dtype(                        \
        self[0].dtype(), [&]<typename T, typename M>() {                      \
            const std::vector<M> values = mta_scalar_values<M>(scalars);      \
            foreach_mta::launch_scalar_list<1, 0, T, M>(                      \
                std::array<const std::vector<Tensor>*, 1>{&self}, values,     \
                FUNCTOR<M>{}, "_foreach_" #NAME "_.ScalarList.cuda");        \
        });                                                                   \
    if (!launched) {                                                           \
        foreach_##NAME##_scalar_list_inplace_cuda(std::move(self), scalars);   \
        return;                                                                \
    }                                                                          \
    mta_bump(self);                                                            \
}

TP_MTA_SCALAR_LIST_INPLACE(add, foreach_mta::BinaryAddScalarList)
TP_MTA_SCALAR_LIST_INPLACE(sub, foreach_mta::BinarySubScalarList)
TP_MTA_SCALAR_LIST_INPLACE(mul, foreach_mta::BinaryMulScalarList)
TP_MTA_SCALAR_LIST_INPLACE(div, foreach_mta::BinaryDivScalarList)
TP_MTA_SCALAR_LIST_INPLACE(pow, foreach_mta::UnaryPowScalarList)
TP_MTA_SCALAR_LIST_INPLACE(clamp_min, foreach_mta::BinaryMaximumScalarList)
TP_MTA_SCALAR_LIST_INPLACE(clamp_max, foreach_mta::BinaryMinimumScalarList)
TP_MTA_SCALAR_LIST_INPLACE(maximum, foreach_mta::BinaryMaximumScalarList)
TP_MTA_SCALAR_LIST_INPLACE(minimum, foreach_mta::BinaryMinimumScalarList)
#undef TP_MTA_SCALAR_LIST_INPLACE

#define TP_MTA_LIST_INPLACE(NAME, FUNCTOR)                                    \
void foreach_##NAME##_list_mta_inplace_cuda(std::vector<Tensor> self,         \
                                            const std::vector<Tensor>& other) {\
    if (!mta_ready(self) ||                                                   \
        !foreach_mta::eligible_pair(self, other)) {                           \
        foreach_##NAME##_list_inplace_cuda(std::move(self), other);           \
        return;                                                               \
    }                                                                         \
    const bool launched = foreach_mta::dispatch_dtype(                        \
        self[0].dtype(), [&]<typename T, typename M>() {                      \
            foreach_mta::launch<2, 0, T, M>(                                  \
                std::array<const std::vector<Tensor>*, 2>{&self, &other},     \
                FUNCTOR<M>{}, "_foreach_" #NAME "_.cuda");                    \
        });                                                                   \
    if (!launched) {                                                          \
        foreach_##NAME##_list_inplace_cuda(std::move(self), other);           \
        return;                                                               \
    }                                                                         \
    mta_bump(self);                                                           \
}

TP_MTA_LIST_INPLACE(mul, foreach_mta::BinaryMulList)
TP_MTA_LIST_INPLACE(div, foreach_mta::BinaryDivList)
TP_MTA_LIST_INPLACE(pow, foreach_mta::BinaryPowList)
TP_MTA_LIST_INPLACE(clamp_min, foreach_mta::BinaryMaximumList)
TP_MTA_LIST_INPLACE(clamp_max, foreach_mta::BinaryMinimumList)
TP_MTA_LIST_INPLACE(maximum, foreach_mta::BinaryMaximumList)
TP_MTA_LIST_INPLACE(minimum, foreach_mta::BinaryMinimumList)
#undef TP_MTA_LIST_INPLACE

void foreach_add_list_mta_inplace_cuda(std::vector<Tensor> self,
                                       const std::vector<Tensor>& other,
                                       Scalar alpha) {
    if (!mta_ready(self) || !foreach_mta::eligible_pair(self, other)) {
        foreach_add_list_inplace_cuda(std::move(self), other, alpha);
        return;
    }
    const bool launched = foreach_mta::dispatch_dtype(
        self[0].dtype(), [&]<typename T, typename M>() {
            foreach_mta::launch<2, 0, T, M>(
                std::array<const std::vector<Tensor>*, 2>{&self, &other},
                foreach_mta::BinaryAddList<M>{alpha.to<M>()},
                "_foreach_add_.list.cuda");
        });
    if (!launched) {
        foreach_add_list_inplace_cuda(std::move(self), other, alpha);
        return;
    }
    mta_bump(self);
}

void foreach_lerp_scalar_mta_inplace_cuda(std::vector<Tensor> self,
                                          const std::vector<Tensor>& end,
                                          Scalar weight) {
    if (!mta_ready(self) || !foreach_mta::eligible_pair(self, end)) {
        foreach_lerp_scalar_inplace_cuda(std::move(self), end, weight);
        return;
    }
    const bool launched = foreach_mta::dispatch_dtype(
        self[0].dtype(), [&]<typename T, typename M>() {
            foreach_mta::launch<2, 0, T, M>(
                std::array<const std::vector<Tensor>*, 2>{&self, &end},
                foreach_mta::BinaryLerp<M>{weight.to<M>()},
                "_foreach_lerp_.cuda");
        });
    if (!launched) {
        foreach_lerp_scalar_inplace_cuda(std::move(self), end, weight);
        return;
    }
    mta_bump(self);
}

void foreach_addcmul_scalar_mta_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& t1,
        const std::vector<Tensor>& t2, Scalar value) {
    if (!mta_ready(self) || !foreach_mta::eligible_ternary(self, t1, t2)) {
        foreach_addcmul_scalar_inplace_cuda(std::move(self), t1, t2, value);
        return;
    }
    const bool launched = foreach_mta::dispatch_dtype(
        self[0].dtype(), [&]<typename T, typename M>() {
            foreach_mta::launch<3, 0, T, M>(
                std::array<const std::vector<Tensor>*, 3>{&self, &t1, &t2},
                foreach_mta::TernaryAddcmul<M>{value.to<M>()},
                "_foreach_addcmul_.cuda");
        });
    if (!launched) {
        foreach_addcmul_scalar_inplace_cuda(std::move(self), t1, t2, value);
        return;
    }
    mta_bump(self);
}

void foreach_addcdiv_scalar_mta_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& t1,
        const std::vector<Tensor>& t2, Scalar value) {
    if (!mta_ready(self) || !foreach_mta::eligible_ternary(self, t1, t2)) {
        foreach_addcdiv_scalar_inplace_cuda(std::move(self), t1, t2, value);
        return;
    }
    const bool launched = foreach_mta::dispatch_dtype(
        self[0].dtype(), [&]<typename T, typename M>() {
            foreach_mta::launch<3, 0, T, M>(
                std::array<const std::vector<Tensor>*, 3>{&self, &t1, &t2},
                foreach_mta::TernaryAddcdiv<M>{value.to<M>()},
                "_foreach_addcdiv_.cuda");
        });
    if (!launched) {
        foreach_addcdiv_scalar_inplace_cuda(std::move(self), t1, t2, value);
        return;
    }
    mta_bump(self);
}

#define TP_MTA_TERNARY_SCALAR_LIST_INPLACE(NAME, FUNCTOR)                     \
void foreach_##NAME##_scalar_list_mta_inplace_cuda(                           \
        std::vector<Tensor> self, const std::vector<Tensor>& t1,              \
        const std::vector<Tensor>& t2, const std::vector<Scalar>& scalars) {   \
    if (!mta_ready(self) ||                                                    \
        !foreach_mta::eligible_ternary(self, t1, t2) ||                        \
        self.size() != scalars.size()) {                                       \
        foreach_##NAME##_scalar_list_inplace_cuda(                             \
            std::move(self), t1, t2, scalars);                                 \
        return;                                                                \
    }                                                                          \
    const bool launched = foreach_mta::dispatch_dtype(                         \
        self[0].dtype(), [&]<typename T, typename M>() {                       \
            const std::vector<M> values = mta_scalar_values<M>(scalars);       \
            foreach_mta::launch_scalar_list<3, 0, T, M>(                       \
                std::array<const std::vector<Tensor>*, 3>{&self, &t1, &t2},    \
                values, FUNCTOR<M>{},                                          \
                "_foreach_" #NAME "_.ScalarList.cuda");                      \
        });                                                                    \
    if (!launched) {                                                            \
        foreach_##NAME##_scalar_list_inplace_cuda(                              \
            std::move(self), t1, t2, scalars);                                  \
        return;                                                                 \
    }                                                                           \
    mta_bump(self);                                                             \
}

TP_MTA_TERNARY_SCALAR_LIST_INPLACE(addcmul, foreach_mta::TernaryAddcmulScalarList)
TP_MTA_TERNARY_SCALAR_LIST_INPLACE(addcdiv, foreach_mta::TernaryAddcdivScalarList)
#undef TP_MTA_TERNARY_SCALAR_LIST_INPLACE

void foreach_lerp_scalar_list_mta_inplace_cuda(
        std::vector<Tensor> self, const std::vector<Tensor>& end,
        const std::vector<Scalar>& weights) {
    if (!mta_ready(self) || !foreach_mta::eligible_pair(self, end) ||
        self.size() != weights.size()) {
        foreach_lerp_scalar_list_inplace_cuda(std::move(self), end, weights);
        return;
    }
    const bool launched = foreach_mta::dispatch_dtype(
        self[0].dtype(), [&]<typename T, typename M>() {
            const std::vector<M> values = mta_scalar_values<M>(weights);
            foreach_mta::launch_scalar_list<2, 0, T, M>(
                std::array<const std::vector<Tensor>*, 2>{&self, &end},
                values, foreach_mta::BinaryLerpScalarList<M>{},
                "_foreach_lerp_.ScalarList.cuda");
        });
    if (!launched) {
        foreach_lerp_scalar_list_inplace_cuda(std::move(self), end, weights);
        return;
    }
    mta_bump(self);
}


std::vector<Tensor> foreach_sub_scalar_mta_ret_cuda(
        const std::vector<Tensor>& self, Scalar s) {
    if (!mta_ready(self) || !foreach_mta::eligible_list(self)) {
        return foreach_sub_scalar_cuda(self, s);
    }
    std::vector<Tensor> out = foreach_alloc_like_cuda(self);
    const bool launched = foreach_mta::dispatch_dtype(
        self[0].dtype(), [&]<typename T, typename M>() {
            foreach_mta::launch<2, 1, T, M>(
                std::array<const std::vector<Tensor>*, 2>{&self, &out},
                foreach_mta::BinarySubScalar<M>{s.to<M>()},
                "_foreach_sub.cuda");
        });
    if (!launched) return foreach_sub_scalar_cuda(self, s);
    return out;
}

void foreach_sub_list_mta_inplace_cuda(std::vector<Tensor> self,
                                       const std::vector<Tensor>& other,
                                       Scalar alpha) {
    if (!mta_ready(self) || !foreach_mta::eligible_pair(self, other)) {
        foreach_sub_list_inplace_cuda(std::move(self), other, alpha);
        return;
    }
    const bool launched = foreach_mta::dispatch_dtype(
        self[0].dtype(), [&]<typename T, typename M>() {
            foreach_mta::launch<2, 0, T, M>(
                std::array<const std::vector<Tensor>*, 2>{&self, &other},
                foreach_mta::BinarySubList<M>{alpha.to<M>()},
                "_foreach_sub_.list.cuda");
        });
    if (!launched) {
        foreach_sub_list_inplace_cuda(std::move(self), other, alpha);
        return;
    }
    mta_bump(self);
}

// ---- returning variants: allocate once, write through MTA --------------

#define TP_MTA_SCALAR_RET(NAME, FUNCTOR)                                      \
std::vector<Tensor> foreach_##NAME##_scalar_mta_ret_cuda(                     \
        const std::vector<Tensor>& self, Scalar s) {                          \
    if (!mta_ready(self) || !foreach_mta::eligible_list(self)) {              \
        return foreach_##NAME##_scalar_cuda(self, s);                         \
    }                                                                         \
    std::vector<Tensor> out = foreach_alloc_like_cuda(self);                  \
    const bool launched = foreach_mta::dispatch_dtype(                        \
        self[0].dtype(), [&]<typename T, typename M>() {                      \
            foreach_mta::launch<2, 1, T, M>(                                  \
                std::array<const std::vector<Tensor>*, 2>{&self, &out},       \
                FUNCTOR<M>{s.to<M>()}, "_foreach_" #NAME ".cuda");            \
        });                                                                   \
    if (!launched) {                                                          \
        return foreach_##NAME##_scalar_cuda(self, s);                         \
    }                                                                         \
    return out;                                                               \
}

TP_MTA_SCALAR_RET(add, foreach_mta::BinaryAddScalar)
TP_MTA_SCALAR_RET(mul, foreach_mta::BinaryMulScalar)
TP_MTA_SCALAR_RET(div, foreach_mta::BinaryDivScalar)
TP_MTA_SCALAR_RET(pow, foreach_mta::UnaryPow)
TP_MTA_SCALAR_RET(clamp_min, foreach_mta::BinaryMaximum)
TP_MTA_SCALAR_RET(clamp_max, foreach_mta::BinaryMinimum)
TP_MTA_SCALAR_RET(maximum, foreach_mta::BinaryMaximum)
TP_MTA_SCALAR_RET(minimum, foreach_mta::BinaryMinimum)
#undef TP_MTA_SCALAR_RET

#define TP_MTA_UNARY_RET(NAME, FUNCTOR)                                       \
std::vector<Tensor> foreach_##NAME##_mta_ret_cuda(                            \
        const std::vector<Tensor>& self) {                                   \
    if (!mta_ready(self) || !foreach_mta::eligible_list(self)) {              \
        return foreach_##NAME##_cuda(self);                                  \
    }                                                                         \
    std::vector<Tensor> out = foreach_alloc_like_cuda(self);                  \
    const bool launched = foreach_mta::dispatch_dtype(                        \
        self[0].dtype(), [&]<typename T, typename M>() {                      \
            foreach_mta::launch<2, 1, T, M>(                                  \
                std::array<const std::vector<Tensor>*, 2>{&self, &out},       \
                FUNCTOR<M>{}, "_foreach_" #NAME ".cuda");                   \
        });                                                                   \
    if (!launched) return foreach_##NAME##_cuda(self);                        \
    return out;                                                               \
}

TP_MTA_UNARY_RET(sqrt, foreach_mta::UnarySqrt)
TP_MTA_UNARY_RET(rsqrt, foreach_mta::UnaryRsqrt)
TP_MTA_UNARY_RET(neg, foreach_mta::UnaryNeg)
TP_MTA_UNARY_RET(abs, foreach_mta::UnaryAbs)
TP_MTA_UNARY_RET(sign, foreach_mta::UnarySign)
TP_MTA_UNARY_RET(reciprocal, foreach_mta::UnaryReciprocal)
TP_MTA_UNARY_RET(acos, foreach_mta::UnaryAcos)
TP_MTA_UNARY_RET(asin, foreach_mta::UnaryAsin)
TP_MTA_UNARY_RET(atan, foreach_mta::UnaryAtan)
TP_MTA_UNARY_RET(ceil, foreach_mta::UnaryCeil)
TP_MTA_UNARY_RET(cos, foreach_mta::UnaryCos)
TP_MTA_UNARY_RET(cosh, foreach_mta::UnaryCosh)
TP_MTA_UNARY_RET(erf, foreach_mta::UnaryErf)
TP_MTA_UNARY_RET(erfc, foreach_mta::UnaryErfc)
TP_MTA_UNARY_RET(exp, foreach_mta::UnaryExp)
TP_MTA_UNARY_RET(expm1, foreach_mta::UnaryExpm1)
TP_MTA_UNARY_RET(floor, foreach_mta::UnaryFloor)
TP_MTA_UNARY_RET(frac, foreach_mta::UnaryFrac)
TP_MTA_UNARY_RET(lgamma, foreach_mta::UnaryLgamma)
TP_MTA_UNARY_RET(log, foreach_mta::UnaryLog)
TP_MTA_UNARY_RET(log10, foreach_mta::UnaryLog10)
TP_MTA_UNARY_RET(log1p, foreach_mta::UnaryLog1p)
TP_MTA_UNARY_RET(log2, foreach_mta::UnaryLog2)
TP_MTA_UNARY_RET(round, foreach_mta::UnaryRound)
TP_MTA_UNARY_RET(sigmoid, foreach_mta::UnarySigmoid)
TP_MTA_UNARY_RET(sin, foreach_mta::UnarySin)
TP_MTA_UNARY_RET(sinh, foreach_mta::UnarySinh)
TP_MTA_UNARY_RET(tan, foreach_mta::UnaryTan)
TP_MTA_UNARY_RET(tanh, foreach_mta::UnaryTanh)
TP_MTA_UNARY_RET(trunc, foreach_mta::UnaryTrunc)
#undef TP_MTA_UNARY_RET

std::vector<Tensor> foreach_zero_mta_ret_cuda(const std::vector<Tensor>& self) {
    if (!mta_ready(self) || !foreach_mta::eligible_list(self)) {
        return foreach_zero_cuda(self);
    }
    std::vector<Tensor> out = foreach_alloc_like_cuda(self);
    const bool launched = foreach_mta::dispatch_dtype(
        self[0].dtype(), [&]<typename T, typename M>() {
            foreach_mta::launch<2, 1, T, M>(
                std::array<const std::vector<Tensor>*, 2>{&self, &out},
                foreach_mta::UnaryZero<M>{}, "_foreach_zero.cuda");
        });
    if (!launched) return foreach_zero_cuda(self);
    return out;
}

#define TP_MTA_LIST_RET(NAME, FUNCTOR)                                        \
std::vector<Tensor> foreach_##NAME##_list_mta_ret_cuda(                       \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other) {  \
    if (!mta_ready(self) || !foreach_mta::eligible_pair(self, other)) {       \
        return foreach_##NAME##_list_cuda(self, other);                      \
    }                                                                         \
    std::vector<Tensor> out = foreach_alloc_like_cuda(self);                  \
    const bool launched = foreach_mta::dispatch_dtype(                        \
        self[0].dtype(), [&]<typename T, typename M>() {                      \
            foreach_mta::launch<3, 2, T, M>(                                  \
                std::array<const std::vector<Tensor>*, 3>{&self, &other, &out},\
                FUNCTOR<M>{}, "_foreach_" #NAME ".List.cuda");              \
        });                                                                   \
    if (!launched) return foreach_##NAME##_list_cuda(self, other);             \
    return out;                                                               \
}

TP_MTA_LIST_RET(maximum, foreach_mta::BinaryMaximumList)
TP_MTA_LIST_RET(minimum, foreach_mta::BinaryMinimumList)
TP_MTA_LIST_RET(clamp_min, foreach_mta::BinaryMaximumList)
TP_MTA_LIST_RET(clamp_max, foreach_mta::BinaryMinimumList)
TP_MTA_LIST_RET(pow, foreach_mta::BinaryPowList)
#undef TP_MTA_LIST_RET

#define TP_MTA_LIST_ALPHA_RET(NAME, FUNCTOR)                                  \
std::vector<Tensor> foreach_##NAME##_list_mta_ret_cuda(                       \
        const std::vector<Tensor>& self, const std::vector<Tensor>& other,    \
        Scalar alpha) {                                                       \
    if (!mta_ready(self) || !foreach_mta::eligible_pair(self, other)) {       \
        return foreach_##NAME##_list_cuda(self, other, alpha);                \
    }                                                                         \
    std::vector<Tensor> out = foreach_alloc_like_cuda(self);                  \
    const bool launched = foreach_mta::dispatch_dtype(                        \
        self[0].dtype(), [&]<typename T, typename M>() {                      \
            foreach_mta::launch<3, 2, T, M>(                                  \
                std::array<const std::vector<Tensor>*, 3>{&self, &other, &out},\
                FUNCTOR<M>{alpha.to<M>()},                                    \
                "_foreach_" #NAME ".List.cuda");                             \
        });                                                                   \
    if (!launched) return foreach_##NAME##_list_cuda(self, other, alpha);      \
    return out;                                                               \
}

TP_MTA_LIST_ALPHA_RET(add, foreach_mta::BinaryAddList)
TP_MTA_LIST_ALPHA_RET(sub, foreach_mta::BinarySubList)
#undef TP_MTA_LIST_ALPHA_RET

#define TP_MTA_SCALAR_LIST_RET(NAME, FUNCTOR)                                  \
std::vector<Tensor> foreach_##NAME##_scalar_list_mta_ret_cuda(                \
        const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) { \
    if (!mta_ready(self) || !foreach_mta::eligible_list(self) ||             \
        self.size() != scalars.size()) {                                      \
        return foreach_##NAME##_scalar_list_cuda(self, scalars);              \
    }                                                                         \
    std::vector<Tensor> out = foreach_alloc_like_cuda(self);                  \
    const bool launched = foreach_mta::dispatch_dtype(                        \
        self[0].dtype(), [&]<typename T, typename M>() {                      \
            const std::vector<M> values = mta_scalar_values<M>(scalars);      \
            foreach_mta::launch_scalar_list<2, 1, T, M>(                       \
                std::array<const std::vector<Tensor>*, 2>{&self, &out},       \
                values, FUNCTOR<M>{}, "_foreach_" #NAME ".ScalarList.cuda");\
        });                                                                   \
    if (!launched) return foreach_##NAME##_scalar_list_cuda(self, scalars);   \
    return out;                                                               \
}

TP_MTA_SCALAR_LIST_RET(clamp_min, foreach_mta::BinaryMaximumScalarList)
TP_MTA_SCALAR_LIST_RET(clamp_max, foreach_mta::BinaryMinimumScalarList)
TP_MTA_SCALAR_LIST_RET(add, foreach_mta::BinaryAddScalarList)
TP_MTA_SCALAR_LIST_RET(sub, foreach_mta::BinarySubScalarList)
TP_MTA_SCALAR_LIST_RET(mul, foreach_mta::BinaryMulScalarList)
TP_MTA_SCALAR_LIST_RET(div, foreach_mta::BinaryDivScalarList)
TP_MTA_SCALAR_LIST_RET(pow, foreach_mta::UnaryPowScalarList)
TP_MTA_SCALAR_LIST_RET(maximum, foreach_mta::BinaryMaximumScalarList)
TP_MTA_SCALAR_LIST_RET(minimum, foreach_mta::BinaryMinimumScalarList)
#undef TP_MTA_SCALAR_LIST_RET

}  // namespace cuda
}  // namespace tensorplay
