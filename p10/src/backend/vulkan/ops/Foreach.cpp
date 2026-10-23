#ifdef USE_VULKAN

#include "Common.h"

#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {
namespace {

// A group walk answers one entry of the single-payload table per tensor, so
// the only contract it adds is that the lists pair up by position: a side
// that names fewer tensors would leave the tail of the walk with no partner,
// and the counts have to agree before any lane is written.
template <typename Fn>
std::vector<Tensor> walk_each(const std::vector<Tensor>& self, Fn&& each) {
  std::vector<Tensor> result;
  result.reserve(self.size());
  for (const Tensor& value : self) {
    result.push_back(each(value));
  }
  return result;
}

template <typename Fn>
void walk_each_inplace(std::vector<Tensor>& self, Fn&& each) {
  for (Tensor& value : self) {
    each(value);
  }
}

template <typename Fn>
std::vector<Tensor> walk_pairs(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& other,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == other.size(),
      "foreach tensor list arguments must have the same length");
  std::vector<Tensor> result;
  result.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    result.push_back(each(self[i], other[i]));
  }
  return result;
}

template <typename Fn>
void walk_pairs_inplace(
    std::vector<Tensor>& self,
    const std::vector<Tensor>& other,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == other.size(),
      "foreach tensor list arguments must have the same length");
  for (size_t i = 0; i < self.size(); ++i) {
    each(self[i], other[i]);
  }
}

template <typename Fn>
std::vector<Tensor> walk_counts(
    const std::vector<Tensor>& self,
    const std::vector<Scalar>& scalars,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == scalars.size(),
      "foreach tensor and scalar lists must have the same length");
  std::vector<Tensor> result;
  result.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    result.push_back(each(self[i], scalars[i]));
  }
  return result;
}

template <typename Fn>
void walk_counts_inplace(
    std::vector<Tensor>& self,
    const std::vector<Scalar>& scalars,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == scalars.size(),
      "foreach tensor and scalar lists must have the same length");
  for (size_t i = 0; i < self.size(); ++i) {
    each(self[i], scalars[i]);
  }
}

// The count one group shares reaches every tensor unchanged, so it has to
// name a single lane.  Held by the host rather than by the walk, it crosses
// as the one upload it is instead of turning the whole group into a read.
Tensor group_payload(const Tensor& payload, const std::vector<Tensor>& self) {
  if (!self.empty() && payload.device() != self.front().device()) {
    return payload.to(self.front().device());
  }
  return payload;
}

Tensor group_count(
    const Tensor& scalar,
    const std::vector<Tensor>& self) {
  TP_CHECK_VALUE(
      scalar.dim() == 0 && scalar.numel() == 1,
      "scalar tensor expected to be 0 dim but it has ", scalar.dim(),
      " dimensions and ", scalar.numel(), " elements");
  return group_payload(scalar, self);
}

// The entries one weighted walk answers: a count is taken as it stands, a
// partner is folded with the weight the walk carries.
#define TP_VK_FOREACH_WEIGHTED(NAME, OP)                                             \
  std::vector<Tensor> foreach_##NAME##_scalar_kernel(                                \
      const std::vector<Tensor>& self, const Scalar& scalar) {                       \
    return walk_each(                                                                \
        self, [&](const Tensor& value) { return tpx::ops::OP(value, scalar); });     \
  }                                                                                  \
  std::vector<Tensor> foreach_##NAME##_count_kernel(                                 \
      const std::vector<Tensor>& self, const Tensor& scalar,                         \
      const Scalar& alpha) {                                                         \
    const Tensor count = group_count(scalar, self);                                  \
    return walk_each(self, [&](const Tensor& value) {                                \
      return tpx::ops::OP(value, count, alpha);                                      \
    });                                                                              \
  }                                                                                  \
  std::vector<Tensor> foreach_##NAME##_list_kernel(                                  \
      const std::vector<Tensor>& self, const std::vector<Tensor>& other,             \
      const Scalar& alpha) {                                                         \
    return walk_pairs(self, other, [&](const Tensor& value,                          \
                                        const Tensor& partner) {                     \
      return tpx::ops::OP(value, partner, alpha);                                    \
    });                                                                              \
  }                                                                                  \
  std::vector<Tensor> foreach_##NAME##_counts_kernel(                                \
      const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {         \
    return walk_counts(self, scalars, [&](const Tensor& value, Scalar count) {        \
      return tpx::ops::OP(value, count);                                             \
    });                                                                              \
  }                                                                                  \
  void foreach_##NAME##_scalar_inplace_kernel(                                       \
      std::vector<Tensor> self, const Scalar& scalar) {                              \
    walk_each_inplace(                                                               \
        self, [&](Tensor& value) { tpx::ops::OP##_(value, scalar); });               \
  }                                                                                  \
  void foreach_##NAME##_count_inplace_kernel(                                        \
      std::vector<Tensor> self, const Tensor& scalar, const Scalar& alpha) {         \
    const Tensor count = group_count(scalar, self);                                  \
    walk_each_inplace(self,                                                          \
        [&](Tensor& value) { tpx::ops::OP##_(value, count, alpha); });               \
  }                                                                                  \
  void foreach_##NAME##_list_inplace_kernel(                                         \
      std::vector<Tensor> self, const std::vector<Tensor>& other,                    \
      const Scalar& alpha) {                                                         \
    walk_pairs_inplace(self, other,                                                  \
        [&](Tensor& value, const Tensor& partner) {                                  \
          tpx::ops::OP##_(value, partner, alpha);                                    \
        });                                                                          \
  }                                                                                  \
  void foreach_##NAME##_counts_inplace_kernel(                                       \
      std::vector<Tensor> self, const std::vector<Scalar>& scalars) {                \
    walk_counts_inplace(                                                             \
        self, scalars,                                                               \
        [&](Tensor& value, Scalar count) { tpx::ops::OP##_(value, count); });        \
  }

// The same eight entries where a partner carries no weight.
#define TP_VK_FOREACH_PLAIN(NAME, OP)                                                \
  std::vector<Tensor> foreach_##NAME##_scalar_kernel(                                \
      const std::vector<Tensor>& self, const Scalar& scalar) {                       \
    return walk_each(                                                                \
        self, [&](const Tensor& value) { return tpx::ops::OP(value, scalar); });     \
  }                                                                                  \
  std::vector<Tensor> foreach_##NAME##_count_kernel(                                 \
      const std::vector<Tensor>& self, const Tensor& scalar) {                       \
    const Tensor count = group_count(scalar, self);                                  \
    return walk_each(                                                                \
        self, [&](const Tensor& value) { return tpx::ops::OP(value, count); });      \
  }                                                                                  \
  std::vector<Tensor> foreach_##NAME##_list_kernel(                                  \
      const std::vector<Tensor>& self, const std::vector<Tensor>& other) {           \
    return walk_pairs(self, other, [&](const Tensor& value,                          \
                                        const Tensor& partner) {                     \
      return tpx::ops::OP(value, partner);                                           \
    });                                                                              \
  }                                                                                  \
  std::vector<Tensor> foreach_##NAME##_counts_kernel(                                \
      const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {         \
    return walk_counts(self, scalars, [&](const Tensor& value, Scalar count) {        \
      return tpx::ops::OP(value, count);                                             \
    });                                                                              \
  }                                                                                  \
  void foreach_##NAME##_scalar_inplace_kernel(                                       \
      std::vector<Tensor> self, const Scalar& scalar) {                              \
    walk_each_inplace(                                                               \
        self, [&](Tensor& value) { tpx::ops::OP##_(value, scalar); });               \
  }                                                                                  \
  void foreach_##NAME##_count_inplace_kernel(                                        \
      std::vector<Tensor> self, const Tensor& scalar) {                              \
    const Tensor count = group_count(scalar, self);                                  \
    walk_each_inplace(                                                               \
        self, [&](Tensor& value) { tpx::ops::OP##_(value, count); });                \
  }                                                                                  \
  void foreach_##NAME##_list_inplace_kernel(                                         \
      std::vector<Tensor> self, const std::vector<Tensor>& other) {                  \
    walk_pairs_inplace(self, other,                                                  \
        [&](Tensor& value, const Tensor& partner) {                                  \
          tpx::ops::OP##_(value, partner);                                           \
        });                                                                          \
  }                                                                                  \
  void foreach_##NAME##_counts_inplace_kernel(                                       \
      std::vector<Tensor> self, const std::vector<Scalar>& scalars) {                \
    walk_counts_inplace(                                                             \
        self, scalars,                                                               \
        [&](Tensor& value, Scalar count) { tpx::ops::OP##_(value, count); });        \
  }

// One formula over a whole group: every tensor takes the walk's own entry, so
// the answer keeps the shape and the width the formula drew.
#define TP_VK_FOREACH_UNARY(NAME, OP)                                             \
  std::vector<Tensor> foreach_##NAME##_kernel(                                      \
      const std::vector<Tensor>& self) {                                            \
    return walk_each(                                                               \
        self, [&](const Tensor& value) { return tpx::ops::OP(value); });            \
  }                                                                                 \
  void foreach_##NAME##_inplace_kernel(std::vector<Tensor> self) {                  \
    walk_each_inplace(                                                              \
        self, [&](Tensor& value) { tpx::ops::OP##_(value); });                       \
  }

// The same formula where this device writes only the fresh shape: the in-place
// entry folds the answer back over the tensor it came from rather than asking
// for a lane the walk never had.
#define TP_VK_FOREACH_UNARY_HELD(NAME, OP)                                          \
  std::vector<Tensor> foreach_##NAME##_kernel(                                      \
      const std::vector<Tensor>& self) {                                            \
    return walk_each(                                                               \
        self, [&](const Tensor& value) { return tpx::ops::OP(value); });            \
  }                                                                                 \
  void foreach_##NAME##_inplace_kernel(std::vector<Tensor> self) {                  \
    walk_each_inplace(self, [&](Tensor& value) {                                     \
      tpx::ops::copy_(value, tpx::ops::OP(value));                                  \
    });                                                                             \
  }

// A bound the whole group shares is taken from the width the payload holds; a
// bound that arrives as a partner is settled by the pair walk, which keeps the
// wider of the two (or the narrower) rather than the count itself.
#define TP_VK_FOREACH_BOUND(NAME, CLAMP, BOUND)                                    \
  std::vector<Tensor> foreach_##NAME##_scalar_kernel(                              \
      const std::vector<Tensor>& self, const Scalar& scalar) {                     \
    return walk_each(                                                              \
        self, [&](const Tensor& value) { return tpx::ops::CLAMP(value, scalar); }); \
  }                                                                                \
  std::vector<Tensor> foreach_##NAME##_counts_kernel(                              \
      const std::vector<Tensor>& self, const std::vector<Scalar>& scalars) {       \
    return walk_counts(self, scalars, [&](const Tensor& value, Scalar bound) {      \
      return tpx::ops::CLAMP(value, bound);                                        \
    });                                                                            \
  }                                                                                \
  std::vector<Tensor> foreach_##NAME##_list_kernel(                                \
      const std::vector<Tensor>& self, const std::vector<Tensor>& other) {         \
    return walk_pairs(self, other, [&](const Tensor& value,                        \
                                        const Tensor& partner) {                    \
      return tpx::ops::BOUND(value, partner);                                      \
    });                                                                            \
  }                                                                                \
  void foreach_##NAME##_scalar_inplace_kernel(                                     \
      std::vector<Tensor> self, const Scalar& scalar) {                            \
    walk_each_inplace(                                                             \
        self, [&](Tensor& value) { tpx::ops::CLAMP##_(value, scalar); });           \
  }                                                                                \
  void foreach_##NAME##_counts_inplace_kernel(                                     \
      std::vector<Tensor> self, const std::vector<Scalar>& scalars) {              \
    walk_counts_inplace(self, scalars, [&](Tensor& value, Scalar bound) {           \
      tpx::ops::CLAMP##_(value, bound);                                            \
    });                                                                            \
  }                                                                                \
  void foreach_##NAME##_list_inplace_kernel(                                       \
      std::vector<Tensor> self, const std::vector<Tensor>& other) {                \
    walk_pairs_inplace(self, other, [&](Tensor& value, const Tensor& partner) {     \
      tpx::ops::copy_(value, tpx::ops::BOUND(value, partner));                     \
    });                                                                            \
  }

// A group raised to a count.  The exponent is either one count, one count per
// tensor, or a payload of its own, and a shared base may be a count or a
// payload held beside the walk.
std::vector<Tensor> foreach_pow_scalar_kernel(
    const std::vector<Tensor>& self,
    const Scalar& exponent) {
  return walk_each(
      self, [&](const Tensor& value) { return tpx::ops::pow(value, exponent); });
}

std::vector<Tensor> foreach_pow_counts_kernel(
    const std::vector<Tensor>& self,
    const std::vector<Scalar>& exponent) {
  return walk_counts(self, exponent,
      [&](const Tensor& value, Scalar count) {
        return tpx::ops::pow(value, count);
      });
}

std::vector<Tensor> foreach_pow_list_kernel(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& exponent) {
  return walk_pairs(self, exponent, [&](const Tensor& value,
                                         const Tensor& power) {
    return tpx::ops::pow(value, power);
  });
}

std::vector<Tensor> foreach_pow_count_kernel(
    const Scalar& self,
    const std::vector<Tensor>& exponent) {
  return walk_each(exponent, [&](const Tensor& power) {
    return tpx::ops::pow(self, power);
  });
}

std::vector<Tensor> foreach_pow_base_kernel(
    const Tensor& self,
    const std::vector<Tensor>& exponent) {
  const Tensor base = group_payload(self, exponent);
  return walk_each(exponent, [&](const Tensor& power) {
    return tpx::ops::pow(base, power);
  });
}

void foreach_pow_scalar_inplace_kernel(
    std::vector<Tensor> self,
    const Scalar& exponent) {
  walk_each_inplace(
      self, [&](Tensor& value) { tpx::ops::pow_(value, exponent); });
}

void foreach_pow_counts_inplace_kernel(
    std::vector<Tensor> self,
    const std::vector<Scalar>& exponent) {
  walk_counts_inplace(self, exponent,
      [&](Tensor& value, Scalar count) { tpx::ops::pow_(value, count); });
}

void foreach_pow_list_inplace_kernel(
    std::vector<Tensor> self,
    const std::vector<Tensor>& exponent) {
  walk_pairs_inplace(self, exponent,
      [&](Tensor& value, const Tensor& power) { tpx::ops::pow_(value, power); });
}

TP_VK_FOREACH_WEIGHTED(add, add)
TP_VK_FOREACH_WEIGHTED(sub, sub)
TP_VK_FOREACH_PLAIN(mul, mul)
TP_VK_FOREACH_PLAIN(div, div)

TP_VK_FOREACH_UNARY(sqrt, sqrt)
TP_VK_FOREACH_UNARY(neg, neg)
TP_VK_FOREACH_UNARY(abs, abs)
TP_VK_FOREACH_UNARY(reciprocal, reciprocal)
TP_VK_FOREACH_UNARY(acos, acos)
TP_VK_FOREACH_UNARY(asin, asin)
TP_VK_FOREACH_UNARY(atan, atan)
TP_VK_FOREACH_UNARY(ceil, ceil)
TP_VK_FOREACH_UNARY(cos, cos)
TP_VK_FOREACH_UNARY(cosh, cosh)
TP_VK_FOREACH_UNARY(erf, erf)
TP_VK_FOREACH_UNARY(exp, exp)
TP_VK_FOREACH_UNARY(expm1, expm1)
TP_VK_FOREACH_UNARY(floor, floor)
TP_VK_FOREACH_UNARY(frac, frac)
TP_VK_FOREACH_UNARY(log, log)
TP_VK_FOREACH_UNARY(log10, log10)
TP_VK_FOREACH_UNARY(log1p, log1p)
TP_VK_FOREACH_UNARY(log2, log2)
TP_VK_FOREACH_UNARY(round, round)
TP_VK_FOREACH_UNARY(sigmoid, sigmoid)
TP_VK_FOREACH_UNARY(sin, sin)
TP_VK_FOREACH_UNARY(sinh, sinh)
TP_VK_FOREACH_UNARY(tan, tan)
TP_VK_FOREACH_UNARY(tanh, tanh)
TP_VK_FOREACH_UNARY(trunc, trunc)
TP_VK_FOREACH_UNARY_HELD(rsqrt, rsqrt)
TP_VK_FOREACH_UNARY_HELD(sign, sign)

TP_VK_FOREACH_BOUND(clamp_min, clamp_min, maximum)
TP_VK_FOREACH_BOUND(clamp_max, clamp_max, minimum)
TP_VK_FOREACH_BOUND(maximum, clamp_min, maximum)
TP_VK_FOREACH_BOUND(minimum, clamp_max, minimum)

#undef TP_VK_FOREACH_WEIGHTED
#undef TP_VK_FOREACH_PLAIN
#undef TP_VK_FOREACH_UNARY
#undef TP_VK_FOREACH_UNARY_HELD
#undef TP_VK_FOREACH_BOUND

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ForeachKernels) {
#define TP_VK_FOREACH_LIBRARY(NAME)                                                  \
  m.impl(                                                                            \
      "_foreach_" #NAME ".Scalar",                                                   \
      &tensorplay::vulkan::ops::foreach_##NAME##_scalar_kernel);                     \
  m.impl(                                                                            \
      "_foreach_" #NAME ".Tensor",                                                   \
      &tensorplay::vulkan::ops::foreach_##NAME##_count_kernel);                      \
  m.impl(                                                                            \
      "_foreach_" #NAME ".List",                                                     \
      &tensorplay::vulkan::ops::foreach_##NAME##_list_kernel);                       \
  m.impl(                                                                            \
      "_foreach_" #NAME ".ScalarList",                                               \
      &tensorplay::vulkan::ops::foreach_##NAME##_counts_kernel);                     \
  m.impl(                                                                            \
      "_foreach_" #NAME "_.Scalar",                                                  \
      &tensorplay::vulkan::ops::foreach_##NAME##_scalar_inplace_kernel);             \
  m.impl(                                                                            \
      "_foreach_" #NAME "_.Tensor",                                                  \
      &tensorplay::vulkan::ops::foreach_##NAME##_count_inplace_kernel);              \
  m.impl(                                                                            \
      "_foreach_" #NAME "_.List",                                                    \
      &tensorplay::vulkan::ops::foreach_##NAME##_list_inplace_kernel);               \
  m.impl(                                                                            \
      "_foreach_" #NAME "_.ScalarList",                                              \
      &tensorplay::vulkan::ops::foreach_##NAME##_counts_inplace_kernel);

  TP_VK_FOREACH_LIBRARY(add)
  TP_VK_FOREACH_LIBRARY(sub)
  TP_VK_FOREACH_LIBRARY(mul)
  TP_VK_FOREACH_LIBRARY(div)
#undef TP_VK_FOREACH_LIBRARY

#define TP_VK_FOREACH_UNARY_LIBRARY(NAME)                                            \
  m.impl(                                                                            \
      "_foreach_" #NAME, &tensorplay::vulkan::ops::foreach_##NAME##_kernel);          \
  m.impl(                                                                            \
      "_foreach_" #NAME "_", &tensorplay::vulkan::ops::foreach_##NAME##_inplace_kernel);

  TP_VK_FOREACH_UNARY_LIBRARY(sqrt)
  TP_VK_FOREACH_UNARY_LIBRARY(rsqrt)
  TP_VK_FOREACH_UNARY_LIBRARY(neg)
  TP_VK_FOREACH_UNARY_LIBRARY(abs)
  TP_VK_FOREACH_UNARY_LIBRARY(reciprocal)
  TP_VK_FOREACH_UNARY_LIBRARY(sign)
  TP_VK_FOREACH_UNARY_LIBRARY(acos)
  TP_VK_FOREACH_UNARY_LIBRARY(asin)
  TP_VK_FOREACH_UNARY_LIBRARY(atan)
  TP_VK_FOREACH_UNARY_LIBRARY(ceil)
  TP_VK_FOREACH_UNARY_LIBRARY(cos)
  TP_VK_FOREACH_UNARY_LIBRARY(cosh)
  TP_VK_FOREACH_UNARY_LIBRARY(erf)
  TP_VK_FOREACH_UNARY_LIBRARY(exp)
  TP_VK_FOREACH_UNARY_LIBRARY(expm1)
  TP_VK_FOREACH_UNARY_LIBRARY(floor)
  TP_VK_FOREACH_UNARY_LIBRARY(frac)
  TP_VK_FOREACH_UNARY_LIBRARY(log)
  TP_VK_FOREACH_UNARY_LIBRARY(log10)
  TP_VK_FOREACH_UNARY_LIBRARY(log1p)
  TP_VK_FOREACH_UNARY_LIBRARY(log2)
  TP_VK_FOREACH_UNARY_LIBRARY(round)
  TP_VK_FOREACH_UNARY_LIBRARY(sigmoid)
  TP_VK_FOREACH_UNARY_LIBRARY(sin)
  TP_VK_FOREACH_UNARY_LIBRARY(sinh)
  TP_VK_FOREACH_UNARY_LIBRARY(tan)
  TP_VK_FOREACH_UNARY_LIBRARY(tanh)
  TP_VK_FOREACH_UNARY_LIBRARY(trunc)
#undef TP_VK_FOREACH_UNARY_LIBRARY

#define TP_VK_FOREACH_BOUND_LIBRARY(NAME)                                                \
  m.impl(                                                                                \
      "_foreach_" #NAME ".Scalar",                                                        \
      &tensorplay::vulkan::ops::foreach_##NAME##_scalar_kernel);                          \
  m.impl(                                                                                \
      "_foreach_" #NAME ".ScalarList",                                                    \
      &tensorplay::vulkan::ops::foreach_##NAME##_counts_kernel);                          \
  m.impl(                                                                                \
      "_foreach_" #NAME ".List",                                                          \
      &tensorplay::vulkan::ops::foreach_##NAME##_list_kernel);                            \
  m.impl(                                                                                \
      "_foreach_" #NAME "_.Scalar",                                                       \
      &tensorplay::vulkan::ops::foreach_##NAME##_scalar_inplace_kernel);                  \
  m.impl(                                                                                \
      "_foreach_" #NAME "_.ScalarList",                                                   \
      &tensorplay::vulkan::ops::foreach_##NAME##_counts_inplace_kernel);                  \
  m.impl(                                                                                \
      "_foreach_" #NAME "_.List",                                                         \
      &tensorplay::vulkan::ops::foreach_##NAME##_list_inplace_kernel);

  TP_VK_FOREACH_BOUND_LIBRARY(clamp_min)
  TP_VK_FOREACH_BOUND_LIBRARY(clamp_max)
  TP_VK_FOREACH_BOUND_LIBRARY(maximum)
  TP_VK_FOREACH_BOUND_LIBRARY(minimum)
#undef TP_VK_FOREACH_BOUND_LIBRARY

  m.impl(
      "_foreach_pow.Scalar",
      &tensorplay::vulkan::ops::foreach_pow_scalar_kernel);
  m.impl(
      "_foreach_pow.ScalarList",
      &tensorplay::vulkan::ops::foreach_pow_counts_kernel);
  m.impl(
      "_foreach_pow.List",
      &tensorplay::vulkan::ops::foreach_pow_list_kernel);
  m.impl(
      "_foreach_pow.ScalarAndTensor",
      &tensorplay::vulkan::ops::foreach_pow_count_kernel);
  m.impl(
      "_foreach_pow.TensorAndTensor",
      &tensorplay::vulkan::ops::foreach_pow_base_kernel);
  m.impl(
      "_foreach_pow_.Scalar",
      &tensorplay::vulkan::ops::foreach_pow_scalar_inplace_kernel);
  m.impl(
      "_foreach_pow_.ScalarList",
      &tensorplay::vulkan::ops::foreach_pow_counts_inplace_kernel);
  m.impl(
      "_foreach_pow_.List",
      &tensorplay::vulkan::ops::foreach_pow_list_inplace_kernel);
}

#endif /* USE_VULKAN */
