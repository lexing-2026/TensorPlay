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
Tensor group_count(
    const Tensor& scalar,
    const std::vector<Tensor>& self) {
  TP_CHECK_VALUE(
      scalar.dim() == 0 && scalar.numel() == 1,
      "scalar tensor expected to be 0 dim but it has ", scalar.dim(),
      " dimensions and ", scalar.numel(), " elements");
  if (!self.empty() && scalar.device() != self.front().device()) {
    return scalar.to(self.front().device());
  }
  return scalar;
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

TP_VK_FOREACH_WEIGHTED(add, add)
TP_VK_FOREACH_WEIGHTED(sub, sub)
TP_VK_FOREACH_PLAIN(mul, mul)
TP_VK_FOREACH_PLAIN(div, div)

#undef TP_VK_FOREACH_WEIGHTED
#undef TP_VK_FOREACH_PLAIN

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
}

#endif /* USE_VULKAN */
