#ifdef USE_VULKAN

#include "Common.h"

#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <optional>
#include <string>
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

// A ternary walk pairs two partners with every tensor, and a fold may carry one
// more count per position beside them.  Each side names the same number of
// tensors or the tail of the walk has no partner to fold.
template <typename Fn>
std::vector<Tensor> walk_triples(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& tensor1,
    const std::vector<Tensor>& tensor2,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == tensor1.size() && self.size() == tensor2.size(),
      "foreach ternary tensor lists must have the same length");
  std::vector<Tensor> result;
  result.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    result.push_back(each(self[i], tensor1[i], tensor2[i]));
  }
  return result;
}

template <typename Fn>
void walk_triples_inplace(
    std::vector<Tensor>& self,
    const std::vector<Tensor>& tensor1,
    const std::vector<Tensor>& tensor2,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == tensor1.size() && self.size() == tensor2.size(),
      "foreach ternary tensor lists must have the same length");
  for (size_t i = 0; i < self.size(); ++i) {
    each(self[i], tensor1[i], tensor2[i]);
  }
}

template <typename Fn>
std::vector<Tensor> walk_triple_counts(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& tensor1,
    const std::vector<Tensor>& tensor2,
    const std::vector<Scalar>& scalars,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == tensor1.size() && self.size() == tensor2.size()
          && self.size() == scalars.size(),
      "foreach ternary tensor/scalar lists must have the same length");
  std::vector<Tensor> result;
  result.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    result.push_back(each(self[i], tensor1[i], tensor2[i], scalars[i]));
  }
  return result;
}

template <typename Fn>
void walk_triple_counts_inplace(
    std::vector<Tensor>& self,
    const std::vector<Tensor>& tensor1,
    const std::vector<Tensor>& tensor2,
    const std::vector<Scalar>& scalars,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == tensor1.size() && self.size() == tensor2.size()
          && self.size() == scalars.size(),
      "foreach ternary tensor/scalar lists must have the same length");
  for (size_t i = 0; i < self.size(); ++i) {
    each(self[i], tensor1[i], tensor2[i], scalars[i]);
  }
}

// A partner that arrives beside a count per position: the walk pairs the two
// tensor lists and names one count for each pair.
template <typename Fn>
std::vector<Tensor> walk_pair_counts(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& other,
    const std::vector<Scalar>& scalars,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == other.size() && self.size() == scalars.size(),
      "foreach tensor/scalar lists must have the same length");
  std::vector<Tensor> result;
  result.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    result.push_back(each(self[i], other[i], scalars[i]));
  }
  return result;
}

template <typename Fn>
void walk_pair_counts_inplace(
    std::vector<Tensor>& self,
    const std::vector<Tensor>& other,
    const std::vector<Scalar>& scalars,
    Fn&& each) {
  TP_CHECK_VALUE(
      self.size() == other.size() && self.size() == scalars.size(),
      "foreach tensor/scalar lists must have the same length");
  for (size_t i = 0; i < self.size(); ++i) {
    each(self[i], other[i], scalars[i]);
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

// A weight held as one payload for a whole group names a count per tensor
// rather than one count shared, and the walk has to know each of them before
// it starts pairing tensors.  The host keeps such a payload: it is read as a
// contiguous rank-one run whose length is the length of the group.
std::vector<Scalar> group_counts(
    const Tensor& scalars,
    size_t expected_length) {
  if (scalars.device().type() != DeviceType::CPU) {
    TP_THROW(
        RuntimeError,
        "Expected scalars to be on CPU, got " + scalars.device().toString() +
            " instead.");
  }
  if (!scalars.is_contiguous()) {
    TP_THROW(RuntimeError, "Expected scalars to be contiguous.");
  }
  if (scalars.dim() != 1) {
    TP_THROW(
        RuntimeError,
        "Expected packed scalar Tensor to be of dimension 1. Got " +
            std::to_string(scalars.dim()) + " instead.");
  }
  if (scalars.size(0) != static_cast<int64_t>(expected_length)) {
    TP_THROW(
        RuntimeError,
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

// The entries one ternary walk answers: two partners fold into every tensor,
// weighed by the count the walk carries, by a count per tensor, or by the run
// of counts held as one payload.  Each entry rides the fold a lone tensor
// reaches, so the arithmetic order stays the one that fold already keeps.
#define TP_VK_FOREACH_TERNARY(NAME, OP)                                                \
  std::vector<Tensor> foreach_##NAME##_scalar_kernel(                                  \
      const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,             \
      const std::vector<Tensor>& tensor2, const Scalar& value) {                       \
    return walk_triples(                                                               \
        self, tensor1, tensor2, [&](const Tensor& base, const Tensor& left,             \
                                   const Tensor& right) {                              \
          return tpx::ops::OP(base, left, right, value);                               \
        });                                                                            \
  }                                                                                    \
  std::vector<Tensor> foreach_##NAME##_counts_kernel(                                  \
      const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,             \
      const std::vector<Tensor>& tensor2,                                              \
      const std::vector<Scalar>& scalars) {                                            \
    return walk_triple_counts(                                                         \
        self, tensor1, tensor2, scalars,                                               \
        [&](const Tensor& base, const Tensor& left, const Tensor& right,                \
            Scalar weight) { return tpx::ops::OP(base, left, right, weight); });        \
  }                                                                                    \
  std::vector<Tensor> foreach_##NAME##_packed_kernel(                                  \
      const std::vector<Tensor>& self, const std::vector<Tensor>& tensor1,             \
      const std::vector<Tensor>& tensor2, const Tensor& scalars) {                     \
    return foreach_##NAME##_counts_kernel(                                             \
        self, tensor1, tensor2, group_counts(scalars, self.size()));                   \
  }                                                                                    \
  void foreach_##NAME##_scalar_inplace_kernel(                                         \
      std::vector<Tensor> self, const std::vector<Tensor>& tensor1,                    \
      const std::vector<Tensor>& tensor2, const Scalar& value) {                       \
    walk_triples_inplace(                                                              \
        self, tensor1, tensor2, [&](Tensor& base, const Tensor& left,                   \
                                    const Tensor& right) {                             \
          tpx::ops::OP##_(base, left, right, value);                                   \
        });                                                                            \
  }                                                                                    \
  void foreach_##NAME##_counts_inplace_kernel(                                         \
      std::vector<Tensor> self, const std::vector<Tensor>& tensor1,                    \
      const std::vector<Tensor>& tensor2, const std::vector<Scalar>& scalars) {        \
    walk_triple_counts_inplace(                                                        \
        self, tensor1, tensor2, scalars,                                               \
        [&](Tensor& base, const Tensor& left, const Tensor& right,                      \
            Scalar weight) { tpx::ops::OP##_(base, left, right, weight); });            \
  }                                                                                    \
  void foreach_##NAME##_packed_inplace_kernel(                                         \
      std::vector<Tensor> self, const std::vector<Tensor>& tensor1,                    \
      const std::vector<Tensor>& tensor2, const Tensor& scalars) {                     \
    foreach_##NAME##_counts_inplace_kernel(                                            \
        self, tensor1, tensor2, group_counts(scalars, self.size()));                   \
  }

// A payload slid toward a partner by a weight: the weight is one count the
// whole group shares, a count named per tensor, or a partner of its own.
std::vector<Tensor> foreach_lerp_scalar_kernel(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& end,
    const Scalar& weight) {
  return walk_pairs(self, end, [&](const Tensor& base, const Tensor& stop) {
    return tpx::ops::lerp(base, stop, weight);
  });
}

std::vector<Tensor> foreach_lerp_counts_kernel(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& end,
    const std::vector<Scalar>& weight) {
  return walk_pair_counts(
      self, end, weight,
      [&](const Tensor& base, const Tensor& stop, Scalar count) {
        return tpx::ops::lerp(base, stop, count);
      });
}

std::vector<Tensor> foreach_lerp_list_kernel(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& end,
    const std::vector<Tensor>& weight) {
  TP_CHECK_VALUE(
      self.size() == end.size() && self.size() == weight.size(),
      "foreach lerp lists must have the same length");
  std::vector<Tensor> result;
  result.reserve(self.size());
  for (size_t i = 0; i < self.size(); ++i) {
    result.push_back(tpx::ops::lerp(self[i], end[i], weight[i]));
  }
  return result;
}

void foreach_lerp_scalar_inplace_kernel(
    std::vector<Tensor> self,
    const std::vector<Tensor>& end,
    const Scalar& weight) {
  walk_pairs_inplace(self, end, [&](Tensor& base, const Tensor& stop) {
    tpx::ops::lerp_(base, stop, weight);
  });
}

void foreach_lerp_counts_inplace_kernel(
    std::vector<Tensor> self,
    const std::vector<Tensor>& end,
    const std::vector<Scalar>& weight) {
  walk_pair_counts_inplace(
      self, end, weight,
      [&](Tensor& base, const Tensor& stop, Scalar count) {
        tpx::ops::lerp_(base, stop, count);
      });
}

void foreach_lerp_list_inplace_kernel(
    std::vector<Tensor> self,
    const std::vector<Tensor>& end,
    const std::vector<Tensor>& weight) {
  TP_CHECK_VALUE(
      self.size() == end.size() && self.size() == weight.size(),
      "foreach lerp lists must have the same length");
  for (size_t i = 0; i < self.size(); ++i) {
    tpx::ops::lerp_(self[i], end[i], weight[i]);
  }
}

// The group that moves instead of folding: a payload carried into the lanes its
// partner names, a fresh copy of every tensor, and a group set to nothing.  A
// move keeps the width of the destination and converts the way in, so a partner
// held in another element type answers beside it; a fresh copy keeps the shape
// and the width the tensor already had.  The route a group takes is decided
// before any lane is written, so a side naming fewer tensors is refused.
std::vector<Tensor> foreach_clone_kernel(
    const std::vector<Tensor>& self,
    std::optional<int64_t> memory_format) {
  return walk_each(self, [&](const Tensor& value) {
    return tpx::ops::clone(value, memory_format);
  });
}

std::vector<Tensor> foreach_copy_kernel(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& src,
    bool non_blocking) {
  return walk_pairs(self, src, [&](const Tensor& destination,
                                   const Tensor& source) {
    // The answer is the destination carried over, not the partner: a fresh copy
    // keeps the width and the shape the group was asked to answer in, and the
    // partner is written into it, converting on the way in.
    Tensor fresh = tpx::ops::clone(destination);
    tpx::ops::copy_(fresh, source, non_blocking);
    return fresh;
  });
}

void foreach_copy_inplace_kernel(
    std::vector<Tensor> self,
    const std::vector<Tensor>& src,
    bool non_blocking) {
  walk_pairs_inplace(
      self, src, [&](Tensor& destination, const Tensor& source) {
        tpx::ops::copy_(destination, source, non_blocking);
      });
}

std::vector<Tensor> foreach_zero_kernel(const std::vector<Tensor>& self) {
  return walk_each(self, [](const Tensor& value) {
    Tensor fresh = tpx::ops::clone(value);
    tpx::ops::zero_(fresh);
    return fresh;
  });
}

void foreach_zero_inplace_kernel(std::vector<Tensor> self) {
  walk_each_inplace(self, [](Tensor& value) { tpx::ops::zero_(value); });
}

// The group folded down to one lane per tensor: the largest lane a tensor holds,
// the length of a tensor, and the sum a length is built from.  A group may be
// told to hold its payloads in another width before the fold starts, which is how
// a run of counts is totalled without the walk overflowing the word it counts in.
std::vector<Tensor> foreach_max_kernel(const std::vector<Tensor>& self) {
  return walk_each(self, [](const Tensor& value) {
    return tpx::ops::amax(value);
  });
}

std::vector<Tensor> foreach_norm_kernel(
    const std::vector<Tensor>& self,
    const Scalar& ord,
    std::optional<DType> dtype) {
  return walk_each(self, [&](const Tensor& value) {
    Tensor held = dtype.has_value() ? tpx::ops::to(value, *dtype) : value;
    return tpx::ops::norm(held, ord.to<double>());
  });
}

std::vector<Tensor> foreach_powsum_kernel(
    const std::vector<Tensor>& self,
    const Scalar& ord,
    std::optional<DType> dtype) {
  return walk_each(self, [&](const Tensor& value) {
    Tensor held = dtype.has_value() ? tpx::ops::to(value, *dtype) : value;
    return tpx::ops::sum(tpx::ops::pow(tpx::ops::abs(held), ord));
  });
}

// A group of products: every tensor is the left hand of a product of its own and
// the partner list names the right hand it is multiplied by, so the shapes one
// tensor pairs under need not agree with the shapes another does.
std::vector<Tensor> foreach_mm_kernel(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& mat2) {
  return walk_pairs(self, mat2, [](const Tensor& left, const Tensor& right) {
    return tpx::ops::mm(left, right);
  });
}

// A group of gradients that were scaled before the backward pass: every tensor
// is scanned for a lane that is not a number, and the flag is raised on the
// device so the scan costs no round trip.  Every tensor is then divided by the
// scale, including the group that raised the flag, since the caller discards
// the gradients once the flag is up.  The scan reads a lane before the division
// touches it, so a scale that turns a finite lane into a non-finite one does
// not raise the flag by itself.
void amp_foreach_non_finite_check_and_unscale_kernel(
    std::vector<Tensor> self, Tensor& found_inf, const Tensor& inv_scale) {
  if (self.empty()) return;
  if (found_inf.numel() != 1 || inv_scale.numel() != 1) {
    TP_THROW(
        RuntimeError,
        "_amp_foreach_non_finite_check_and_unscale_: found_inf and inv_scale "
        "must be 1-element tensors");
  }
  if (found_inf.dtype() != DType::Float32 ||
      inv_scale.dtype() != DType::Float32) {
    TP_THROW(
        RuntimeError,
        "_amp_foreach_non_finite_check_and_unscale_: found_inf and inv_scale "
        "must be float32");
  }
  switch (self.front().dtype()) {
    case DType::Float32:
    case DType::Float16:
      break;
    default:
      TP_THROW(
          NotImplementedError,
          "_amp_foreach_non_finite_check_and_unscale_: unsupported dtype");
  }

  Tensor failed = tpx::ops::any(
      tpx::ops::logical_not(tpx::ops::isfinite(self.front())));
  for (size_t i = 1; i < self.size(); ++i) {
    failed = tpx::ops::logical_or(
        failed,
        tpx::ops::any(tpx::ops::logical_not(tpx::ops::isfinite(self[i]))));
  }
  tpx::ops::copy_(found_inf, tpx::ops::where(failed, 1.0, found_inf));
  for (Tensor& value : self) {
    tpx::ops::copy_(value, tpx::ops::mul(value, inv_scale));
  }
}

// A step ends by turning the flag the gradient scan raised into a new loss
// scale.  The update is a state machine over two one-lane tensors - the scale
// and the count of clean steps since it last changed - and both stay on the
// device so no branch needs the flag or the count on the host.  A raised flag
// backs the scale off and clears the count; a clean step counts up, and the
// step that reaches the interval grows the scale once and clears the count.
// A growth that would leave the largest finite single-precision value keeps
// the previous scale instead.  The flag is read as a truth, so any lane that
// is not zero, including a not-a-number, counts as raised.
Tensor& amp_update_scale_kernel(
    Tensor& self,
    Tensor& growth_tracker,
    const Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval) {
  if (!self.device().is_vulkan() || !growth_tracker.device().is_vulkan() ||
      !found_inf.device().is_vulkan()) {
    TP_THROW(
        RuntimeError,
        "_amp_update_scale_: every tensor must be a Vulkan tensor");
  }
  if (self.numel() != 1 || growth_tracker.numel() != 1 ||
      found_inf.numel() != 1) {
    TP_THROW(
        RuntimeError,
        "_amp_update_scale_: self, growth_tracker and found_inf must be "
        "1-element tensors");
  }
  if (self.dtype() != DType::Float32 || found_inf.dtype() != DType::Float32) {
    TP_THROW(
        RuntimeError, "_amp_update_scale_: self and found_inf must be float32");
  }
  if (growth_tracker.dtype() != DType::Int32) {
    TP_THROW(
        RuntimeError, "_amp_update_scale_: growth_tracker must be int32");
  }

  const Tensor failed = tpx::ops::ne(found_inf, 0.0);
  const Tensor incremented = tpx::ops::add(growth_tracker, 1);
  const Tensor reached = tpx::ops::eq(incremented, growth_interval);
  const Tensor scaled_up = tpx::ops::mul(self, growth_factor);
  const Tensor grown =
      tpx::ops::where(tpx::ops::isfinite(scaled_up), scaled_up, self);
  const Tensor backed = tpx::ops::mul(self, backoff_factor);
  const Tensor scale = tpx::ops::where(
      failed, backed, tpx::ops::where(reached, grown, self));
  const Tensor tracker = tpx::ops::where(
      tpx::ops::logical_or(failed, reached), 0, incremented);
  tpx::ops::copy_(growth_tracker, tracker);
  tpx::ops::copy_(self, scale);
  return self;
}

// An optimizer group updates every parameter from its own gradient and state,
// so the lists have to pair up by position and every entry has to hold the
// same shape, width and device as the parameter it belongs to.  The checks run
// before the first write: a state that does not match its parameter would
// otherwise be addressed through the wrong layout.
void check_optimizer_pair(
    const Tensor& param, const Tensor& grad, const char* op_name) {
  if (!param.defined() || !grad.defined()) {
    TP_THROW(
        ValueError,
        std::string(op_name) + ": parameters and gradients must be defined");
  }
  if (!param.device().is_vulkan() || param.device() != grad.device() ||
      !param.is_contiguous() || !grad.is_contiguous() ||
      param.shape() != grad.shape() || param.dtype() != grad.dtype() ||
      param.dtype() != DType::Float32) {
    TP_THROW(
        NotImplementedError,
        std::string(op_name) +
            ": requires contiguous float32 parameter/gradient pairs with "
            "matching layout on the device");
  }
}

void check_optimizer_state(
    const Tensor& param,
    const Tensor& state,
    const char* op_name,
    const char* what) {
  if (!state.defined()) {
    TP_THROW(
        ValueError,
        std::string(op_name) + ": required optimizer state is undefined");
  }
  if (!state.is_contiguous() || state.shape() != param.shape() ||
      state.dtype() != param.dtype() || state.device() != param.device()) {
    TP_THROW(
        NotImplementedError,
        std::string(op_name) + ": " + what +
            " must match its parameter layout");
  }
}

// Descend one step per parameter: an optional weight decay folds into the
// gradient, an optional momentum buffer keeps its running value (the first
// step starts it from the gradient itself, and only then does the dampening
// factor apply), and a Nesterov step reads the gradient through that buffer.
// Every term is a walk over the whole payload, so the update stays on the
// device.
std::vector<Tensor> foreach_sgd_kernel(
    const std::vector<Tensor>& params,
    const std::vector<Tensor>& grads,
    const std::vector<Tensor>& momentum_buffers,
    double lr,
    double momentum,
    double dampening,
    double weight_decay,
    bool nesterov,
    bool first_momentum_step) {
  const size_t count = params.size();
  const bool has_momentum = momentum != 0.0;
  if (grads.size() != count ||
      ((has_momentum || !momentum_buffers.empty()) &&
       momentum_buffers.size() != count)) {
    TP_THROW(ValueError, "_foreach_sgd: tensor list sizes must match");
  }
  for (size_t i = 0; i < count; ++i) {
    check_optimizer_pair(params[i], grads[i], "_foreach_sgd");
    if (has_momentum) {
      check_optimizer_state(
          params[i], momentum_buffers[i], "_foreach_sgd", "momentum buffer");
    }
  }
  if (params.empty()) return params;

  const bool has_wd = weight_decay != 0.0;
  for (size_t i = 0; i < count; ++i) {
    Tensor param = params[i];
    Tensor grad = has_wd
        ? tpx::ops::add(grads[i], tpx::ops::mul(param, weight_decay))
        : grads[i];
    if (has_momentum) {
      Tensor buffer = momentum_buffers[i];
      const Tensor value = first_momentum_step
          ? grad
          : tpx::ops::add(
                tpx::ops::mul(buffer, momentum),
                tpx::ops::mul(grad, 1.0 - dampening));
      tpx::ops::copy_(buffer, value);
      grad = nesterov ? tpx::ops::add(grad, tpx::ops::mul(value, momentum))
                      : value;
    }
    tpx::ops::copy_(param, tpx::ops::sub(param, tpx::ops::mul(grad, lr)));
  }
  return params;
}

// Descend one Adam step per parameter: the weight decay folds into the
// gradient, the two moment averages keep their running values, the AMSGrad
// maximum keeps the larger second average, and the bias-corrected step divides
// by the corrected root of that average plus the epsilon.  The bias correction
// is one count per parameter, so its constants are folded before the walk.
std::vector<Tensor> foreach_adam_kernel(
    const std::vector<Tensor>& params,
    const std::vector<Tensor>& grads,
    const std::vector<Tensor>& exp_avgs,
    const std::vector<Tensor>& exp_avg_sqs,
    const std::vector<Tensor>& max_exp_avg_sqs,
    const std::vector<int64_t>& steps,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    bool amsgrad) {
  const size_t count = params.size();
  if (steps.size() != count ||
      exp_avgs.size() != count || exp_avg_sqs.size() != count ||
      (amsgrad && max_exp_avg_sqs.size() != count) ||
      (!max_exp_avg_sqs.empty() && max_exp_avg_sqs.size() != count)) {
    TP_THROW(
        ValueError,
        "_foreach_adam: tensor and step list sizes must match the parameter "
        "list");
  }
  for (size_t i = 0; i < count; ++i) {
    check_optimizer_pair(params[i], grads[i], "_foreach_adam");
    check_optimizer_state(
        params[i], exp_avgs[i], "_foreach_adam", "first moment");
    check_optimizer_state(
        params[i], exp_avg_sqs[i], "_foreach_adam", "second moment");
    if (amsgrad) {
      check_optimizer_state(
          params[i], max_exp_avg_sqs[i], "_foreach_adam",
          "maximum second moment");
    }
  }
  if (params.empty()) return params;

  const bool has_wd = weight_decay != 0.0;
  for (size_t i = 0; i < count; ++i) {
    Tensor param = params[i];
    Tensor first_moment = exp_avgs[i];
    Tensor second_moment = exp_avg_sqs[i];
    const double correction1 =
        1.0 - std::pow(beta1, static_cast<double>(steps[i]));
    const double correction2 =
        1.0 - std::pow(beta2, static_cast<double>(steps[i]));
    const Tensor grad = has_wd
        ? tpx::ops::add(grads[i], tpx::ops::mul(param, weight_decay))
        : grads[i];
    const Tensor first = tpx::ops::add(
        tpx::ops::mul(first_moment, beta1),
        tpx::ops::mul(grad, 1.0 - beta1));
    tpx::ops::copy_(first_moment, first);
    const Tensor second = tpx::ops::add(
        tpx::ops::mul(second_moment, beta2),
        tpx::ops::mul(tpx::ops::mul(grad, grad), 1.0 - beta2));
    tpx::ops::copy_(second_moment, second);
    Tensor carried = second;
    if (amsgrad) {
      Tensor maximum_second = max_exp_avg_sqs[i];
      carried = tpx::ops::maximum(maximum_second, second);
      tpx::ops::copy_(maximum_second, carried);
    }
    const Tensor denom = tpx::ops::add(
        tpx::ops::div(tpx::ops::sqrt(carried), std::sqrt(correction2)), eps);
    tpx::ops::copy_(
        param,
        tpx::ops::sub(
            param, tpx::ops::div(tpx::ops::mul(first, lr / correction1),
                                 denom)));
  }
  return params;
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
TP_VK_FOREACH_UNARY(erfc, erfc)
TP_VK_FOREACH_UNARY(lgamma, lgamma)
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

TP_VK_FOREACH_TERNARY(addcmul, addcmul)
TP_VK_FOREACH_TERNARY(addcdiv, addcdiv)

#undef TP_VK_FOREACH_WEIGHTED
#undef TP_VK_FOREACH_PLAIN
#undef TP_VK_FOREACH_UNARY
#undef TP_VK_FOREACH_UNARY_HELD
#undef TP_VK_FOREACH_BOUND
#undef TP_VK_FOREACH_TERNARY

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
  TP_VK_FOREACH_UNARY_LIBRARY(erfc)
  TP_VK_FOREACH_UNARY_LIBRARY(lgamma)
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

#define TP_VK_FOREACH_TERNARY_LIBRARY(NAME)                                            \
  m.impl(                                                                                \
      "_foreach_" #NAME ".Scalar",                                                        \
      &tensorplay::vulkan::ops::foreach_##NAME##_scalar_kernel);                          \
  m.impl(                                                                                \
      "_foreach_" #NAME ".ScalarList",                                                    \
      &tensorplay::vulkan::ops::foreach_##NAME##_counts_kernel);                           \
  m.impl(                                                                                \
      "_foreach_" #NAME ".Tensor",                                                        \
      &tensorplay::vulkan::ops::foreach_##NAME##_packed_kernel);                           \
  m.impl(                                                                                \
      "_foreach_" #NAME "_.Scalar",                                                       \
      &tensorplay::vulkan::ops::foreach_##NAME##_scalar_inplace_kernel);                   \
  m.impl(                                                                                \
      "_foreach_" #NAME "_.ScalarList",                                                   \
      &tensorplay::vulkan::ops::foreach_##NAME##_counts_inplace_kernel);                   \
  m.impl(                                                                                \
      "_foreach_" #NAME "_.Tensor",                                                       \
      &tensorplay::vulkan::ops::foreach_##NAME##_packed_inplace_kernel);

  TP_VK_FOREACH_TERNARY_LIBRARY(addcmul)
  TP_VK_FOREACH_TERNARY_LIBRARY(addcdiv)
#undef TP_VK_FOREACH_TERNARY_LIBRARY

  m.impl(
      "_foreach_lerp.Scalar",
      &tensorplay::vulkan::ops::foreach_lerp_scalar_kernel);
  m.impl(
      "_foreach_lerp.ScalarList",
      &tensorplay::vulkan::ops::foreach_lerp_counts_kernel);
  m.impl(
      "_foreach_lerp.List",
      &tensorplay::vulkan::ops::foreach_lerp_list_kernel);
  m.impl(
      "_foreach_lerp_.Scalar",
      &tensorplay::vulkan::ops::foreach_lerp_scalar_inplace_kernel);
  m.impl(
      "_foreach_lerp_.ScalarList",
      &tensorplay::vulkan::ops::foreach_lerp_counts_inplace_kernel);
  m.impl(
      "_foreach_lerp_.List",
      &tensorplay::vulkan::ops::foreach_lerp_list_inplace_kernel);

  m.impl(
      "_foreach_clone",
      &tensorplay::vulkan::ops::foreach_clone_kernel);
  m.impl(
      "_foreach_copy",
      &tensorplay::vulkan::ops::foreach_copy_kernel);
  m.impl(
      "_foreach_copy_",
      &tensorplay::vulkan::ops::foreach_copy_inplace_kernel);
  m.impl(
      "_foreach_zero",
      &tensorplay::vulkan::ops::foreach_zero_kernel);
  m.impl(
      "_foreach_zero_",
      &tensorplay::vulkan::ops::foreach_zero_inplace_kernel);

  m.impl(
      "_foreach_max",
      &tensorplay::vulkan::ops::foreach_max_kernel);
  m.impl(
      "_foreach_norm.Scalar",
      &tensorplay::vulkan::ops::foreach_norm_kernel);
  m.impl(
      "_foreach_powsum.Scalar",
      &tensorplay::vulkan::ops::foreach_powsum_kernel);

  m.impl(
      "_foreach_mm",
      &tensorplay::vulkan::ops::foreach_mm_kernel);

  m.impl(
      "_amp_foreach_non_finite_check_and_unscale_",
      &tensorplay::vulkan::ops::amp_foreach_non_finite_check_and_unscale_kernel);
  m.impl(
      "_amp_update_scale_", &tensorplay::vulkan::ops::amp_update_scale_kernel);
  m.impl("_foreach_sgd", &tensorplay::vulkan::ops::foreach_sgd_kernel);
  m.impl("_foreach_adam", &tensorplay::vulkan::ops::foreach_adam_kernel);

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
