#pragma once

#include "TensorIterator.h"
#include "irange.h"
#include <algorithm>
#include <array>
#include <vector>

namespace tensorplay {

struct DimCounter {
  DimCounter(const DimVector& shape, Range range);

  void increment(const std::array<int64_t, 2>& step);
  bool is_done() const;
  std::array<int64_t, 2> max_2d_step() const;

  const DimVector& shape;
  Range range;
  std::vector<int64_t> values;
  int64_t offset;
};

namespace internal {

inline void get_data_ptrs(
    char** ptrs,
    const std::vector<char*>& base,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& counter) {
  const auto ntensors = base.size();
  const auto ndim = counter.size();
  std::copy(base.begin(), base.end(), ptrs);
  for (const auto dim : irange(ndim)) {
    int64_t value = counter[dim];
    for (const auto arg : irange(ntensors)) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
}

inline void serial_for_each(
    const DimVector& shape,
    const std::vector<int64_t>& strides,
    char** base_ptrs,
    size_t ntensors,
    const TensorIteratorBase::loop2d_t& loop,
    Range range) {
  const auto ndim = shape.size();
  TP_CHECK(
      strides.size() == ntensors * std::max(size_t{2}, ndim),
      "incorrect strides size");

  if (ndim <= 1) {
    if (range.begin == 0) {
      loop(base_ptrs, strides.data(), range.size(), 1);
    } else {
      std::vector<char*> ptrs(ntensors);
      std::vector<int64_t> counter = {range.begin};
      get_data_ptrs(ptrs.data(), {base_ptrs, base_ptrs + ntensors}, strides, counter);
      loop(ptrs.data(), strides.data(), range.size(), 1);
    }
  } else {
    std::vector<char*> ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    while (!counter.is_done()) {
      get_data_ptrs(
          ptrs.data(), {base_ptrs, base_ptrs + ntensors}, strides, counter.values);
      auto step = counter.max_2d_step();
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      counter.increment(step);
    }
  }
}

} // namespace internal
} // namespace tensorplay