#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "QuantTensorOperators.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

template <typename T>
inline bool topk_is_nan(T value) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return std::isnan(value);
  } else if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
    return std::isnan(static_cast<float>(value));
  } else {
    return false;
  }
}

template <typename T, bool HasNaN, bool Largest>
struct TopKCompare {
  bool operator()(const std::pair<T, int64_t>& lhs,
                  const std::pair<T, int64_t>& rhs) const {
    if constexpr (HasNaN) {
      const bool lhs_nan = topk_is_nan(lhs.first);
      const bool rhs_nan = topk_is_nan(rhs.first);
      if constexpr (Largest) {
        return (lhs_nan && !rhs_nan) || (lhs.first > rhs.first);
      }
      return (!lhs_nan && rhs_nan) || (lhs.first < rhs.first);
    }
    if constexpr (Largest) {
      return lhs.first > rhs.first;
    }
    return lhs.first < rhs.first;
  }
};

template <typename T, typename Compare>
inline void select_topk(std::vector<std::pair<T, int64_t>>& queue,
                        int64_t k, bool sorted, Compare compare) {
  const int64_t dim_size = static_cast<int64_t>(queue.size());
  if (k <= dim_size / 64) {
    std::partial_sort(queue.begin(), queue.begin() + k, queue.end(), compare);
  } else {
    std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(), compare);
    if (sorted) {
      std::sort(queue.begin(), queue.begin() + k, compare);
    }
  }
}

template <typename T, bool HasNaN>
inline void select_topk_dir(std::vector<std::pair<T, int64_t>>& queue,
                            int64_t k, bool sorted, bool largest) {
  // Which end is the top is known before the queue is partitioned, so it picks
  // a comparator rather than being asked of one: a selection runs the ordering
  // hundreds of times per element, and a question asked that often is worth
  // compiling away.
  if (largest) {
    select_topk(queue, k, sorted, TopKCompare<T, HasNaN, true>{});
  } else {
    select_topk(queue, k, sorted, TopKCompare<T, HasNaN, false>{});
  }
}

// Ordering with the value rule first and the index breaking a tie, so an equal
// value keeps the earlier occurrence.  That is the order a stable sort of the row
// answers with, and the bounded selection below needs a total order to hold the
// heap up.
template <typename T, bool Largest>
inline bool topk_ahead(const std::pair<T, int64_t>& lhs,
                       const std::pair<T, int64_t>& rhs) {
  using ValueRule = TopKCompare<T, false, Largest>;
  if (ValueRule{}(lhs, rhs)) return true;
  return !ValueRule{}(rhs, lhs) && lhs.second < rhs.second;
}

template <typename T, bool Largest>
struct TopKTotalOrder {
  bool operator()(const std::pair<T, int64_t>& lhs,
                  const std::pair<T, int64_t>& rhs) const {
    return topk_ahead<T, Largest>(lhs, rhs);
  }
};

// The best k entries seen so far, held in a heap of exactly k slots whose root is
// the worst of them.  A row is read once and nothing outside those slots is
// written, which is the whole point: the full queue costs a pair per element.
// Entries arrive in ascending index order, so a newcomer can only displace the
// root on its value - a tie with the root keeps the earlier index - and the
// ordinary rejection is one compare against the root's value.
//
// A row holding a NaN reports false instead of an answer: NaN is not placed by
// value, so the arrival test would drop it quietly, and the caller folds such a
// row with the full queue.
template <typename scalar_t, typename accscalar_t, bool Largest>
bool fill_topk_heap(const scalar_t* input_data, int64_t input_base,
                    int64_t inner, int64_t dim_size,
                    std::pair<accscalar_t, int64_t>* heap, int64_t k) {
  using elem_t = std::pair<accscalar_t, int64_t>;
  auto ahead = [](const elem_t& lhs, const elem_t& rhs) {
    return topk_ahead<accscalar_t, Largest>(lhs, rhs);
  };
  auto better_value = [](accscalar_t candidate, accscalar_t worst) {
    return Largest ? candidate > worst : candidate < worst;
  };
  int64_t filled = 0;
  for (int64_t column = 0; column < dim_size; ++column) {
    const accscalar_t value =
        static_cast<accscalar_t>(input_data[input_base + column * inner]);
    if (topk_is_nan(value)) return false;
    if (filled < k) {
      heap[filled] = elem_t{value, column};
      int64_t i = filled++;
      while (i > 0) {
        const int64_t parent = (i - 1) / 2;
        // The root is the worst entry, so a parent that is better than its child
        // is the violation to repair.
        if (!ahead(heap[parent], heap[i])) break;
        std::swap(heap[parent], heap[i]);
        i = parent;
      }
      continue;
    }
    if (!better_value(value, heap[0].first)) continue;
    heap[0] = elem_t{value, column};
    int64_t i = 0;
    for (;;) {
      const int64_t left = 2 * i + 1;
      const int64_t right = left + 1;
      int64_t worst = i;
      if (left < k && ahead(heap[worst], heap[left])) worst = left;
      if (right < k && ahead(heap[worst], heap[right])) worst = right;
      if (worst == i) break;
      std::swap(heap[worst], heap[i]);
      i = worst;
    }
  }
  return filled == k;
}

template <typename scalar_t, typename accscalar_t>
void topk_kernel_cpu_impl(const Tensor& input, Tensor& values, Tensor& indices,
                          int64_t k, bool largest, bool sorted,
                          int64_t outer, int64_t inner, int64_t dim_size) {
  using elem_t = std::pair<accscalar_t, int64_t>;
  const scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* value_data = values.data_ptr<scalar_t>();
  int64_t* index_data = indices.data_ptr<int64_t>();
  const int64_t rows = outer * inner;
  if (rows == 0 || k == 0) return;
  // A row is long enough relative to k for the bounded walk to pay for itself.
  // Below that the queue is already small, and sifting a heap whose depth grows
  // with k costs more than the partition it would replace.
  const bool use_heap = k * 32 <= dim_size;

#pragma omp parallel
  {
    // Only a row that meets a NaN needs the full queue, so its pages are touched
    // on the first such row rather than by every thread up front.
    std::vector<elem_t> queue;
    std::vector<elem_t> heap;
    if (use_heap) heap.resize(static_cast<size_t>(k));
#pragma omp for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t outer_index = row / inner;
      const int64_t inner_index = row % inner;
      const int64_t input_base = outer_index * dim_size * inner + inner_index;
      const int64_t output_base = outer_index * k * inner + inner_index;
      bool row_is_clean = false;
      if (use_heap) {
        elem_t* slots = heap.data();
        row_is_clean = largest
            ? fill_topk_heap<scalar_t, accscalar_t, true>(
                  input_data, input_base, inner, dim_size, slots, k)
            : fill_topk_heap<scalar_t, accscalar_t, false>(
                  input_data, input_base, inner, dim_size, slots, k);
        if (row_is_clean) {
          // The full-queue route answers this row in order even when the caller
          // did not ask for it, because that is what a partial sort leaves behind;
          // the bounded walk keeps that answer instead of a heap order.
          if (largest) {
            std::sort(heap.begin(), heap.end(),
                      TopKTotalOrder<accscalar_t, true>{});
          } else {
            std::sort(heap.begin(), heap.end(),
                      TopKTotalOrder<accscalar_t, false>{});
          }
          for (int64_t column = 0; column < k; ++column) {
            value_data[output_base + column * inner] =
                static_cast<scalar_t>(heap[static_cast<size_t>(column)].first);
            index_data[output_base + column * inner] =
                heap[static_cast<size_t>(column)].second;
          }
          continue;
        }
      }
      if (queue.size() < static_cast<size_t>(dim_size)) {
        queue.resize(static_cast<size_t>(dim_size));
      }
      bool has_nan = false;
      for (int64_t column = 0; column < dim_size; ++column) {
        const accscalar_t value = static_cast<accscalar_t>(
            input_data[input_base + column * inner]);
        queue[static_cast<size_t>(column)] = {value, column};
        has_nan |= topk_is_nan(value);
      }
      if (has_nan) {
        select_topk_dir<accscalar_t, true>(queue, k, sorted, largest);
      } else {
        select_topk_dir<accscalar_t, false>(queue, k, sorted, largest);
      }
      for (int64_t column = 0; column < k; ++column) {
        value_data[output_base + column * inner] =
            static_cast<scalar_t>(queue[static_cast<size_t>(column)].first);
        index_data[output_base + column * inner] =
            queue[static_cast<size_t>(column)].second;
      }
    }
  }
}

} // namespace

std::tuple<Tensor, Tensor> topk_kernel_cpu(const Tensor& self, int64_t k, int64_t dim,
                                           bool largest, bool sorted, int64_t impl) {
  if (isQuantizedType(self.dtype())) {
    quantized::q_scale(self);
    auto result = topk_kernel_cpu(quantized::storage_view(self), k, dim, largest, sorted, impl);
    return {quantized::with_quantizer(std::get<0>(result), self), std::get<1>(result)};
  }
  (void)impl;
  Tensor input = self.contiguous();
  const int64_t ndim = input.dim();
  if (ndim == 0) {
    TP_CHECK(dim == 0 || dim == -1, "topk: dimension out of range");
  } else {
    if (dim < 0) dim += ndim;
    TP_CHECK(dim >= 0 && dim < ndim, "topk: dimension out of range");
  }
  const int64_t dim_size = ndim == 0 ? 1 : input.size(dim);
  if (k < 0 || k > dim_size) {
    TP_THROW(RuntimeError, "topk: k must be in [0, dimension size]");
  }

  std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
  if (ndim != 0) shape[static_cast<size_t>(dim)] = k;
  Tensor values = Tensor::empty(shape, input.dtype(), input.device());
  Tensor indices = Tensor::empty(shape, DType::Int64, input.device());
  if (k == 0 || input.numel() == 0) return {values, indices};

  int64_t outer = 1;
  int64_t inner = 1;
  for (int64_t axis = 0; axis < (ndim == 0 ? 0 : dim); ++axis) outer *= input.size(axis);
  for (int64_t axis = ndim == 0 ? 0 : dim + 1; axis < ndim; ++axis) inner *= input.size(axis);

  switch (input.dtype()) {
    case DType::UInt8:
      topk_kernel_cpu_impl<uint8_t, uint8_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Int8:
      topk_kernel_cpu_impl<int8_t, int8_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Int16:
      topk_kernel_cpu_impl<int16_t, int16_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Int32:
      topk_kernel_cpu_impl<int32_t, int32_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Int64:
      topk_kernel_cpu_impl<int64_t, int64_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::UInt16:
      topk_kernel_cpu_impl<uint16_t, uint16_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::UInt32:
      topk_kernel_cpu_impl<uint32_t, uint32_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::UInt64:
      topk_kernel_cpu_impl<uint64_t, uint64_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Float16:
      topk_kernel_cpu_impl<Half, Half>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::BFloat16:
      topk_kernel_cpu_impl<BFloat16, float>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Float32:
      topk_kernel_cpu_impl<float, float>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Float64:
      topk_kernel_cpu_impl<double, double>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Bool:
    case DType::ComplexFloat:
    case DType::ComplexDouble:
    case DType::ComplexHalf:
    case DType::BComplex32:
    case DType::Float8_e4m3fn:
    case DType::Float8_e5m2:
    case DType::Float8_e4m3fnuz:
    case DType::Float8_e5m2fnuz:
    case DType::Float8_e8m0fnu:
    case DType::Undefined:
    case DType::NumOptions:
      TP_THROW(NotImplementedError, "topk: unsupported dtype");
  }
  return {values, indices};
}

TENSORPLAY_LIBRARY_IMPL(CPU, TopKKernels) {
  m.impl("topk", topk_kernel_cpu);
}

} // namespace cpu
} // namespace tensorplay
