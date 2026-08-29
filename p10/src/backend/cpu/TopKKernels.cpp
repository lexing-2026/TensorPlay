#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
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

template <typename T, bool HasNaN>
struct TopKCompare {
  bool largest;

  bool operator()(const std::pair<T, int64_t>& lhs,
                  const std::pair<T, int64_t>& rhs) const {
    if constexpr (HasNaN) {
      const bool lhs_nan = topk_is_nan(lhs.first);
      const bool rhs_nan = topk_is_nan(rhs.first);
      if (largest) {
        return (lhs_nan && !rhs_nan) || (lhs.first > rhs.first);
      }
      return (!lhs_nan && rhs_nan) || (lhs.first < rhs.first);
    }
    return largest ? (lhs.first > rhs.first) : (lhs.first < rhs.first);
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
      std::sort(queue.begin(), queue.begin() + k - 1, compare);
    }
  }
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

#pragma omp parallel
  {
    std::vector<elem_t> queue(static_cast<size_t>(dim_size));
#pragma omp for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t outer_index = row / inner;
      const int64_t inner_index = row % inner;
      const int64_t input_base = outer_index * dim_size * inner + inner_index;
      const int64_t output_base = outer_index * k * inner + inner_index;
      bool has_nan = false;
      for (int64_t column = 0; column < dim_size; ++column) {
        const accscalar_t value = static_cast<accscalar_t>(
            input_data[input_base + column * inner]);
        queue[static_cast<size_t>(column)] = {value, column};
        has_nan |= topk_is_nan(value);
      }
      if (has_nan) {
        select_topk(queue, k, sorted, TopKCompare<accscalar_t, true>{largest});
      } else {
        select_topk(queue, k, sorted, TopKCompare<accscalar_t, false>{largest});
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
