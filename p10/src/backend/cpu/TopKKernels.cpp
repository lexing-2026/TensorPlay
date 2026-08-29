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

template <typename T>
inline bool topk_before(T lhs, int64_t lhs_index, T rhs, int64_t rhs_index,
                        bool largest) {
  const bool lhs_nan = topk_is_nan(lhs);
  const bool rhs_nan = topk_is_nan(rhs);
  if (lhs_nan != rhs_nan) return largest ? lhs_nan : !lhs_nan;
  if (lhs_nan) return lhs_index < rhs_index;
  if (lhs < rhs) return !largest;
  if (rhs < lhs) return largest;
  return lhs_index < rhs_index;
}

template <typename T>
void topk_kernel_cpu_impl(const Tensor& input, Tensor& values, Tensor& indices,
                          int64_t k, bool largest, bool sorted,
                          int64_t outer, int64_t inner, int64_t dim_size) {
  const T* input_data = input.data_ptr<T>();
  T* value_data = values.data_ptr<T>();
  int64_t* index_data = indices.data_ptr<int64_t>();
  const int64_t rows = outer * inner;
  if (rows == 0 || k == 0) return;

#pragma omp parallel
  {
    std::vector<std::pair<T, int64_t>> queue(static_cast<size_t>(dim_size));
#pragma omp for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t outer_index = row / inner;
      const int64_t inner_index = row % inner;
      const int64_t input_base = outer_index * dim_size * inner + inner_index;
      const int64_t output_base = outer_index * k * inner + inner_index;
      for (int64_t column = 0; column < dim_size; ++column) {
        queue[static_cast<size_t>(column)] = {
            input_data[input_base + column * inner], column};
      }
      auto comparator = [largest](const auto& lhs, const auto& rhs) {
        return topk_before(lhs.first, lhs.second, rhs.first, rhs.second, largest);
      };
      if (sorted) {
        if (k <= dim_size / 64) {
          std::partial_sort(queue.begin(), queue.begin() + k, queue.end(), comparator);
        } else {
          std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(), comparator);
          std::sort(queue.begin(), queue.begin() + k, comparator);
        }
      } else {
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(), comparator);
      }
      for (int64_t column = 0; column < k; ++column) {
        value_data[output_base + column * inner] =
            queue[static_cast<size_t>(column)].first;
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
      topk_kernel_cpu_impl<uint8_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Int8:
      topk_kernel_cpu_impl<int8_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Int16:
      topk_kernel_cpu_impl<int16_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Int32:
      topk_kernel_cpu_impl<int32_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Int64:
      topk_kernel_cpu_impl<int64_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::UInt16:
      topk_kernel_cpu_impl<uint16_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::UInt32:
      topk_kernel_cpu_impl<uint32_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::UInt64:
      topk_kernel_cpu_impl<uint64_t>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Float16:
      topk_kernel_cpu_impl<Half>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::BFloat16:
      topk_kernel_cpu_impl<BFloat16>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Float32:
      topk_kernel_cpu_impl<float>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
      break;
    case DType::Float64:
      topk_kernel_cpu_impl<double>(input, values, indices, k, largest, sorted, outer, inner, dim_size);
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
