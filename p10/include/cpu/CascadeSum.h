#pragma once

#include "TensorIterator.h"
#include "Parallel.h"
#include "cpu/vec/ReducedFloat.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace tensorplay::cpu {
inline namespace CPU_CAPABILITY {
namespace sum_detail {

using tensorplay::vec::Vectorized;

inline int64_t ceil_log2(int64_t value) {
  if (value <= 2) {
    return 1;
  }
  uint64_t remaining = static_cast<uint64_t>(value - 1);
  int64_t result = 0;
  while (remaining != 0) {
    ++result;
    remaining >>= 1;
  }
  return result;
}

template <typename T>
struct LoadPolicy {
  static constexpr int64_t memsize() {
    return sizeof(T);
  }

  static T load(const char* __restrict data, int64_t stride, int64_t index) {
    const auto* ptr = reinterpret_cast<const T*>(data + index * stride);
    return *ptr;
  }
};

template <typename T>
struct LoadPolicy<Vectorized<T>> {
  static constexpr int64_t memsize() {
    return sizeof(T) * Vectorized<T>::size();
  }

  static Vectorized<T> load(
      const char* __restrict data, int64_t stride, int64_t index) {
    return Vectorized<T>::loadu(data + index * stride);
  }
};

template <typename Scalar, typename Acc>
struct CastLoadPolicy {
  static constexpr int64_t memsize() {
    return sizeof(Scalar);
  }

  static Acc load(
      const char* __restrict data, int64_t stride, int64_t index) {
    const auto* ptr = reinterpret_cast<const Scalar*>(data + index * stride);
    return static_cast<Acc>(*ptr);
  }
};

template <typename T>
struct CastLoadPolicy<T, T> : LoadPolicy<T> {};

template <typename Scalar, typename Acc>
struct InnerSumCastLoadPolicy {
  using Vec = Vectorized<Acc>;

  static constexpr int64_t memsize() {
    return sizeof(Scalar) * 2 * Vec::size();
  }

  static Vec load(
      const char* __restrict data, int64_t stride, int64_t index) {
    const auto* ptr = reinterpret_cast<const Scalar*>(data + index * stride);
    const auto values = tensorplay::vec::load_reduced_float_pair(ptr);
    return values.first + values.second;
  }
};

template <typename T>
struct InnerSumCastLoadPolicy<T, T> : LoadPolicy<Vectorized<T>> {};

template <typename Scalar, typename Acc>
struct OuterSumCastLoadPolicy {
  using Vec = Vectorized<Acc>;

  static constexpr int64_t memsize() {
    return sizeof(Scalar) * Vec::size();
  }

  static Vec load(
      const char* __restrict data, int64_t stride, int64_t index) {
    const auto* ptr = reinterpret_cast<const Scalar*>(data + index * stride);
    return tensorplay::vec::load_reduced_float(ptr);
  }
};

template <typename T>
struct OuterSumCastLoadPolicy<T, T> : LoadPolicy<Vectorized<T>> {};

template <typename T>
struct StorePolicy {
  static void store(
      char* __restrict data, int64_t stride, int64_t index, T value) {
    auto* ptr = reinterpret_cast<T*>(data + index * stride);
    *ptr += value;
  }
};

template <typename Scalar, typename Acc>
struct CastStoreAccumulate {
  static void store(
      char* __restrict data, int64_t stride, int64_t index, Acc value) {
    auto* ptr = reinterpret_cast<Scalar*>(data + index * stride);
    *ptr += static_cast<Scalar>(value);
  }
};

template <typename Store, typename T>
inline void store_sum(
    char* __restrict data, int64_t stride, int64_t index, T value) {
  Store::store(data, stride, index, value);
}

template <typename Store, typename T, size_t N>
inline void store_sum(
    char* __restrict data,
    int64_t stride,
    int64_t index,
    const std::array<T, N>& values) {
  char* base = data + stride * index;
  for (size_t k = 0; k < N; ++k) {
    Store::store(base, stride, static_cast<int64_t>(k), values[k]);
  }
}

template <typename Store, typename T>
inline void store_sum(
    char* __restrict data,
    int64_t stride,
    int64_t index,
    const Vectorized<T>& values) {
  alignas(64) std::array<T, Vectorized<T>::size()> scalar_values{};
  values.store(scalar_values.data());
  store_sum<Store>(data, stride, index, scalar_values);
}

template <typename T, int64_t N, typename Load>
std::array<T, N> multi_row_sum(
    const char* __restrict data,
    int64_t row_stride,
    int64_t col_stride,
    int64_t size) {
  constexpr int64_t levels = 4;
  const int64_t level_power = std::max<int64_t>(4, ceil_log2(size) / levels);
  const int64_t level_step = int64_t(1) << level_power;
  const int64_t level_mask = level_step - 1;

  std::array<std::array<T, N>, levels> acc{};
  for (auto& row : acc) {
    row.fill(T(0));
  }

  int64_t i = 0;
  for (; i + level_step <= size;) {
    for (int64_t j = 0; j < level_step; ++j, ++i) {
      const char* sum_base = data + i * row_stride;
      #if !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (int64_t k = 0; k < N; ++k) {
        acc[0][k] += Load::load(sum_base, col_stride, k);
      }
    }

    for (int64_t j = 1; j < levels; ++j) {
      #if !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (int64_t k = 0; k < N; ++k) {
        acc[j][k] += acc[j - 1][k];
        acc[j - 1][k] = T(0);
      }

      const int64_t mask = level_mask << (j * level_power);
      if ((i & mask) != 0) {
        break;
      }
    }
  }

  for (; i < size; ++i) {
    const char* sum_base = data + i * row_stride;
    #if !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t k = 0; k < N; ++k) {
      acc[0][k] += Load::load(sum_base, col_stride, k);
    }
  }

  #if !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (int64_t j = 1; j < levels; ++j) {
    #if !defined(COMPILING_FOR_MIN_SIZE)
    # pragma unroll
    #endif
    for (int64_t k = 0; k < N; ++k) {
      acc[0][k] += acc[j][k];
    }
  }
  return acc[0];
}

template <typename T, typename Load>
T row_sum(const char* __restrict data, int64_t stride, int64_t size) {
  constexpr int64_t ilp_factor = 4;
  const int64_t size_ilp = size / ilp_factor;
  auto partial_sums = multi_row_sum<T, ilp_factor, Load>(
      data, stride * ilp_factor, stride, size_ilp);

  for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
    partial_sums[0] += Load::load(data, stride, i);
  }
  #if !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (int64_t k = 1; k < ilp_factor; ++k) {
    partial_sums[0] += partial_sums[k];
  }
  return partial_sums[0];
}

template <typename T, typename VecLoad, typename ScalarLoad, typename Store>
void vectorized_inner_sum(
    char* __restrict data[2],
    int64_t outer_stride,
    int64_t out_stride,
    int64_t size0,
    int64_t size1) {
  using Vec = Vectorized<T>;
  constexpr int64_t vec_stride = VecLoad::memsize();
  constexpr int64_t scalar_stride = ScalarLoad::memsize();
  constexpr int64_t vec_numel = vec_stride / scalar_stride;
  const int64_t vec_size = size0 / vec_numel;

  for (int64_t j = 0; j < size1; ++j) {
    const char* row = data[1] + j * outer_stride;
    auto vec_acc = row_sum<Vec, VecLoad>(row, vec_stride, vec_size);

    T final_acc = 0;
    for (int64_t k = vec_size * vec_numel; k < size0; ++k) {
      final_acc += ScalarLoad::load(row, scalar_stride, k);
    }

    alignas(64) std::array<T, Vec::size()> partials{};
    vec_acc.store(partials.data());
    for (const auto value : partials) {
      final_acc += value;
    }
    store_sum<Store>(data[0], out_stride, j, final_acc);
  }
}

template <typename T, typename Load, typename Store>
void scalar_inner_sum(
    char* __restrict data[2],
    const int64_t in_strides[2],
    int64_t out_stride,
    int64_t size0,
    int64_t size1) {
  for (int64_t j = 0; j < size1; ++j) {
    const char* row = data[1] + j * in_strides[1];
    const T value = row_sum<T, Load>(row, in_strides[0], size0);
    store_sum<Store>(data[0], out_stride, j, value);
  }
}

template <typename T, typename VecLoad, typename ScalarLoad, typename Store>
void vectorized_outer_sum(
    char* __restrict data[2],
    int64_t inner_stride,
    int64_t out_stride,
    int64_t size0,
    int64_t size1) {
  using Vec = Vectorized<T>;
  constexpr int64_t scalar_stride = ScalarLoad::memsize();
  constexpr int64_t vec_stride = VecLoad::memsize();
  constexpr int64_t rows = 4;

  int64_t j = 0;
  for (; j + rows * Vec::size() <= size1; j += rows * Vec::size()) {
    const char* row = data[1] + j * scalar_stride;
    auto sums = multi_row_sum<Vec, rows, VecLoad>(
        row, inner_stride, vec_stride, size0);
    for (int64_t i = 0; i < rows; ++i) {
      store_sum<Store>(data[0], out_stride, j + i * Vec::size(), sums[i]);
    }
  }

  for (; j + Vec::size() <= size1; j += Vec::size()) {
    const char* row = data[1] + j * scalar_stride;
    const Vec sums = row_sum<Vec, VecLoad>(row, inner_stride, size0);
    store_sum<Store>(data[0], out_stride, j, sums);
  }

  for (; j < size1; ++j) {
    const char* row = data[1] + j * scalar_stride;
    const T value = row_sum<T, ScalarLoad>(row, inner_stride, size0);
    store_sum<Store>(data[0], out_stride, j, value);
  }
}

template <typename Acc, typename Scalar = Acc>
bool contiguous_sum_dim(const Tensor& input, Tensor& output, int64_t dim) {
  if (!input.is_contiguous() || input.dim() == 0) {
    return false;
  }
  const int64_t ndim = input.dim();
  if (dim != 0 && dim != ndim - 1) {
    return false;
  }

  const int64_t reduce_size = input.size(dim);
  if (input.numel() == 0 || reduce_size == 0) {
    return true;
  }

  using Vec = Vectorized<Acc>;
  using InnerVecLoad = InnerSumCastLoadPolicy<Scalar, Acc>;
  using OuterVecLoad = OuterSumCastLoadPolicy<Scalar, Acc>;
  using ScalarLoad = CastLoadPolicy<Scalar, Acc>;
  using Store = CastStoreAccumulate<Scalar, Acc>;
  char* output_data = static_cast<char*>(output.data_ptr());
  const char* input_data = static_cast<const char*>(input.data_ptr());

  if (dim == ndim - 1) {
    const int64_t rows = input.numel() / reduce_size;
    const int64_t row_grain = std::max<int64_t>(
        1, tensorplay::parallel::GRAIN_SIZE / reduce_size);
    tensorplay::parallel::parallel_for(
        0, rows, row_grain, [&](int64_t begin, int64_t end) {
          char* data[2] = {
              output_data + begin * static_cast<int64_t>(sizeof(Scalar)),
              const_cast<char*>(input_data +
                                begin * reduce_size * static_cast<int64_t>(sizeof(Scalar)))};
          vectorized_inner_sum<Acc, InnerVecLoad, ScalarLoad, Store>(
              data,
              reduce_size * static_cast<int64_t>(sizeof(Scalar)),
              static_cast<int64_t>(sizeof(Scalar)),
              reduce_size,
              end - begin);
        });
    return true;
  }

  const int64_t columns = input.numel() / reduce_size;
  const int64_t group = 4 * Vec::size();
  const int64_t approximate_grain = std::max<int64_t>(
      1, tensorplay::parallel::GRAIN_SIZE / reduce_size);
  const int64_t column_grain = approximate_grain >= group
      ? (approximate_grain / group) * group
      : group;
  tensorplay::parallel::parallel_for(
      0, columns, column_grain, [&](int64_t begin, int64_t end) {
        char* data[2] = {
            output_data + begin * static_cast<int64_t>(sizeof(Scalar)),
            const_cast<char*>(input_data + begin * static_cast<int64_t>(sizeof(Scalar)))};
        vectorized_outer_sum<Acc, OuterVecLoad, ScalarLoad, Store>(
            data,
            columns * static_cast<int64_t>(sizeof(Scalar)),
            static_cast<int64_t>(sizeof(Scalar)),
            reduce_size,
            end - begin);
      });
  return true;
}

template <typename T, typename Load, typename Store>
void scalar_outer_sum(
    char* __restrict data[2],
    const int64_t in_strides[2],
    int64_t out_stride,
    int64_t size0,
    int64_t size1) {
  constexpr int64_t rows = 4;
  int64_t j = 0;
  for (; j + rows - 1 < size1; j += rows) {
    const char* row = data[1] + j * in_strides[1];
    auto sums = multi_row_sum<T, rows, Load>(
        row, in_strides[0], in_strides[1], size0);
    store_sum<Store>(data[0], out_stride, j, sums);
  }

  for (; j < size1; ++j) {
    const char* row = data[1] + j * in_strides[1];
    const T value = row_sum<T, Load>(row, in_strides[0], size0);
    store_sum<Store>(data[0], out_stride, j, value);
  }
}

template <typename Scalar>
inline void scalar_sum_loop(
    char* __restrict data[3],
    const int64_t strides[3],
    int64_t begin,
    int64_t end) {
  for (int64_t i = begin; i < end; ++i) {
    auto* out = reinterpret_cast<Scalar*>(data[0] + i * strides[0]);
    const auto* in = reinterpret_cast<const Scalar*>(data[2] + i * strides[2]);
    *out += *in;
  }
}

template <typename F>
inline void unary_outer_loop(
    char* __restrict data[2], const int64_t strides[2], int64_t n, F fn) {
  for (int64_t j = 0; j < n; ++j) {
    fn();
    data[0] += strides[0];
    data[1] += strides[1];
  }
}

template <typename Acc, typename Scalar = Acc>
void cascade_sum(TensorIteratorBase& iter) {
  iter.output_base().zero_();
  iter.parallel_reduce(
      [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
        int64_t in_strides[] = {strides[1], strides[3]};
        int64_t out_strides[] = {strides[0], strides[2]};

        if (out_strides[0] != 0 && out_strides[1] == 0) {
          std::swap(in_strides[0], in_strides[1]);
          std::swap(out_strides[0], out_strides[1]);
          std::swap(size0, size1);
        }

        if (out_strides[0] != 0 && out_strides[1] != 0) {
          int64_t outer_strides[] = {strides[2], strides[3]};
          unary_outer_loop(data, outer_strides, size1, [&] {
            char* ptrs[3] = {data[0], data[0], data[1]};
            int64_t inner_strides[] = {strides[0], strides[0], strides[1]};
            scalar_sum_loop<Scalar>(ptrs, inner_strides, 0, size0);
          });
          return;
        }

        TP_CHECK(out_strides[0] == 0, "sum iterator output layout is invalid");
        using ScalarLoad = CastLoadPolicy<Scalar, Acc>;
        using Store = CastStoreAccumulate<Scalar, Acc>;
        using InnerVecLoad = InnerSumCastLoadPolicy<Scalar, Acc>;
        using OuterVecLoad = OuterSumCastLoadPolicy<Scalar, Acc>;
        constexpr int64_t input_vector_size =
            InnerVecLoad::memsize() / sizeof(Scalar);

        if (in_strides[0] == sizeof(Scalar) && size0 >= input_vector_size) {
          vectorized_inner_sum<Acc, InnerVecLoad, ScalarLoad, Store>(
              data, in_strides[1], out_strides[1], size0, size1);
        } else if (in_strides[1] == sizeof(Scalar) &&
                   size1 >= input_vector_size) {
          vectorized_outer_sum<Acc, OuterVecLoad, ScalarLoad, Store>(
              data, in_strides[0], out_strides[1], size0, size1);
        } else if (in_strides[0] < in_strides[1]) {
          scalar_inner_sum<Acc, ScalarLoad, Store>(
              data, in_strides, out_strides[1], size0, size1);
        } else {
          scalar_outer_sum<Acc, ScalarLoad, Store>(
              data, in_strides, out_strides[1], size0, size1);
        }
      });
}

}  // namespace sum_detail
}  // namespace CPU_CAPABILITY
}  // namespace tensorplay::cpu
