#pragma once

#include "DType.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tensorplay {
namespace cuda {
namespace topk_detail {

template <typename T>
__device__ inline T topk_load(const T* pointer) {
  if constexpr (std::is_same<T, Half>::value ||
                std::is_same<T, BFloat16>::value) {
    return *pointer;
  } else {
    return __ldg(pointer);
  }
}

template <typename T>
struct TopKRadixTraits;

template <>
struct TopKRadixTraits<uint8_t> {
  using key_type = uint32_t;
  static constexpr int bit_count = 8;
  __device__ static inline key_type encode(uint8_t value) { return value; }
  __device__ static inline uint8_t deconvert(key_type value) {
    return static_cast<uint8_t>(value);
  }
};

template <>
struct TopKRadixTraits<uint16_t> {
  using key_type = uint32_t;
  static constexpr int bit_count = 16;
  __device__ static inline key_type encode(uint16_t value) { return value; }
  __device__ static inline uint16_t deconvert(key_type value) {
    return static_cast<uint16_t>(value);
  }
};

template <>
struct TopKRadixTraits<uint32_t> {
  using key_type = uint32_t;
  static constexpr int bit_count = 32;
  __device__ static inline key_type encode(uint32_t value) { return value; }
  __device__ static inline uint32_t deconvert(key_type value) { return value; }
};

template <>
struct TopKRadixTraits<uint64_t> {
  using key_type = uint64_t;
  static constexpr int bit_count = 64;
  __device__ static inline key_type encode(uint64_t value) { return value; }
  __device__ static inline uint64_t deconvert(key_type value) { return value; }
};

template <>
struct TopKRadixTraits<int8_t> {
  using key_type = uint32_t;
  static constexpr int bit_count = 8;
  __device__ static inline key_type encode(int8_t value) {
    return static_cast<key_type>(static_cast<int32_t>(value) + 128);
  }
  __device__ static inline int8_t deconvert(key_type value) {
    return static_cast<int8_t>(static_cast<int32_t>(value) - 128);
  }
};

template <>
struct TopKRadixTraits<int16_t> {
  using key_type = uint32_t;
  static constexpr int bit_count = 16;
  __device__ static inline key_type encode(int16_t value) {
    return static_cast<key_type>(static_cast<int32_t>(value) + 32768);
  }
  __device__ static inline int16_t deconvert(key_type value) {
    return static_cast<int16_t>(static_cast<int32_t>(value) - 32768);
  }
};

template <>
struct TopKRadixTraits<int32_t> {
  using key_type = uint32_t;
  static constexpr int bit_count = 32;
  __device__ static inline key_type encode(int32_t value) {
    return static_cast<key_type>(static_cast<int64_t>(value) + 2147483648LL);
  }
  __device__ static inline int32_t deconvert(key_type value) {
    return static_cast<int32_t>(static_cast<int64_t>(value) - 2147483648LL);
  }
};

template <>
struct TopKRadixTraits<int64_t> {
  using key_type = uint64_t;
  static constexpr int bit_count = 64;
  __device__ static inline key_type encode(int64_t value) {
    return static_cast<key_type>(value) ^ 0x8000000000000000ULL;
  }
  __device__ static inline int64_t deconvert(key_type value) {
    return static_cast<int64_t>(value ^ 0x8000000000000000ULL);
  }
};

template <>
struct TopKRadixTraits<float> {
  using key_type = uint32_t;
  static constexpr int bit_count = 32;
  __device__ static inline key_type encode(float value) {
    const key_type bits = static_cast<key_type>(__float_as_int(value));
    const key_type mask = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return value == value ? bits ^ mask : 0xffffffffu;
  }
  __device__ static inline float deconvert(key_type value) {
    const key_type mask = (value & 0x80000000u) ? 0x80000000u : 0xffffffffu;
    return __int_as_float(value ^ mask);
  }
};

template <>
struct TopKRadixTraits<double> {
  using key_type = uint64_t;
  static constexpr int bit_count = 64;
  __device__ static inline key_type encode(double value) {
    const key_type bits = static_cast<key_type>(__double_as_longlong(value));
    const key_type mask = (bits >> 63) ? 0xffffffffffffffffULL
                                       : 0x8000000000000000ULL;
    return value == value ? bits ^ mask : 0xffffffffffffffffULL;
  }
  __device__ static inline double deconvert(key_type value) {
    const key_type mask = (value >> 63) ? 0x8000000000000000ULL
                                        : 0xffffffffffffffffULL;
    return __longlong_as_double(value ^ mask);
  }
};

template <>
struct TopKRadixTraits<Half> {
  using key_type = uint32_t;
  static constexpr int bit_count = 16;
  __device__ static inline key_type encode(Half value) {
    const key_type bits = value.x;
    const key_type mask = (bits & 0x8000u) ? 0xffffu : 0x8000u;
    const float converted = static_cast<float>(value);
    return converted == converted ? bits ^ mask : 0xffffu;
  }
  __device__ static inline Half deconvert(key_type value) {
    const key_type mask = (value & 0x8000u) ? 0x8000u : 0xffffu;
    return Half(static_cast<uint16_t>(value ^ mask), Half::from_bits());
  }
};

template <>
struct TopKRadixTraits<BFloat16> {
  using key_type = uint32_t;
  static constexpr int bit_count = 16;
  __device__ static inline key_type encode(BFloat16 value) {
    const key_type bits = value.x;
    const key_type mask = (bits & 0x8000u) ? 0xffffu : 0x8000u;
    const float converted = static_cast<float>(value);
    return converted == converted ? bits ^ mask : 0xffffu;
  }
  __device__ static inline BFloat16 deconvert(key_type value) {
    const key_type mask = (value & 0x8000u) ? 0x8000u : 0xffffu;
    BFloat16 result;
    result.x = static_cast<uint16_t>(value ^ mask);
    return result;
  }
};
template <typename Key>
__device__ inline Key topk_set_bitfield(Key value, Key field, int pos) {
  const Key mask = static_cast<Key>(3) << pos;
  return (value & ~mask) | ((field & static_cast<Key>(3)) << pos);
}

template <typename T, typename IndexType>
__device__ inline void topk_count_radix_using_mask(
    IndexType counts[4], IndexType* smem,
    typename TopKRadixTraits<T>::key_type desired,
    typename TopKRadixTraits<T>::key_type desired_mask, int digit_pos,
    IndexType slice_size, IndexType within_slice_stride, const T* data) {
  using Key = typename TopKRadixTraits<T>::key_type;
  for (int i = 0; i < 4; ++i) counts[i] = 0;
  if (threadIdx.x < 4) smem[threadIdx.x] = 0;
  __syncthreads();

  unsigned long long active = __ballot_sync(0xffffffffffffffffull,
                                            static_cast<IndexType>(threadIdx.x) < slice_size);
  for (IndexType i = static_cast<IndexType>(threadIdx.x); i < slice_size;) {
    const Key value = TopKRadixTraits<T>::encode(
        topk_load(&data[i * within_slice_stride]));
    const bool has_value = (value & desired_mask) == desired;
    const Key digit = (value >> digit_pos) & static_cast<Key>(3);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      counts[j] += static_cast<IndexType>(__popcll(__ballot_sync(
          active, has_value && digit == static_cast<Key>(j))));
    }
    i += static_cast<IndexType>(blockDim.x);
    active = __ballot_sync(active, i < slice_size);
  }

  if ((threadIdx.x & 31) == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if constexpr (sizeof(IndexType) == sizeof(uint32_t)) {
        atomicAdd(&smem[i], counts[i]);
      } else {
        atomicAdd(reinterpret_cast<unsigned long long*>(&smem[i]),
                  static_cast<unsigned long long>(counts[i]));
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i) counts[i] = smem[i];
  __syncthreads();
}

template <typename T, typename IndexType>
__device__ inline T topk_find_pattern(
    IndexType* smem, const T* data, IndexType slice_size,
    IndexType within_slice_stride,
    typename TopKRadixTraits<T>::key_type desired,
    typename TopKRadixTraits<T>::key_type desired_mask) {
  using Key = typename TopKRadixTraits<T>::key_type;
  T* pattern = reinterpret_cast<T*>(smem);
  if (threadIdx.x < 2) pattern[threadIdx.x] = static_cast<T>(0);
  __syncthreads();

  const IndexType iterations =
      ((slice_size + static_cast<IndexType>(blockDim.x) - 1) /
       static_cast<IndexType>(blockDim.x)) * static_cast<IndexType>(blockDim.x);
  for (IndexType i = static_cast<IndexType>(threadIdx.x); i < iterations;
       i += static_cast<IndexType>(blockDim.x)) {
    const bool in_range = i < slice_size;
    const T value = in_range
        ? topk_load(&data[i * within_slice_stride])
        : static_cast<T>(0);
    const Key encoded = TopKRadixTraits<T>::encode(value);
    if (in_range && (encoded & desired_mask) == desired) {
      pattern[0] = static_cast<T>(1);
      pattern[1] = value;
    }
    __syncthreads();
    const T found = pattern[0];
    const T result = pattern[1];
    __syncthreads();
    if (found != static_cast<T>(0)) return result;
  }
  return static_cast<T>(0);
}

template <typename T, typename IndexType>
__device__ inline void topk_radix_select(
    const T* data, IndexType k, bool largest, IndexType slice_size,
    IndexType within_slice_stride, IndexType* smem, T* top_k) {
  using Key = typename TopKRadixTraits<T>::key_type;
  constexpr int radix_size = 4;
  constexpr int radix_bits = 2;
  constexpr int radix_mask = radix_size - 1;
  IndexType counts[radix_size];
  Key desired = 0;
  Key desired_mask = 0;
  IndexType k_to_find = k;

  for (int digit_pos = TopKRadixTraits<T>::bit_count - radix_bits;
       digit_pos >= 0; digit_pos -= radix_bits) {
    topk_count_radix_using_mask<T, IndexType>(
        counts, smem, desired, desired_mask, digit_pos, slice_size,
        within_slice_stride, data);

    auto found_unique = [&](int i, IndexType count) -> bool {
      if (count == 1 && k_to_find == 1) {
        desired = topk_set_bitfield(
            desired, static_cast<Key>(i), digit_pos);
        desired_mask = topk_set_bitfield(
            desired_mask, static_cast<Key>(radix_mask), digit_pos);
        *top_k = topk_find_pattern<T, IndexType>(
            smem, data, slice_size, within_slice_stride, desired,
            desired_mask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, IndexType count) -> bool {
      if (count >= k_to_find) {
        desired = topk_set_bitfield(
            desired, static_cast<Key>(i), digit_pos);
        desired_mask = topk_set_bitfield(
            desired_mask, static_cast<Key>(radix_mask), digit_pos);
        return true;
      }
      k_to_find -= count;
      return false;
    };

    if (largest) {
#pragma unroll
      for (int i = radix_size - 1; i >= 0; --i) {
        const IndexType count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < radix_size; ++i) {
        const IndexType count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    }
  }
  *top_k = TopKRadixTraits<T>::deconvert(desired);
}

}
}
}
