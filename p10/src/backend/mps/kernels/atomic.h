#pragma once
#include <metal_atomic>
namespace p10 {
namespace metal {

// Atomic operations helper
template <typename T>
struct AtomicType {};
template <typename T>
using AtomicType_t = typename AtomicType<T>::type;

template <typename AT, typename T>
static inline void atomic_binary_op_helper(
    device ::metal::atomic<AT>* data,
    long offset,
    T value,
    T (*op)(T, T)) {
  auto ptr = data + offset;
  auto old = ::metal::atomic_load_explicit(ptr, ::metal::memory_order_relaxed);
  T val;
  do {
    val = op(old, value);
  } while (!::metal::atomic_compare_exchange_weak_explicit(
      ptr,
      &old,
      val,
      ::metal::memory_order_relaxed,
      ::metal::memory_order_relaxed));
}

template <>
struct AtomicType<float> {
  using type = ::metal::atomic<float>;
  static inline void atomic_add(device type* data, long offset, float value) {
    ::metal::atomic_fetch_add_explicit(
        data + offset, value, ::metal::memory_order_relaxed);
  }
  static inline void atomic_binary_op(
      device type* data,
      long offset,
      float value,
      float (*op)(float, float)) {
    atomic_binary_op_helper(data, offset, value, op);
  }
};

template <>
struct AtomicType<int> {
  using type = ::metal::atomic<int>;
  static inline void atomic_add(device type* data, long offset, int value) {
    ::metal::atomic_fetch_add_explicit(
        data + offset, value, ::metal::memory_order_relaxed);
  }
  static inline void atomic_binary_op(
      device type* data,
      long offset,
      int value,
      int (*op)(int, int)) {
    atomic_binary_op_helper(data, offset, value, op);
  }
};

// Metal 3.2+ does not provide atomics on half-precision floats; simulate them
// with a 32-bit atomic compare-and-exchange over the surrounding word.
template <typename T>
static inline void atomic_add_helper(
    device ::metal::atomic<uint>* data,
    long offset,
    T value) {
  // atomic<uint> requires 4-byte alignment; fix up misaligned pointers
  auto addr = reinterpret_cast<ulong>(data);
  auto misalign = (addr % alignof(::metal::atomic<uint>)) / sizeof(T);
  data = reinterpret_cast<device ::metal::atomic<uint>*>(
      reinterpret_cast<device char*>(data) - misalign * sizeof(T));
  offset += misalign;

  constexpr auto elem_per_enum = sizeof(uint) / sizeof(T);
  auto ptr = data + (offset / elem_per_enum);
  auto old = ::metal::atomic_load_explicit(ptr, ::metal::memory_order_relaxed);
  T val;
  do {
    val = old + value;
  } while (!::metal::atomic_compare_exchange_weak_explicit(
      ptr,
      &old,
      val,
      ::metal::memory_order_relaxed,
      ::metal::memory_order_relaxed));
  ::metal::atomic_store_explicit(ptr, old, ::metal::memory_order_relaxed);
}

template <>
struct AtomicType<half> {
  using type = ::metal::atomic<half>;
  static inline void atomic_add(device type* data, long offset, half value) {
    atomic_add_helper(data, offset, value);
  }
};

template <>
struct AtomicType<bfloat> {
  using type = ::metal::atomic<bfloat>;
  static inline void atomic_add(device type* data, long offset, bfloat value) {
    atomic_add_helper(data, offset, value);
  }
};

} // namespace metal
} // namespace p10