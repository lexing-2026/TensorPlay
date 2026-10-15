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

// Metal 3.2+ has no atomics on half-precision floats; simulate them with a
// 32-bit atomic compare-and-exchange over the surrounding word.
template <typename T>
static inline void atomic_add_helper(
    device ::metal::atomic<uint>* data,
    long offset,
    T value) {
  // atomic<uint> requires 4-byte alignment; fix up misaligned pointers.
  auto addr = reinterpret_cast<ulong>(data);
  auto misalign = (addr % alignof(::metal::atomic<uint>)) / sizeof(T);
  data = reinterpret_cast<device ::metal::atomic<uint>*>(
      reinterpret_cast<device char*>(data) - misalign * sizeof(T));
  offset += misalign;

  constexpr auto elem_per_enum = sizeof(uint) / sizeof(T);
  auto ptr = data + (offset / elem_per_enum);
  auto old = ::metal::atomic_load_explicit(ptr, ::metal::memory_order_relaxed);
  union {
    uint i;
    T t[elem_per_enum];
  } val;
  do {
    val.i = old;
    val.t[offset & (elem_per_enum - 1)] += value;
  } while (!::metal::atomic_compare_exchange_weak_explicit(
      ptr,
      &old,
      val.i,
      ::metal::memory_order_relaxed,
      ::metal::memory_order_relaxed));
}

template <typename T>
static inline void atomic_binary_op_helper(
    device ::metal::atomic<uint>* data,
    long offset,
    T value,
    T (*Op)(T, T)) {
  // atomic<uint> requires 4-byte alignment; fix up misaligned pointers.
  auto addr = reinterpret_cast<ulong>(data);
  auto misalign = (addr % alignof(::metal::atomic<uint>)) / sizeof(T);
  data = reinterpret_cast<device ::metal::atomic<uint>*>(
      reinterpret_cast<device char*>(data) - misalign * sizeof(T));
  offset += misalign;

  constexpr auto elem_per_enum = sizeof(uint) / sizeof(T);
  auto ptr = data + (offset / elem_per_enum);
  auto old = ::metal::atomic_load_explicit(ptr, ::metal::memory_order_relaxed);
  union {
    uint i;
    T t[elem_per_enum];
  } val;
  do {
    val.i = old;
    val.t[offset & (elem_per_enum - 1)] =
        Op(val.t[offset & (elem_per_enum - 1)], value);
  } while (!::metal::atomic_compare_exchange_weak_explicit(
      ptr,
      &old,
      val.i,
      ::metal::memory_order_relaxed,
      ::metal::memory_order_relaxed));
}

template <>
struct AtomicType<half> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, half value) {
    atomic_add_helper(data, offset, value);
  }
  static inline void atomic_binary_op(
      device type* data,
      long offset,
      half value,
      half (*op)(half, half)) {
    atomic_binary_op_helper(data, offset, value, op);
  }
};

template <>
struct AtomicType<short> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, short value) {
    atomic_add_helper(data, offset, value);
  }
  static inline void atomic_binary_op(
      device type* data,
      long offset,
      short value,
      short (*op)(short, short)) {
    atomic_binary_op_helper(data, offset, value, op);
  }
};

template <>
struct AtomicType<char> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, char value) {
    atomic_add_helper(data, offset, value);
  }
  static inline void atomic_binary_op(
      device type* data,
      long offset,
      char value,
      char (*op)(char, char)) {
    atomic_binary_op_helper(data, offset, value, op);
  }
};

template <>
struct AtomicType<uchar> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, uchar value) {
    atomic_add_helper(data, offset, value);
  }
  static inline void atomic_binary_op(
      device type* data,
      long offset,
      uchar value,
      uchar (*op)(uchar, uchar)) {
    atomic_binary_op_helper(data, offset, value, op);
  }
};

template <>
struct AtomicType<bfloat> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, bfloat value) {
    atomic_add_helper<bfloat>(data, offset, value);
  }
  static inline void atomic_binary_op(
      device type* data,
      long offset,
      bfloat value,
      bfloat (*op)(bfloat, bfloat)) {
    atomic_binary_op_helper(data, offset, value, op);
  }
};

// Metal's atomic_bool is 4 bytes, so unaligned bool updates fall back to a
// packed compare-and-exchange. Accumulating booleans is an OR: a false input
// leaves the word untouched.
template <>
struct AtomicType<bool> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, bool value) {
    if (!value) {
      return;
    }
    auto ptr = data + (offset >> 2);
    auto old =
        ::metal::atomic_load_explicit(ptr, ::metal::memory_order_relaxed);
    union {
      uint i;
      bool t[4];
    } val;
    do {
      val.i = old;
      val.t[offset & 3] = true;
    } while (!::metal::atomic_compare_exchange_weak_explicit(
        ptr,
        &old,
        val.i,
        ::metal::memory_order_relaxed,
        ::metal::memory_order_relaxed));
  }
  // Generic packed-CAS supports any bool op (AND for prod/amin, OR for amax).
  static inline void atomic_binary_op(
      device type* data,
      long offset,
      bool value,
      bool (*op)(bool, bool)) {
    atomic_binary_op_helper<bool>(data, offset, value, op);
  }
};

// ComplexHalf atomic add: CAS over the 32-bit word holding both halves.
template <>
struct AtomicType<half2> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, half2 value) {
    auto ptr = data + offset;
    auto old =
        ::metal::atomic_load_explicit(ptr, ::metal::memory_order_relaxed);
    while (!::metal::atomic_compare_exchange_weak_explicit(
        ptr,
        &old,
        as_type<uint>(as_type<half2>(old) + value),
        ::metal::memory_order_relaxed,
        ::metal::memory_order_relaxed))
      ;
  }
};

// Metal has no 64-bit atomic add; the two-word trick below is eventually
// consistent: concurrent writers may interleave, but every write is applied
// to both words so the stored value converges to the true sum.
template <>
struct AtomicType<long> {
  using type = ::metal::atomic<uint>;
  static inline void atomic_add(device type* data, long offset, long value) {
    const auto value_bits = as_type<ulong>(value);
    const uint low = static_cast<uint>(value_bits);
    uint high = static_cast<uint>(value_bits >> 32);
    auto ptr = data + (offset << 1);
    auto old_low =
        atomic_fetch_add_explicit(ptr, low, ::metal::memory_order_relaxed);
    high += (old_low + low < old_low) ? 1 : 0;
    atomic_fetch_add_explicit(ptr + 1, high, ::metal::memory_order_relaxed);
  }
};

// ComplexFloat atomic add: two independent 32-bit adds, eventually
// consistent under races.
template <>
struct AtomicType<float2> {
  using type = ::metal::atomic<float>;
  static inline void atomic_add(device type* data, long offset, float2 value) {
    auto ptr = data + (offset << 1);
    atomic_fetch_add_explicit(ptr + 0, value.x, ::metal::memory_order_relaxed);
    atomic_fetch_add_explicit(ptr + 1, value.y, ::metal::memory_order_relaxed);
  }
};

} // namespace metal
} // namespace p10