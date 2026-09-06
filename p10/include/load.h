#pragma once
#include <cstring>

namespace tensorplay {

// Lightweight equivalent of c10::Load: unaligned load of a trivially
// copyable type from a raw pointer, safe for arbitrary alignment.
template <typename T>
inline T load(const void* ptr) {
  T value;
  std::memcpy(&value, ptr, sizeof(T));
  return value;
}

// Typed-pointer overload so generic elementwise code (e.g. the
// dtype-conversion loop in vec_base) can pass its iterator directly.
template <typename T>
inline T load(const T* ptr) {
  return load<T>(static_cast<const void*>(ptr));
}

} // namespace tensorplay