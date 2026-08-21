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

} // namespace tensorplay