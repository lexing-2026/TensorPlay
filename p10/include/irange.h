#pragma once
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace tensorplay {

// Lightweight equivalent of c10::irange: for (const auto i : irange(n)) { ... }
// yields i in [0, n) with the same signed/unsigned semantics as torch.
template <typename T>
class irange {
  static_assert(std::is_integral_v<T>, "irange requires an integral type");

 public:
  class iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    iterator(T value) : value_(value) {}
    T operator*() const { return value_; }
    iterator& operator++() { ++value_; return *this; }
    bool operator==(const iterator& other) const { return value_ == other.value_; }
    bool operator!=(const iterator& other) const { return value_ != other.value_; }

   private:
    T value_;
  };

  // c10 parity: [begin, end) with end<=begin yielding an EMPTY range -- the
  // range-for end test is `!=`, so the terminator is clamped up to begin.
  // reorder_dimensions relies on this for 0-dim reductions
  // (irange(1, ndim()) must not iterate when ndim()==0).
  irange(T begin, T end) : begin_(begin), end_(std::max(begin, end)) {}
  explicit irange(T end) : begin_(T{0}), end_(std::max(T{0}, end)) {}

  iterator begin() const { return iterator(begin_); }
  iterator end() const { return iterator(end_); }

 private:
  T begin_;
  T end_;
};

} // namespace tensorplay