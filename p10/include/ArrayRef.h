#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <iterator>
#include "Macros.h"

namespace tensorplay {

// ArrayRef<int64_t> — a lightweight view over a contiguous run of int64_t,
// mirroring c10::IntArrayRef. Does not own the data; the underlying buffer
// must outlive the reference.
class P10_API IntArrayRef {
public:
    using iterator = const int64_t*;
    using const_iterator = const int64_t*;
    using size_type = size_t;
    using value_type = int64_t;

    IntArrayRef() : data_(nullptr), length_(0) {}
    IntArrayRef(const int64_t* data, size_t length) : data_(data), length_(length) {}
    IntArrayRef(const std::vector<int64_t>& vec) : data_(vec.data()), length_(vec.size()) {}
    template <size_t N>
    constexpr IntArrayRef(const int64_t (&arr)[N]) : data_(arr), length_(N) {}

    iterator begin() const { return data_; }
    iterator end() const { return data_ + length_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + length_; }

    bool empty() const { return length_ == 0; }
    size_t size() const { return length_; }
    const int64_t* data() const { return data_; }

    int64_t operator[](size_t index) const { return data_[index]; }

    /// The first element. Asserts the array is non-empty.
    int64_t front() const { return data_[0]; }

    /// The last element. Asserts the array is non-empty.
    int64_t back() const { return data_[length_ - 1]; }

    /// An ArrayRef whose elements are a (possibly empty) subrange of this one.
    IntArrayRef slice(size_t begin, size_t end) const {
        if (begin >= end) return IntArrayRef();
        return IntArrayRef(data_ + begin, end - begin);
    }

    /// Explicit materialization into an owned vector (mirrors c10 .vec()).
    std::vector<int64_t> vec() const {
        return std::vector<int64_t>(data_, data_ + length_);
    }

    operator std::vector<int64_t>() const { return vec(); }

    bool equals(IntArrayRef other) const {
        if (length_ != other.length_) return false;
        if (data_ == other.data_) return true;
        for (size_t i = 0; i < length_; ++i) {
            if (data_[i] != other.data_[i]) return false;
        }
        return true;
    }

    friend bool operator==(IntArrayRef a, IntArrayRef b) { return a.equals(b); }
    friend bool operator!=(IntArrayRef a, IntArrayRef b) { return !a.equals(b); }
    friend bool operator==(const std::vector<int64_t>& a, IntArrayRef b) { return IntArrayRef(a).equals(b); }
    friend bool operator==(IntArrayRef a, const std::vector<int64_t>& b) { return a.equals(IntArrayRef(b)); }
    friend bool operator!=(const std::vector<int64_t>& a, IntArrayRef b) { return !Intarrayref_eq(a, b); }
    friend bool operator!=(IntArrayRef a, const std::vector<int64_t>& b) { return !a.equals(IntArrayRef(b)); }

private:
    static bool Intarrayref_eq(const std::vector<int64_t>& a, IntArrayRef b) {
        return IntArrayRef(a).equals(b);
    }

    const int64_t* data_;
    size_t length_;
};

} // namespace tensorplay
