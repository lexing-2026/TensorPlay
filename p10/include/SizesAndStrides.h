#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "Macros.h"
#include "ArrayRef.h"

namespace tensorplay {

// Manages a tensor's sizes and strides with inline storage for up to
// kInlineSize dimensions, mirroring c10::impl::SizesAndStrides. Tensors of
// rank <= kInlineSize (the overwhelming majority) are created and copied with
// zero heap allocations.
class P10_API SizesAndStrides {
public:
    static constexpr size_t kInlineSize = 5;

    // Default constructor: 0-dim (scalar) tensor.
    SizesAndStrides() = default;

    // Constructor with sizes only (computes contiguous strides)
    explicit SizesAndStrides(const std::vector<int64_t>& sizes);

    // Constructor with sizes and strides
    SizesAndStrides(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides);

    SizesAndStrides(const SizesAndStrides& other);
    SizesAndStrides(SizesAndStrides&& other) noexcept;
    SizesAndStrides& operator=(const SizesAndStrides& other);
    SizesAndStrides& operator=(SizesAndStrides&& other) noexcept;

    friend void swap(SizesAndStrides& a, SizesAndStrides& b) noexcept {
        using std::swap;
        swap(a.size_, b.size_);
        for (size_t i = 0; i < kInlineSize; ++i) {
            swap(a.inline_sizes_[i], b.inline_sizes_[i]);
            swap(a.inline_strides_[i], b.inline_strides_[i]);
        }
        swap(a.heap_sizes_, b.heap_sizes_);
        swap(a.heap_strides_, b.heap_strides_);
    }

    // Views over the storage (no copies; mirrors c10 IntArrayRef accessors).
    IntArrayRef sizes() const { return IntArrayRef(sizes_data(), size_); }
    IntArrayRef strides() const { return IntArrayRef(strides_data(), size_); }

    // Get size at specific dimension
    int64_t size(size_t dim) const { return sizes_data()[dim]; }

    // Get stride at specific dimension
    int64_t stride(size_t dim) const { return strides_data()[dim]; }

    // Get number of dimensions
    size_t dim() const { return size_; }

    // Check if empty (no dimensions)
    bool empty() const { return size_ == 0; }

    // Resize (updates sizes and recomputes contiguous strides)
    void resize(const std::vector<int64_t>& new_sizes);

    // Set sizes and strides
    void set_sizes_and_strides(const std::vector<int64_t>& new_sizes, const std::vector<int64_t>& new_strides);

    // Set size at specific dimension
    void set_size(size_t dim, int64_t new_size);

    // Set stride at specific dimension
    void set_stride(size_t dim, int64_t new_stride);

    // Compute total number of elements
    int64_t numel() const;

    // Check if storage is contiguous
    bool is_contiguous() const;

    // Compute strides for contiguous storage
    static std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& sizes);

    // Convert to string representation
    std::string toString() const;

    // Equality operator
    bool operator==(const SizesAndStrides& other) const {
        if (size_ != other.size_) return false;
        for (size_t i = 0; i < size_; ++i) {
            if (sizes_data()[i] != other.sizes_data()[i]) return false;
            if (strides_data()[i] != other.strides_data()[i]) return false;
        }
        return true;
    }

    bool operator!=(const SizesAndStrides& other) const {
        return !(*this == other);
    }

private:
    bool is_heap() const { return size_ > kInlineSize; }
    const int64_t* sizes_data() const { return is_heap() ? heap_sizes_.get() : inline_sizes_; }
    int64_t* sizes_data() { return is_heap() ? heap_sizes_.get() : inline_sizes_; }
    const int64_t* strides_data() const { return is_heap() ? heap_strides_.get() : inline_strides_; }
    int64_t* strides_data() { return is_heap() ? heap_strides_.get() : inline_strides_; }

    // Grows/shrinks storage to hold exactly `new_size` dims. Contents are not
    // preserved across a heap/inline transition; callers initialize after.
    void set_extent(size_t new_size);

    size_t size_ = 0;
    int64_t inline_sizes_[kInlineSize] = {};
    int64_t inline_strides_[kInlineSize] = {};
    std::unique_ptr<int64_t[]> heap_sizes_;   // null while rank <= kInlineSize
    std::unique_ptr<int64_t[]> heap_strides_;
};

} // namespace tensorplay
