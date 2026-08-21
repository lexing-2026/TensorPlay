#include "SizesAndStrides.h"
#include "Exception.h"
#include <sstream>
#include <algorithm>

namespace tensorplay {

SizesAndStrides::SizesAndStrides(const std::vector<int64_t>& sizes) {
    set_extent(sizes.size());
    if (!sizes.empty()) {
        std::copy(sizes.begin(), sizes.end(), sizes_data());
    }
    auto strides = compute_contiguous_strides(sizes);
    if (!strides.empty()) {
        std::copy(strides.begin(), strides.end(), strides_data());
    }
}

SizesAndStrides::SizesAndStrides(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides) {
    if (sizes.size() != strides.size()) {
        TP_THROW(ValueError, "Sizes and strides must have the same length");
    }
    set_extent(sizes.size());
    if (!sizes.empty()) {
        std::copy(sizes.begin(), sizes.end(), sizes_data());
        std::copy(strides.begin(), strides.end(), strides_data());
    }
}

SizesAndStrides::SizesAndStrides(const SizesAndStrides& other)
    : size_(other.size_),
      inline_sizes_{},
      inline_strides_{},
      heap_sizes_(other.is_heap() ? std::unique_ptr<int64_t[]>(new int64_t[other.size_]) : nullptr),
      heap_strides_(other.is_heap() ? std::unique_ptr<int64_t[]>(new int64_t[other.size_]) : nullptr) {
    if (size_ == 0) return;
    if (is_heap()) {
        std::copy(other.sizes_data(), other.sizes_data() + size_, heap_sizes_.get());
        std::copy(other.strides_data(), other.strides_data() + size_, heap_strides_.get());
    } else {
        std::copy(other.inline_sizes_, other.inline_sizes_ + size_, inline_sizes_);
        std::copy(other.inline_strides_, other.inline_strides_ + size_, inline_strides_);
    }
}

SizesAndStrides::SizesAndStrides(SizesAndStrides&& other) noexcept
    : size_(other.size_),
      inline_sizes_{},
      inline_strides_{},
      heap_sizes_(std::move(other.heap_sizes_)),
      heap_strides_(std::move(other.heap_strides_)) {
    if (!is_heap() && size_ > 0) {
        std::copy(other.inline_sizes_, other.inline_sizes_ + size_, inline_sizes_);
        std::copy(other.inline_strides_, other.inline_strides_ + size_, inline_strides_);
    }
}

SizesAndStrides& SizesAndStrides::operator=(const SizesAndStrides& other) {
    if (this != &other) {
        SizesAndStrides tmp(other);
        swap(*this, tmp);
    }
    return *this;
}

SizesAndStrides& SizesAndStrides::operator=(SizesAndStrides&& other) noexcept {
    if (this != &other) {
        SizesAndStrides tmp(std::move(other));
        swap(*this, tmp);
    }
    return *this;
}

void SizesAndStrides::set_extent(size_t new_size) {
    const bool was_heap = is_heap();
    const bool will_be_heap = new_size > kInlineSize;
    if (will_be_heap) {
        if (!was_heap || size_ != new_size) {
            heap_sizes_.reset(new int64_t[new_size]);
            heap_strides_.reset(new int64_t[new_size]);
        }
    } else if (was_heap) {
        heap_sizes_.reset();
        heap_strides_.reset();
    }
    size_ = new_size;
}

void SizesAndStrides::resize(const std::vector<int64_t>& new_sizes) {
    set_extent(new_sizes.size());
    if (!new_sizes.empty()) {
        std::copy(new_sizes.begin(), new_sizes.end(), sizes_data());
    }
    auto strides = compute_contiguous_strides(new_sizes);
    if (!strides.empty()) {
        std::copy(strides.begin(), strides.end(), strides_data());
    }
}

void SizesAndStrides::set_sizes_and_strides(const std::vector<int64_t>& new_sizes, const std::vector<int64_t>& new_strides) {
    if (new_sizes.size() != new_strides.size()) {
        TP_THROW(ValueError, "Sizes and strides must have the same length");
    }

    set_extent(new_sizes.size());
    if (!new_sizes.empty()) {
        std::copy(new_sizes.begin(), new_sizes.end(), sizes_data());
        std::copy(new_strides.begin(), new_strides.end(), strides_data());
    }
}

void SizesAndStrides::set_size(size_t dim, int64_t new_size) {
    if (dim >= size_) {
        TP_THROW(IndexError, "Dimension out of range");
    }

    sizes_data()[dim] = new_size;
    // Note: We don't automatically recompute strides here to preserve custom strides
}

void SizesAndStrides::set_stride(size_t dim, int64_t new_stride) {
    if (dim >= size_) {
        TP_THROW(IndexError, "Dimension out of range");
    }

    strides_data()[dim] = new_stride;
}

int64_t SizesAndStrides::numel() const {
    if (size_ == 0) {
        // 0-dimensional tensor has 1 element (scalar)
        return 1;
    }

    int64_t result = 1;
    for (size_t i = 0; i < size_; ++i) {
        result *= sizes_data()[i];
    }

    return result;
}

bool SizesAndStrides::is_contiguous() const {
    if (size_ == 0) {
        return true;
    }

    // Walk expected contiguous strides without materializing them. A
    // dimension of extent 1 accepts any stride, matching PyTorch semantics.
    int64_t expected = 1;
    for (size_t i = size_; i > 0; --i) {
        if (strides_data()[i - 1] != expected && sizes_data()[i - 1] != 1) {
            return false;
        }
        if (sizes_data()[i - 1] != 1) {
            expected *= sizes_data()[i - 1];
        }
    }

    return true;
}

std::vector<int64_t> SizesAndStrides::compute_contiguous_strides(const std::vector<int64_t>& sizes) {
    std::vector<int64_t> strides(sizes.size());

    if (sizes.empty()) {
        return strides;
    }

    // Compute strides from last dimension to first
    strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }

    return strides;
}

std::string SizesAndStrides::toString() const {
    std::ostringstream oss;
    oss << "SizesAndStrides(sizes=[";
    for (size_t i = 0; i < size_; ++i) {
        if (i > 0) oss << ", ";
        oss << sizes_data()[i];
    }
    oss << "], strides=[";
    for (size_t i = 0; i < size_; ++i) {
        if (i > 0) oss << ", ";
        oss << strides_data()[i];
    }
    oss << "])";
    return oss.str();
}

} // namespace tensorplay
