#include "SizesAndStrides.h"
#include "Exception.h"
#include <sstream>

namespace tensorplay {

SizesAndStrides::SizesAndStrides(const std::vector<int64_t>& sizes)
    : sizes_(sizes), strides_(compute_contiguous_strides(sizes)) {}

SizesAndStrides::SizesAndStrides(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides)
    : sizes_(sizes), strides_(strides) {
    
    if (sizes.size() != strides.size()) {
        TP_THROW(ValueError, "Sizes and strides must have the same length");
    }
}

void SizesAndStrides::resize(const std::vector<int64_t>& new_sizes) {
    sizes_ = new_sizes;
    strides_ = compute_contiguous_strides(new_sizes);
}

void SizesAndStrides::set_sizes_and_strides(const std::vector<int64_t>& new_sizes, const std::vector<int64_t>& new_strides) {
    if (new_sizes.size() != new_strides.size()) {
        TP_THROW(ValueError, "Sizes and strides must have the same length");
    }
    
    sizes_ = new_sizes;
    strides_ = new_strides;
}

void SizesAndStrides::set_size(size_t dim, int64_t new_size) {
    if (dim >= sizes_.size()) {
        TP_THROW(IndexError, "Dimension out of range");
    }
    
    sizes_[dim] = new_size;
    // Note: We don't automatically recompute strides here to preserve custom strides
}

void SizesAndStrides::set_stride(size_t dim, int64_t new_stride) {
    if (dim >= strides_.size()) {
        TP_THROW(IndexError, "Dimension out of range");
    }
    
    strides_[dim] = new_stride;
}

int64_t SizesAndStrides::numel() const {
    if (sizes_.empty()) {
        // 0-dimensional tensor has 1 element (scalar)
        return 1;
    }
    
    int64_t result = 1;
    for (int64_t size : sizes_) {
        result *= size;
    }
    
    return result;
}

bool SizesAndStrides::is_contiguous() const {
    if (sizes_.empty()) {
        return true;
    }
    
    // Compute expected contiguous strides
    std::vector<int64_t> expected_strides = compute_contiguous_strides(sizes_);
    
    // Check if actual strides match expected strides
    return strides_ == expected_strides;
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
    
    for (size_t i = 0; i < sizes_.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << sizes_[i];
    }
    oss << "], strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << strides_[i];
    }
    oss << "])";
    return oss.str();
}

} // namespace tensorplay
