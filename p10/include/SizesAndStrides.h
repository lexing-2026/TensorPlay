#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "Macros.h"

namespace tensorplay {

// Class to manage tensor sizes and strides
class P10_API SizesAndStrides {
private:
    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
    
public:
    // Default constructor
    SizesAndStrides() = default;
    
    // Constructor with sizes only (computes contiguous strides)
    explicit SizesAndStrides(const std::vector<int64_t>& sizes);
    
    // Constructor with sizes and strides
    SizesAndStrides(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides);
    
    // Copy constructor
    SizesAndStrides(const SizesAndStrides& other) = default;
    
    // Move constructor
    SizesAndStrides(SizesAndStrides&& other) noexcept = default;
    
    // Copy assignment operator
    SizesAndStrides& operator=(const SizesAndStrides& other) = default;
    
    // Move assignment operator
    SizesAndStrides& operator=(SizesAndStrides&& other) noexcept = default;
    
    // Get sizes
    const std::vector<int64_t>& sizes() const { return sizes_; }
    
    // Get strides
    const std::vector<int64_t>& strides() const { return strides_; }
    
    // Get size at specific dimension
    int64_t size(size_t dim) const { return sizes_[dim]; }
    
    // Get stride at specific dimension
    int64_t stride(size_t dim) const { return strides_[dim]; }
    
    // Get number of dimensions
    size_t dim() const { return sizes_.size(); }
    
    // Check if empty (no dimensions)
    bool empty() const { return sizes_.empty(); }
    
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
        return sizes_ == other.sizes_ && strides_ == other.strides_;
    }
    
    bool operator!=(const SizesAndStrides& other) const {
        return !(*this == other);
    }
};

} // namespace tensorplay
