#pragma once

#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <random>
#include <cstring>
#include "Macros.h"
#include "DType.h"
#include "Device.h"
#include "VariableVersion.h"
#include "Storage.h"
#include "SizesAndStrides.h"

namespace tensorplay {

class Tensor;

class P10_API TensorImpl {
private:
    Storage storage_;
    size_t storage_offset_;
    SizesAndStrides sizes_and_strides_;
    DType dtype_;
    Device device_;
    tensorplay::VariableVersion version_counter_;

    bool is_contiguous_;
    bool is_channels_last_;
    bool is_autograd_shared_ = false; // Flag for Autograd optimizations

    // Opaque pointer to OneDNN memory descriptor (std::shared_ptr<dnnl::memory::desc>)
    // std::shared_ptr<void> onednn_md_;
    
    struct SharedState {
        Storage storage;
        std::shared_ptr<void> onednn_md;
        std::shared_ptr<void> onednn_memory_cache; // Cache for OneDNN memory object (reordered)
    };
    std::shared_ptr<SharedState> shared_state_;

public:
    TensorImpl();
    TensorImpl(const std::vector<int64_t>& sizes, DType dtype, const Device& device = Device());
    TensorImpl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, const Device& device = Device());
    TensorImpl(Storage storage, const std::vector<int64_t>& sizes, DType dtype, size_t storage_offset = 0);
    TensorImpl(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset = 0);
    
    // Copy/Move
    TensorImpl(const TensorImpl& other);
    TensorImpl(TensorImpl&& other) noexcept;
    TensorImpl& operator=(const TensorImpl& other);
    TensorImpl& operator=(TensorImpl&& other) noexcept;
    
    ~TensorImpl() = default;

    // Storage
    const Storage& storage() const { return shared_state_->storage; }
    size_t storage_offset() const { return storage_offset_; }
    void set_storage(Storage storage) { shared_state_->storage = storage; }
    void set_storage_offset(size_t offset) { storage_offset_ = offset; }
    bool has_storage() const { return shared_state_ && shared_state_->storage.defined(); }
    
    // Metadata
    const SizesAndStrides& sizes_and_strides() const { return sizes_and_strides_; }
    const std::vector<int64_t>& sizes() const { return sizes_and_strides_.sizes(); }
    const std::vector<int64_t>& strides() const { return sizes_and_strides_.strides(); }
    int64_t size(size_t dim) const { return sizes_and_strides_.size(dim); }
    int64_t stride(size_t dim) const { return sizes_and_strides_.stride(dim); }
    size_t dim() const { return sizes_and_strides_.dim(); }
    int64_t numel() const { return sizes_and_strides_.numel(); }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    
    size_t itemsize() const { return elementSize(dtype_); }
    bool is_contiguous() const { return is_contiguous_; }
    
    uint32_t version() const { return version_counter_.current_version(); }
    
    // Copy metadata (storage, sizes, strides, dtype, device) from another TensorImpl
    // preserving autograd_meta
    void copy_metadata_from(const TensorImpl& other);
    
    // Access to raw data pointer (typed)
    template<typename T>
    T* data() const {
        if (!has_storage()) return nullptr;
        // Check type?
        return static_cast<T*>(shared_state_->storage.data()) + storage_offset_; 
    }
    
    void* data() const {
        if (!has_storage()) return nullptr;
        size_t elem_size = elementSize(dtype_);
        return static_cast<char*>(shared_state_->storage.data()) + storage_offset_ * elem_size;
    }

    void set_onednn_md(std::shared_ptr<void> md) { shared_state_->onednn_md = md; }
    std::shared_ptr<void> get_onednn_md() const { return shared_state_->onednn_md; }
    bool has_onednn_md() const { return shared_state_->onednn_md != nullptr; }

    void set_onednn_memory_cache(std::shared_ptr<void> mem) { shared_state_->onednn_memory_cache = mem; }
    std::shared_ptr<void> get_onednn_memory_cache() const { return shared_state_->onednn_memory_cache; }
    bool has_onednn_memory_cache() const { return shared_state_->onednn_memory_cache != nullptr; }

    void set_autograd_shared(bool shared) { is_autograd_shared_ = shared; }
    bool is_autograd_shared() const { return is_autograd_shared_; }
    
    // Share storage state from another TensorImpl
    void share_storage_from(const TensorImpl& other) {
        shared_state_ = other.shared_state_;
    }
};

} // namespace tensorplay
