#pragma once

#include <memory>
#include "StorageImpl.h"
#include "Macros.h"

namespace tensorplay {

class P10_API Storage {
public:
    Storage() = default;
    
    // Create new storage with size
    explicit Storage(size_t nbytes, Allocator* allocator = nullptr) {
        if (!allocator) allocator = getCPUAllocator();
        impl_ = std::make_shared<StorageImpl>(nbytes, allocator);
    }
    
    // Create storage from existing DataPtr
    Storage(DataPtr&& data_ptr, size_t nbytes, Allocator* allocator = nullptr) {
        // If allocator is provided, we assume it produced the data or can handle it? 
        // Usually if we wrap existing data, we don't have an allocator that produced it in the same sense.
        // We set resizable to false by default for wrapped data.
        impl_ = std::make_shared<StorageImpl>(std::move(data_ptr), nbytes, allocator, false);
    }

    // Accessors
    void* data() const { return impl_ ? impl_->data() : nullptr; }
    
    template<typename T>
    T* data() const { return static_cast<T*>(data()); }
    
    size_t nbytes() const { return impl_ ? impl_->nbytes : 0; }
    
    Device device() const { return impl_ ? impl_->data_ptr.device_ : Device(DeviceType::CPU); }
    
    // Validity
    bool defined() const { return impl_ != nullptr; }
    
    // Resize (only if resizable)
    void set_nbytes(size_t new_nbytes) {
        if (!impl_) {
             impl_ = std::make_shared<StorageImpl>(new_nbytes, getCPUAllocator());
             return;
        }
        impl_->set_nbytes(new_nbytes);
    }
    
    Allocator* allocator() const { return impl_ ? impl_->allocator : nullptr; }
    
    // Use count for debugging
    long use_count() const { return impl_.use_count(); }

private:
    std::shared_ptr<StorageImpl> impl_;
};

} // namespace tensorplay
