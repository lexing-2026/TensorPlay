#pragma once

#include "StorageImpl.h"
#include "Macros.h"

namespace tensorplay {

class P10_API Storage {
public:
    Storage() = default;

    // Create new storage with size
    explicit Storage(size_t nbytes, Allocator* allocator = nullptr) {
        if (!allocator) allocator = getCPUAllocator();
        impl_ = make_intrusive<StorageImpl>(nbytes, allocator);
    }

    Storage(size_t nbytes, Allocator* allocator, const Device& device) {
        if (!allocator) allocator = getCPUAllocator();
        impl_ = make_intrusive<StorageImpl>(nbytes, allocator, device);
    }

    // Create storage from existing DataPtr
    Storage(DataPtr&& data_ptr, size_t nbytes, Allocator* allocator = nullptr) {
        // If allocator is provided, we assume it produced the data or can handle it?
        // Usually if we wrap existing data, we don't have an allocator that produced it in the same sense.
        // We set resizable to false by default for wrapped data.
        impl_ = make_intrusive<StorageImpl>(std::move(data_ptr), nbytes, allocator, false);
    }

    // Accessors
    void* data() const { return impl_ ? impl_->data() : nullptr; }

    template<typename T>
    T* data() const { return static_cast<T*>(data()); }

    size_t nbytes() const { return impl_ ? impl_->nbytes : 0; }

    Device device() const { return impl_ ? impl_->data_ptr.device_ : Device(DeviceType::CPU); }

    // Validity
    bool defined() const { return impl_ != nullptr; }

    // Identity: true when both storages wrap the same StorageImpl (a view
    // always shares its base's StorageImpl).
    bool is_same(const Storage& other) const { return impl_ == other.impl_; }

    // Resize (only if resizable)
    bool resizable() const { return impl_ && impl_->resizable; }

    void set_nbytes(size_t new_nbytes) {
        if (!impl_) {
             impl_ = make_intrusive<StorageImpl>(new_nbytes, getCPUAllocator());
             return;
        }
        impl_->set_nbytes(new_nbytes);
    }

    Allocator* allocator() const { return impl_ ? impl_->allocator : nullptr; }

    // Use count for debugging
    uint32_t use_count() const { return impl_.use_count(); }

    // Borrows the underlying StorageImpl.  For backend internals only; the
    // handle stays owned by this Storage.
    const intrusive_ptr<StorageImpl>& unsafeGetStorageImpl() const {
        return impl_;
    }

private:
    intrusive_ptr<StorageImpl> impl_;
};

} // namespace tensorplay
