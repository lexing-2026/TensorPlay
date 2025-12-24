#pragma once

#include <functional>
#include <utility>
#include <iostream>
#include "Device.h"

namespace tensorplay {

// A Deleter function type: std::function for flexibility
using DeleterFn = std::function<void(void*)>;

// A simple deleter that does nothing
inline void deleteNothing(void*) {}

// A simple deleter for C++ new[]
inline void deleteCPP(void* data) {
    delete[] static_cast<char*>(data);
}

// DataPtr is a move-only smart pointer that manages a pointer to data and its deleter.
// It is similar to std::unique_ptr but with a type-erased deleter and context.
class DataPtr {
private:
    void* data_;
    DeleterFn deleter_;

public:
    Device device_;

    DataPtr() : data_(nullptr), deleter_(nullptr), device_(DeviceType::Unknown) {}

    DataPtr(void* data, DeleterFn deleter, Device device)
        : data_(data), deleter_(deleter), device_(device) {}

    // Move constructor
    DataPtr(DataPtr&& other) noexcept : DataPtr() {
        swap(other);
    }

    // Move assignment
    DataPtr& operator=(DataPtr&& other) noexcept {
        if (this != &other) {
            clear();
            swap(other);
        }
        return *this;
    }

    // No copy
    DataPtr(const DataPtr&) = delete;
    DataPtr& operator=(const DataPtr&) = delete;

    ~DataPtr() {
        clear();
    }

    void clear() {
        if (deleter_) {
            deleter_(data_);
        }
        data_ = nullptr;
        deleter_ = nullptr;
    }

    // Swap method
    void swap(DataPtr& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(deleter_, other.deleter_);
        std::swap(device_, other.device_);
    }

    // Accessors with const correctness
    void* get() { 
        check_device();
        return data_; 
    }
    
    const void* get() const { 
        check_device();
        return data_; 
    }

    // Helper to cast
    template<typename T>
    T* cast() { 
        check_device();
        return static_cast<T*>(data_); 
    }

    template<typename T>
    const T* cast() const { 
        check_device();
        return static_cast<const T*>(data_); 
    }
    
    // Release ownership
    void* release() {
        void* ptr = data_;
        data_ = nullptr;
        deleter_ = nullptr;
        return ptr;
    }
    
    operator bool() const { return data_ != nullptr; }

private:
    void check_device() const {
        // Debug check for device validity
#ifndef NDEBUG
        if (data_ && device_.type() == DeviceType::Unknown) {
            std::cerr << "Warning: Accessing DataPtr with Unknown device!" << std::endl;
        }
#endif
    }
};

} // namespace tensorplay
