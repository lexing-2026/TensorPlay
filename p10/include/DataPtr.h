#pragma once

#include <utility>
#include <iostream>
#include "Device.h"

namespace tensorplay {

// A Deleter function type: plain function pointer (mirrors c10::DeleterFnPtr).
// Unlike std::function this is trivially copyable, comparable, and never
// allocates. Stateful deleters carry their state through `ctx` instead of a
// capturing lambda.
using DeleterFnPtr = void (*)(void*);

// A simple deleter that does nothing
inline void deleteNothing(void*) {}

// A simple deleter for C++ new[]
inline void deleteCPP(void* data) {
    delete[] static_cast<char*>(data);
}

// DataPtr is a move-only smart pointer that manages a pointer to data and its
// deleter. It is similar to std::unique_ptr but with a type-erased deleter and
// context, mirroring c10::DataPtr / c10::UniqueVoidPtr: `deleter_` is invoked
// with `ctx_` (not `data_`) which allows the deleter to be a capture-less
// function pointer while still carrying arbitrary state (e.g. an allocator
// block or an owned PyObject*).
class DataPtr {
private:
    void* data_;
    void* ctx_;
    DeleterFnPtr deleter_;

public:
    DataPtr() : data_(nullptr), ctx_(nullptr), deleter_(nullptr), device_(DeviceType::Unknown) {}

    DataPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
        : data_(data), ctx_(ctx), deleter_(ctx_deleter), device_(device) {}

    // Convenience constructor for stateless deleters where the context is the
    // data pointer itself.
    DataPtr(void* data, DeleterFnPtr deleter, Device device)
        : data_(data), ctx_(data), deleter_(deleter), device_(device) {}

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
            deleter_(ctx_);
        }
        data_ = nullptr;
        ctx_ = nullptr;
        deleter_ = nullptr;
    }
    // Swap method
    void swap(DataPtr& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(ctx_, other.ctx_);
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

    // The context passed to the deleter (mirrors c10::DataPtr::get_context()).
    void* get_context() const { return ctx_; }

    DeleterFnPtr get_deleter() const { return deleter_; }

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

    // Release ownership: the caller takes responsibility for freeing the
    // context. The deleter is dropped without being invoked.
    void* release() {
        void* ptr = data_;
        data_ = nullptr;
        ctx_ = nullptr;
        deleter_ = nullptr;
        return ptr;
    }

    operator bool() const { return data_ != nullptr; }

    Device device_;

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
