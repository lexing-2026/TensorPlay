#pragma once

#include <unordered_map>
#include <string>
#include <mutex>
#include "DispatchKey.h"
#include "Device.h"
#include "Exception.h"
#include "Macros.h"

#include <iostream>

namespace tensorplay {

// Helper to determine dispatch key
inline DispatchKey computeDispatchKey(const Device& device) {
    if (device.is_cuda()) return DispatchKey::CUDA;
    return DispatchKey::CPU;
}

// Generic kernel function pointer type (type-erased)
using KernelFunction = void*;

class P10_API Dispatcher {
public:
    static Dispatcher& singleton();

    // Register a kernel for a specific operator and dispatch key
    void registerKernel(const std::string& op_name, DispatchKey key, KernelFunction kernel);

    // Get the kernel for a specific operator and dispatch key
    KernelFunction getKernel(const std::string& op_name, DispatchKey key);

private:
    Dispatcher() = default;
    
    struct OpDispatchTable {
        std::unordered_map<DispatchKey, KernelFunction> kernels;
    };

    std::unordered_map<std::string, OpDispatchTable> operators_;
    std::mutex mutex_;
};

// Helper for type-safe dispatch
template<typename Return, typename... Args>
class DispatchStub {
public:
    static Return call(const std::string& op_name, DispatchKey key, Args... args) {
        auto kernel_void = Dispatcher::singleton().getKernel(op_name, key);
        if (!kernel_void) {
            TP_THROW(NotImplementedError, "Kernel not found for op: " + op_name + " on backend: " + toString(key));
        }
        
        using FuncType = Return(*)(Args...);
        auto kernel = reinterpret_cast<FuncType>(kernel_void);
        return kernel(std::forward<Args>(args)...);
    }
};

// Macro for registration (DEPRECATED: Use TENSORPLAY_LIBRARY_IMPL instead)
#define TENSORPLAY_REGISTER_KERNEL(OP_NAME, KEY, FUNC) \
    static struct Register##OP_NAME##KEY { \
        Register##OP_NAME##KEY() { \
            ::tensorplay::Dispatcher::singleton().registerKernel(#OP_NAME, ::tensorplay::DispatchKey::KEY, (::tensorplay::KernelFunction)FUNC); \
        } \
    } register_##OP_NAME##KEY;

// --------------------------------------------------------------------------
// Library API (Optimization for bulk registration)
// --------------------------------------------------------------------------

class Library {
public:
    explicit Library(DispatchKey key) : key_(key) {}
    
    // Type-safe registration helper
    template<typename Func>
    Library& impl(const std::string& name, Func func) {
        Dispatcher::singleton().registerKernel(name, key_, (KernelFunction)func);
        return *this;
    }

private:
    DispatchKey key_;
};

#define TENSORPLAY_LIBRARY_IMPL(KEY, NAME) \
    static void TP_CONCAT(tensorplay_library_init_, NAME)(::tensorplay::Library&); \
    static struct TP_CONCAT(TensorPlayLibraryInit_, NAME) { \
        TP_CONCAT(TensorPlayLibraryInit_, NAME)() { \
            ::tensorplay::Library lib(::tensorplay::DispatchKey::KEY); \
            TP_CONCAT(tensorplay_library_init_, NAME)(lib); \
        } \
    } TP_CONCAT(tensorplay_library_init_instance_, NAME); \
    static void TP_CONCAT(tensorplay_library_init_, NAME)(::tensorplay::Library& m)


} // namespace tensorplay
