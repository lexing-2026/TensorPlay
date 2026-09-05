#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#if defined(__GLIBC__)
#include <execinfo.h>
#endif
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>
#include "DispatchKey.h"
#include "Device.h"
#include "Exception.h"
#include "Macros.h"

#include <iostream>

namespace tensorplay {

namespace autocast {
// Declared here (defined in autocast_mode.cpp) so the dispatch choke point can
// consult autocast state without pulling autocast_mode.h -- and its Tensor.h
// dependency -- into every dispatcher user.
P10_API bool is_enabled(DispatchKey autocast_key);
} // namespace autocast

// Helper to determine the backend dispatch key for a device
inline DispatchKey computeDispatchKey(const Device& device) {
    if (device.is_cuda()) return DispatchKey::CUDA;
    if (device.is_vulkan()) return DispatchKey::Vulkan;
    return DispatchKey::CPU;
}

// Generic kernel function pointer type (type-erased)
using KernelFunction = void*;

constexpr std::size_t kDispatchKeyCount =
    static_cast<std::size_t>(DispatchKey::EndOfKeys);

inline constexpr std::size_t dispatchKeyIndex(DispatchKey key) noexcept {
    return static_cast<std::size_t>(key);
}

// A stable operator table.  Registration is serialized by Dispatcher, while
// reads in the hot path are atomic and do not take the registry mutex.
struct P10_API DispatchTable {
    explicit DispatchTable(std::string name) : name(std::move(name)) {
        for (auto& kernel : kernels) {
            kernel.store(nullptr, std::memory_order_relaxed);
        }
    }

    DispatchTable(const DispatchTable&) = delete;
    DispatchTable& operator=(const DispatchTable&) = delete;

    std::string name;
    std::array<std::atomic<KernelFunction>, kDispatchKeyCount> kernels;
};

class P10_API OperatorHandle {
public:
    OperatorHandle() noexcept = default;

    KernelFunction getKernel(DispatchKey key) const noexcept {
        if (!table_ || dispatchKeyIndex(key) >= kDispatchKeyCount) {
            return nullptr;
        }
        auto kernel = table_->kernels[dispatchKeyIndex(key)].load(std::memory_order_acquire);
        // Composite fallthrough: a backend with no kernel of its own is served
        // by the backend-neutral composite registration; an explicit backend
        // kernel overrides it.
        if (!kernel && is_backend_key(key)) {
            constexpr auto kCompositeIdx = dispatchKeyIndex(DispatchKey::Composite);
            static_assert(kCompositeIdx < kDispatchKeyCount, "composite key out of range");
            kernel = table_->kernels[kCompositeIdx].load(std::memory_order_acquire);
        }
        return kernel;
    }

    const char* name() const noexcept {
        return table_ ? table_->name.c_str() : "<undefined>";
    }

    explicit operator bool() const noexcept { return table_ != nullptr; }

private:
    friend class Dispatcher;

    explicit OperatorHandle(const DispatchTable* table) noexcept : table_(table) {}

    const DispatchTable* table_ = nullptr;
};

class P10_API Dispatcher {
public:
    static Dispatcher& singleton();

    // Register a kernel for a specific operator and dispatch key
    void registerKernel(const std::string& op_name, DispatchKey key, KernelFunction kernel);

    // Get the kernel for a specific operator and dispatch key
    KernelFunction getKernel(const std::string& op_name, DispatchKey key);

    // Resolve an operator once and use the returned handle for subsequent
    // calls.  The table is owned by the process-lifetime Dispatcher singleton.
    OperatorHandle findHandle(const std::string& op_name);
    OperatorHandle findHandle(const char* op_name) {
        return findHandle(std::string(op_name));
    }

    // Introspection for debugging tools: all registered operator names.
    std::vector<std::string> operator_names() const;

    // KernelFunction for a key WITHOUT the composite fallback (null when the
    // key has no direct registration).  Purely for dump/introspection use.
    KernelFunction direct_kernel(const std::string& op_name, DispatchKey key) const;

    // Nonzero when a kernel is registered under `key` for `op_name`; a null
    // kernel pointer is a valid registration (catch-all), so presence is
    // tracked by the slot being non-null-for-registration.  Because slots
    // store the function pointer itself, a registered catch-all cannot be
    // distinguished from an empty slot; registrars must therefore always
    // store a real function pointer.  Returns false for unknown ops.
    bool has_kernel(const std::string& op_name, DispatchKey key) const;

private:
    Dispatcher() = default;

    std::unordered_map<std::string, std::unique_ptr<DispatchTable>> operators_;
    mutable std::mutex mutex_;
};
// Helper for type-safe dispatch
template<typename Return, typename... Args>
class DispatchStub {
public:
    static Return call(const OperatorHandle& handle, DispatchKey key, Args... args) {
        // Choke-point enforcement of the autocast exclusion contract: a
        // disabled Autocast key behaves as if no kernel were registered, no
        // matter which path reached here. Non-autocast keys pay one predicted
        // range compare.
        if (is_autocast_key(key) && !autocast::is_enabled(key)) [[unlikely]] {
            TP_THROW(NotImplementedError, "Autocast kernel dispatched while autocast is disabled for op: " +
                std::string(handle.name()) + " on key: " + toString(key));
        }
        auto kernel_void = handle.getKernel(key);
        if (!kernel_void) {
            if (getenv("TP_TRACE_KERNEL_MISS") != nullptr) {
                fprintf(stderr, "[kernel-miss] op=%s key=%d\n",
                        handle.name(), static_cast<int>(key));
#if defined(__GLIBC__)
                void* bt[32];
                int n = backtrace(bt, 32);
                backtrace_symbols_fd(bt, n, 2);
#endif
            }
            TP_THROW(NotImplementedError, "Kernel not found for op: " +
                std::string(handle.name()) + " on backend: " + toString(key));
        }

        using FuncType = Return(*)(Args...);
        auto kernel = reinterpret_cast<FuncType>(kernel_void);
        if constexpr (std::is_void_v<Return>) {
            kernel(std::forward<Args>(args)...);
        } else {
            return kernel(std::forward<Args>(args)...);
        }
    }

    static Return call(const std::string& op_name, DispatchKey key, Args... args) {
        return call(Dispatcher::singleton().findHandle(op_name), key,
                    std::forward<Args>(args)...);
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
