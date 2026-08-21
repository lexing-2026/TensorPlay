#pragma once

// Port of ATen/native/DispatchStub.h. Implements instruction-set specific
// function dispatch:
//
// Kernels that may make use of specialized instruction sets (e.g. AVX2) are
// compiled multiple times with different compiler flags (e.g. -mavx2). A
// DispatchStub contains a table of function pointers for a kernel. At runtime,
// the fastest available kernel is chosen based on the features reported by
// the CPU.
//
// Example:
//
// In native/MyKernel.h:
//   using fn_type = void(*)(const Tensor& x);
//   DECLARE_DISPATCH(fn_type, stub)
//
// In native/MyKernel.cpp
//   DEFINE_DISPATCH(stub);
//
// In native/cpu/MyKernel.cpp:
//   namespace {
//     // use anonymous namespace so that different cpu versions won't conflict
//     void kernel(const Tensor& x) { ... }
//   }
//   REGISTER_DISPATCH(stub, &kernel);
//
// To call:
//   stub(DeviceType::CPU, tensor);

#include <atomic>
#include <utility>
#include <variant>

#include "Device.h"
#include "Macros.h"

namespace tensorplay {
namespace cpu {

enum class CPUCapability {
  DEFAULT = 0,
#if defined(HAVE_AVX2_CPU_DEFINITION)
  AVX2 = 1,
#endif
#if defined(HAVE_AVX512_CPU_DEFINITION)
  AVX512 = 2,
#endif
  NUM_OPTIONS
};

// Enum for error types
enum class ErrorType {
  MissingDeviceKernel,
  DeviceNotSupported
};

// Alias for the return type using std::variant
using DispatchResult = std::variant<void*, ErrorType>;

P10_API CPUCapability get_cpu_capability();

template <typename FnPtr, typename T>
struct CPUDispatchStub;

/**
 * The sole purpose of this class is to outline methods that don't need to be
 * specialized or otherwise inlined and duplicated (by the compiler due to
 * template expansion), since it causes size bloat if there are a significant
 * number of specialization of the DispatchStub<> class.
 */
struct P10_API CPUDispatchStubImpl {

  // The DispatchStubImpl::try_get_call_ptr() method is used to get the call
  // pointer for a given device type. If the call pointer is not found,
  // DispatchStubImpl::try_get_call_ptr() returns an ErrorType.
  // The main difference between try_get_call_ptr() and get_call_ptr() is that
  // try_get_call_ptr() will return the ErrorType and not raise an exception.
  DispatchResult try_get_call_ptr(
    DeviceType device_type
    , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , void *AVX2
#endif
  );

  // Analogous to try_get_call_ptr(), but it will return the ErrorType and not
  // raise an exception.
  DispatchResult try_choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
  );

  void* get_call_ptr(
    DeviceType device_type
    , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , void *AVX2
#endif
  );

  /**
   * The CPU Dispatch actual method is chosen in decreasing order of preference by
   * DispatchStubImpl::choose_cpu_impl() in case none is found by
   * DispatchStubImpl::get_call_ptr() in cpu_dispatch_ptr.
   */
  void* choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
  );

  // Fixing dispatch error in Windows debug builds.
  // See https://github.com/pytorch/pytorch/issues/22681 for more details.
  #if defined(_MSC_VER) && defined(_DEBUG)
    std::atomic<void*> cpu_dispatch_ptr;
  #else
    std::atomic<void*> cpu_dispatch_ptr{nullptr};
  #endif
};

template <typename rT, typename T, typename... Args>
struct CPUDispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  CPUDispatchStub() = default;
  CPUDispatchStub(const CPUDispatchStub&) = delete;
  CPUDispatchStub& operator=(const CPUDispatchStub&) = delete;

private:
  FnPtr get_call_ptr(const DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX512_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX512)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX2)
#endif
      )
    );
  }

public:
  template <typename... ArgTypes>
  rT operator()(DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

  // Returns true if the dispatcher has a kernel registered for this device
  // type.
  bool is_device_supported(const DeviceType device_type) {
    auto result = impl.try_get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX512_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX512)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX2)
#endif
      );
    if (std::holds_alternative<ErrorType>(result)){
      return false;
    }
    return true;
  }

  static P10_API FnPtr DEFAULT;
#ifdef HAVE_AVX512_CPU_DEFINITION
  static P10_API FnPtr AVX512;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  static P10_API FnPtr AVX2;
#endif
private:
  CPUDispatchStubImpl impl;
};

#define DECLARE_DISPATCH(fn, name)                                                         \
  struct name##_DECLARE_DISPATCH_type : CPUDispatchStub<fn, name##_DECLARE_DISPATCH_type> {   \
    name##_DECLARE_DISPATCH_type() = default;                                              \
    name##_DECLARE_DISPATCH_type(const name##_DECLARE_DISPATCH_type&) = delete;            \
    name##_DECLARE_DISPATCH_type& operator=(const name##_DECLARE_DISPATCH_type&) = delete; \
    name##_DECLARE_DISPATCH_type(name##_DECLARE_DISPATCH_type&&) = delete;                 \
    name##_DECLARE_DISPATCH_type& operator=(name##_DECLARE_DISPATCH_type&&) = delete;      \
    ~name##_DECLARE_DISPATCH_type() = default;                                             \
  };                                                                                       \
  extern P10_API struct name##_DECLARE_DISPATCH_type name;

#define DEFINE_DISPATCH(name) struct name##_DECLARE_DISPATCH_type name

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <> name##_DECLARE_DISPATCH_type::FnPtr P10_API CPUDispatchStub<name##_DECLARE_DISPATCH_type::FnPtr, struct name##_DECLARE_DISPATCH_type>::arch = fn;

#ifdef HAVE_AVX512_CPU_DEFINITION
#define REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX512, fn)
#else
#define REGISTER_AVX512_DISPATCH(name, fn)
#endif

#ifdef HAVE_AVX2_CPU_DEFINITION
#define REGISTER_AVX2_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX2, fn)
#else
#define REGISTER_AVX2_DISPATCH(name, fn)
#endif

// Macro to register the same kernel for all CPU arch types. This is useful
// if a kernel does not benefit from being recompiled across different arch types.
#define REGISTER_ALL_CPU_DISPATCH(name, fn)                                    \
  REGISTER_ARCH_DISPATCH(name, DEFAULT, fn)                                    \
  REGISTER_AVX512_DISPATCH(name, fn)                                           \
  REGISTER_AVX2_DISPATCH(name, fn)

#define REGISTER_NO_CPU_DISPATCH(name)                                         \
  REGISTER_ALL_CPU_DISPATCH(name, nullptr)

#if defined(CPU_CAPABILITY)
// REGISTER_DISPATCH now dispatches an AVX512 kernel to nullptr but registers other dispatches.
// ALSO_REGISTER_AVX512_DISPATCH should be used for ensuring AVX512 dispatch, among others.
#ifdef CPU_CAPABILITY_AVX512
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, ((void*)(fn) ? nullptr : nullptr))
#else
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif
#define ALSO_REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif

} // namespace cpu
} // namespace tensorplay