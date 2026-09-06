#pragma once

// function dispatch:
//
// Kernels that may make use of specialized instruction sets (e.g. AVX2) are
// compiled multiple times with different compiler flags (e.g. -mavx2). A
// DispatchStub contains a table of function pointers for a kernel. At runtime,
// the fastest available kernel is chosen based on the features reported by
// the CPU.
//
// Architecture tiers: on x86 the tiers are AVX2/AVX512; PowerPC builds
// dispatch to a VSX tier; s390x builds dispatch to a ZVECTOR tier (vector
// extension, requires the VXE hardware feature); aarch64 builds dispatch to
// SVE256/SVE128 tiers selected by the SVE vector length; riscv64 builds
// dispatch to RVVM1/RVVM2 tiers selected by the vector register length. The
// tier set is selected by the build system via the HAVE_*_CPU_DEFINITION
// macros, so exactly one architecture family exists in any given binary.
//
// Supported device types for registration: CPU plus accelerator backends
// (CUDA, HIP, MPS, MTIA, XPU, HPU and PrivateUse1).  A backend registers
// its kernel through the matching REGISTER_*_DISPATCH macro; each non-CPU
// device owns a plain pointer slot resolved on first call.
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
#if defined(HAVE_VSX_CPU_DEFINITION)
  VSX = 1,
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
  ZVECTOR = 1,
#elif defined(HAVE_SVE_CPU_DEFINITION)
  SVE256 = 1,
  SVE128 = 2,
#elif defined(HAVE_RVV_CPU_DEFINITION)
  RVVM2 = 1,
  RVVM1 = 2,
#else
  AVX2 = 1,
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
#ifdef HAVE_VSX_CPU_DEFINITION
      , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , void *ZVECTOR
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
      , void *SVE128
      , void *SVE256
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
      , void *RVVM1
      , void *RVVM2
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
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
    , void *SVE128
    , void *SVE256
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
    , void *RVVM1
    , void *RVVM2
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
#ifdef HAVE_VSX_CPU_DEFINITION
      , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , void *ZVECTOR
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
      , void *SVE128
      , void *SVE256
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
      , void *RVVM1
      , void *RVVM2
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
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
    , void *SVE128
    , void *SVE256
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
    , void *RVVM1
    , void *RVVM2
#endif
  );

  // Fixing dispatch error in Windows debug builds.
  #if defined(_MSC_VER) && defined(_DEBUG)
    std::atomic<void*> cpu_dispatch_ptr;
  #else
    std::atomic<void*> cpu_dispatch_ptr{nullptr};
  #endif
  // Accelerator backend slots; each stays null until a backend registers
  // a kernel via the matching REGISTER_*_DISPATCH macro.
  void* cuda_dispatch_ptr = nullptr;
  void* hip_dispatch_ptr = nullptr;
  void* mps_dispatch_ptr = nullptr;
  void* mtia_dispatch_ptr = nullptr;
  void* hpu_dispatch_ptr = nullptr;
  void* privateuse1_dispatch_ptr = nullptr;
#if defined(USE_XPU)
  void* xpu_dispatch_ptr = nullptr;
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
#ifdef HAVE_VSX_CPU_DEFINITION
      , reinterpret_cast<void*>(VSX)
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , reinterpret_cast<void*>(ZVECTOR)
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
      , reinterpret_cast<void*>(SVE128)
      , reinterpret_cast<void*>(SVE256)
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
      , reinterpret_cast<void*>(RVVM1)
      , reinterpret_cast<void*>(RVVM2)
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

  void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
    impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_hip_dispatch_ptr(FnPtr fn_ptr) {
    impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_mps_dispatch_ptr(FnPtr fn_ptr) {
    impl.mps_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_mtia_dispatch_ptr(FnPtr fn_ptr) {
    impl.mtia_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_hpu_dispatch_ptr(FnPtr fn_ptr) {
    impl.hpu_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_privateuse1_dispatch_ptr(FnPtr fn_ptr) {
    impl.privateuse1_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

#if defined(USE_XPU)
  void set_xpu_dispatch_ptr(FnPtr fn_ptr) {
    impl.xpu_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }
#endif

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
#ifdef HAVE_VSX_CPU_DEFINITION
      , reinterpret_cast<void*>(VSX)
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , reinterpret_cast<void*>(ZVECTOR)
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
      , reinterpret_cast<void*>(SVE128)
      , reinterpret_cast<void*>(SVE256)
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
      , reinterpret_cast<void*>(RVVM1)
      , reinterpret_cast<void*>(RVVM2)
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
#ifdef HAVE_VSX_CPU_DEFINITION
  static P10_API FnPtr VSX;
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  static P10_API FnPtr ZVECTOR;
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
  static P10_API FnPtr SVE128;
  static P10_API FnPtr SVE256;
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
  static P10_API FnPtr RVVM1;
  static P10_API FnPtr RVVM2;
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
  template <> name##_DECLARE_DISPATCH_type::FnPtr CPUDispatchStub<name##_DECLARE_DISPATCH_type::FnPtr, struct name##_DECLARE_DISPATCH_type>::arch = fn;

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

#ifdef HAVE_VSX_CPU_DEFINITION
#define REGISTER_VSX_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, VSX, fn)
#else
#define REGISTER_VSX_DISPATCH(name, fn)
#endif

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
#define REGISTER_ZVECTOR_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, ZVECTOR, fn)
#else
#define REGISTER_ZVECTOR_DISPATCH(name, fn)
#endif

#ifdef HAVE_SVE_CPU_DEFINITION
#define REGISTER_SVE128_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, SVE128, fn)
#define REGISTER_SVE256_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, SVE256, fn)
#else
#define REGISTER_SVE128_DISPATCH(name, fn)
#define REGISTER_SVE256_DISPATCH(name, fn)
#endif

#ifdef HAVE_RVV_CPU_DEFINITION
#define REGISTER_RVVM1_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, RVVM1, fn)
#define REGISTER_RVVM2_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, RVVM2, fn)
#else
#define REGISTER_RVVM1_DISPATCH(name, fn)
#define REGISTER_RVVM2_DISPATCH(name, fn)
#endif

// Macro to register the same kernel for all CPU arch types. This is useful
// if a kernel does not benefit from being recompiled across different arch types.
#define REGISTER_ALL_CPU_DISPATCH(name, fn)                                    \
  REGISTER_ARCH_DISPATCH(name, DEFAULT, fn)                                    \
  REGISTER_AVX512_DISPATCH(name, fn)                                           \
  REGISTER_AVX2_DISPATCH(name, fn)                                             \
  REGISTER_VSX_DISPATCH(name, fn)                                              \
  REGISTER_ZVECTOR_DISPATCH(name, fn)                                          \
  REGISTER_SVE128_DISPATCH(name, fn)                                           \
  REGISTER_SVE256_DISPATCH(name, fn)                                           \
  REGISTER_RVVM1_DISPATCH(name, fn)                                            \
  REGISTER_RVVM2_DISPATCH(name, fn)

#define REGISTER_NO_CPU_DISPATCH(name)                                         \
  REGISTER_ALL_CPU_DISPATCH(name, nullptr)

#if defined(CPU_CAPABILITY)
// REGISTER_DISPATCH now dispatches an AVX512 kernel to nullptr but registers other dispatches.
// ALSO_REGISTER_AVX512_DISPATCH should be used for ensuring AVX512 dispatch, among others.
// ALSO_REGISTER_SVE256_DISPATCH should be used for ensuring SVE256 dispatch, among others.
#ifdef CPU_CAPABILITY_AVX512
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, ((void*)(fn) ? nullptr : nullptr))
#else
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif
#define ALSO_REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#define ALSO_REGISTER_SVE128_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#define ALSO_REGISTER_SVE256_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif

// Accelerator device registration: each backend TU registers its kernel
// into the stub's device slot via a static registrar object.  The CUDA
// registrar doubles for HIP in the ROCm build, reusing the CUDA registration
// macro for the hipified call sites.
namespace {
template <typename DispatchStub>
struct RegisterCUDADispatch {
  RegisterCUDADispatch(DispatchStub& stub, typename DispatchStub::FnPtr value) {
    stub.set_cuda_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterXPUDispatch {
  RegisterXPUDispatch(DispatchStub& stub, typename DispatchStub::FnPtr value) {
    stub.set_xpu_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterHPUDispatch {
  RegisterHPUDispatch(DispatchStub& stub, typename DispatchStub::FnPtr value) {
    stub.set_hpu_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterMPSDispatch {
  RegisterMPSDispatch(DispatchStub& stub, typename DispatchStub::FnPtr value) {
    stub.set_mps_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterHIPDispatch {
  RegisterHIPDispatch(DispatchStub& stub, typename DispatchStub::FnPtr value) {
    stub.set_hip_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterMTIADispatch {
  RegisterMTIADispatch(DispatchStub& stub, typename DispatchStub::FnPtr value) {
    stub.set_mtia_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterPRIVATEUSE1Dispatch {
  RegisterPRIVATEUSE1Dispatch(DispatchStub& stub, typename DispatchStub::FnPtr value) {
    stub.set_privateuse1_dispatch_ptr(value);
  }
};
} // anonymous namespace

#define REGISTER_CUDA_DISPATCH(name, fn)                                      \
  static RegisterCUDADispatch<struct name##_DECLARE_DISPATCH_type>            \
      name##__cuda_register(name, fn);

#define REGISTER_HIP_DISPATCH(name, fn)                                       \
  static RegisterHIPDispatch<struct name##_DECLARE_DISPATCH_type>             \
      name##__hip_register(name, fn);

#define REGISTER_MPS_DISPATCH(name, fn)                                       \
  static RegisterMPSDispatch<struct name##_DECLARE_DISPATCH_type>             \
      name##__mps_register(name, fn);

#define REGISTER_MTIA_DISPATCH(name, fn)                                      \
  static RegisterMTIADispatch<struct name##_DECLARE_DISPATCH_type>            \
      name##__mtia_register(name, fn);

#define REGISTER_HPU_DISPATCH(name, fn)                                       \
  static RegisterHPUDispatch<struct name##_DECLARE_DISPATCH_type>             \
      name##__hpu_register(name, fn);

#define REGISTER_XPU_DISPATCH(name, fn)                                       \
  static RegisterXPUDispatch<struct name##_DECLARE_DISPATCH_type>             \
      name##__xpu_register(name, fn);

#define REGISTER_PRIVATEUSE1_DISPATCH(name, fn)                               \
  static RegisterPRIVATEUSE1Dispatch<struct name##_DECLARE_DISPATCH_type>     \
      name##__privateuse1_register(name, fn);

// NB: This macro must be used in an actual 'cu' file; if you try using
// it from a 'cpp' file it will not work!
#if defined(__CUDACC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(__HIPCC__)
// The ROCm build hipifies CUDA call sites, so kernel registration keeps
// going through the CUDA slot there.
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(__OBJC__) && defined(USE_MPS)
// NB: this macro must be used from a 'mm' file in order to dispatch a MPS kernel
#define REGISTER_DISPATCH(name, fn) REGISTER_MPS_DISPATCH(name, fn)
#elif defined(CPU_CAPABILITY)
// REGISTER_DISPATCH now dispatches an AVX512 kernel to nullptr but registers other dispatches.
// ALSO_REGISTER_AVX512_DISPATCH should be used for ensuring AVX512 dispatch, among others.
// ALSO_REGISTER_SVE256_DISPATCH should be used for ensuring SVE256 dispatch, among others.
#ifdef CPU_CAPABILITY_AVX512
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, ((void*)(fn) ? nullptr : nullptr))
#else
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif
#define ALSO_REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#define ALSO_REGISTER_SVE128_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#define ALSO_REGISTER_SVE256_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif

} // namespace cpu
} // namespace tensorplay