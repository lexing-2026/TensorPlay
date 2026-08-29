
#include "DispatchStub.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#include "cpu/vec/intrinsics.h"

namespace tensorplay {
namespace cpu {

static CPUCapability compute_cpu_capability() {
  if (const char* envar = std::getenv("TP_CPU_CAPABILITY")) {
#ifdef HAVE_AVX512_CPU_DEFINITION
    if (std::strcmp(envar, "avx512") == 0) {
      return CPUCapability::AVX512;
    }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    if (std::strcmp(envar, "avx2") == 0) {
      return CPUCapability::AVX2;
    }
#endif
    if (std::strcmp(envar, "default") == 0) {
      return CPUCapability::DEFAULT;
    }
  }

#ifdef HAVE_AVX512_CPU_DEFINITION
  // GCC supports some AVX512 intrinsics such as _mm512_set_epi16 only in
  // versions 9 & beyond. So, we want to ensure that only releases built with
  // supported compilers on supported hardware return CPU Capability AVX512,
  // if it's supported on the hardware TensorPlay is running on.
  if (tp_cpu_supports_avx512() && tp_cpu_supports_avx2()) {
    return CPUCapability::AVX512;
  }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  if (tp_cpu_supports_avx2()) {
    return CPUCapability::AVX2;
  }
#endif

  return CPUCapability::DEFAULT;
}

CPUCapability get_cpu_capability() {
  static CPUCapability capability = compute_cpu_capability();
  return capability;
}

DispatchResult CPUDispatchStubImpl::try_get_call_ptr(
  const DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
) {
  switch (device_type) {
    case DeviceType::CPU: {
      // Use memory_order_relaxed here since even if two threads race,
      // they will still compute the same value for cpu_dispatch_ptr.
      auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
      if (!fptr) {
        auto result = try_choose_cpu_impl(
          DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
          , AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
          , AVX2
#endif
        );
        if (!std::holds_alternative<ErrorType>(result)) {
          cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
        }
        return result;
      }
      return DispatchResult(fptr);
    }
    default:
      return ErrorType::DeviceNotSupported;
  }
}

void* CPUDispatchStubImpl::get_call_ptr(
  const DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
) {
  auto result = try_get_call_ptr(
      device_type,
      DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      ,
      AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      ,
      AVX2
#endif
  );
  if (std::holds_alternative<ErrorType>(result)) {
    auto error = std::get<ErrorType>(result);
    switch (error) {
      case ErrorType::MissingDeviceKernel:
        throw std::runtime_error("DispatchStub: missing kernel for device type");
      case ErrorType::DeviceNotSupported:
        throw std::runtime_error("DispatchStub: unsupported device type");
    }
  }

  void* fptr = std::get<void*>(result);
  return fptr;
}

DispatchResult CPUDispatchStubImpl::try_choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
  ){

  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (__builtin_expect(!AVX512, 0)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      return AVX2 != nullptr ? DispatchResult(AVX2) : ErrorType::MissingDeviceKernel;
    } else {
      return DispatchResult(AVX512);
    }
  }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    return AVX2 != nullptr ? DispatchResult(AVX2) : ErrorType::MissingDeviceKernel;
  }
#endif
  return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
}

void* CPUDispatchStubImpl::choose_cpu_impl(
  void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
) {
  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (__builtin_expect(!AVX512, 0)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      if (AVX2 == nullptr) {
        throw std::runtime_error("DispatchStub: missing AVX2 kernel");
      }
      return AVX2;
    } else {
      return AVX512;
    }
  }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    if (AVX2 == nullptr) {
      throw std::runtime_error("DispatchStub: missing AVX2 kernel");
    }
    return AVX2;
  }
#endif
  if (DEFAULT == nullptr) {
    throw std::runtime_error("DispatchStub: missing default kernel");
  }
  return DEFAULT;
}

} // namespace cpu
}  // namespace tensorplay