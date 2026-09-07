
#include "DispatchStub.h"

#include "Macros.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "Exception.h"
#include "cpu/vec/intrinsics.h"

namespace tensorplay {
namespace cpu {

static CPUCapability compute_cpu_capability() {
  if (const char* envar = std::getenv("TP_CPU_CAPABILITY")) {
#ifdef HAVE_VSX_CPU_DEFINITION
    if (std::strcmp(envar, "vsx") == 0) {
      return CPUCapability::VSX;
    }
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
    if (std::strcmp(envar, "zvector") == 0) {
      return CPUCapability::ZVECTOR;
    }
#elif defined(HAVE_SVE_CPU_DEFINITION)
    int sve_vl = tp_cpu_sve_vector_length_bits();
    if (sve_vl > 0) {
      if (std::strcmp(envar, "sve256") == 0) {
        if (sve_vl == 256) {
          return CPUCapability::SVE256;
        }
        TP_WARN("SVE256 capability not available on hardware. Falling back to DEFAULT");
        return CPUCapability::DEFAULT;
      }
      if (std::strcmp(envar, "sve128") == 0) {
        if (sve_vl == 128) {
          return CPUCapability::SVE128;
        }
        TP_WARN("SVE128 capability not available on hardware. Falling back to DEFAULT");
        return CPUCapability::DEFAULT;
      }
      if (std::strcmp(envar, "sve") == 0) {
        if (sve_vl == 256) {
          return CPUCapability::SVE256;
        }
        if (sve_vl == 128) {
          return CPUCapability::SVE128;
        }
        TP_WARN("SVE capability not available on hardware. Falling back to DEFAULT");
        return CPUCapability::DEFAULT;
      }
    }
#else
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
#endif
    if (std::strcmp(envar, "default") == 0) {
      return CPUCapability::DEFAULT;
    }
    TP_WARN("ignoring invalid value for TP_CPU_CAPABILITY: ", envar);
  }

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  // vxe is needed for fp32 vector instructions
  if (tp_cpu_supports_zvector()) {
    return CPUCapability::ZVECTOR;
  }
#endif

#if defined(__linux__) && defined(HAVE_SVE_CPU_DEFINITION)
  if (tp_cpu_has_arm_sve_bf16()) {
    int sve_vl = tp_cpu_sve_vector_length_bits();
    if (sve_vl == 256) {
      return CPUCapability::SVE256;
    }
    if (sve_vl == 128) {
      return CPUCapability::SVE128;
    }
  }
#endif

#ifdef HAVE_RVV_CPU_DEFINITION
  // The RVVM tiers split on the vector register width: RVVM1 for VLEN=128,
  // RVVM2 for VLEN=256 (and wider).  Kernels rely on the v1.0 vector ISA.
  if (tp_cpu_has_rvv()) {
    int rvv_vl = tp_cpu_rvv_vector_length_bits();
    if (rvv_vl >= 256) {
      return CPUCapability::RVVM2;
    }
    if (rvv_vl >= 128) {
      return CPUCapability::RVVM1;
    }
  }
#endif

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

#ifdef HAVE_VSX_CPU_DEFINITION
  // Every PowerPC target this library supports executes the VSX vector
  // facilities, and the compiler gates the tier on -mvsx alone; there is no
  // further runtime probe to run, so the capability is unconditional.
  return CPUCapability::VSX;
#else
  return CPUCapability::DEFAULT;
#endif
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
#ifdef HAVE_VSX_CPU_DEFINITION
          , VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
          , ZVECTOR
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
          , SVE128
          , SVE256
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
          , RVVM1
          , RVVM2
#endif
        );
        if (!std::holds_alternative<ErrorType>(result)) {
          cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
        }
        return result;
      }
      return DispatchResult(fptr);
    }

    case DeviceType::CUDA:
      return cuda_dispatch_ptr != nullptr ? DispatchResult(cuda_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::HIP:
      return hip_dispatch_ptr != nullptr ? DispatchResult(hip_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::MPS:
      return mps_dispatch_ptr != nullptr ? DispatchResult(mps_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::MTIA:
      return mtia_dispatch_ptr != nullptr ? DispatchResult(mtia_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::HPU:
      return hpu_dispatch_ptr != nullptr ? DispatchResult(hpu_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::PrivateUse1:
      return privateuse1_dispatch_ptr != nullptr ? DispatchResult(privateuse1_dispatch_ptr) : ErrorType::MissingDeviceKernel;

#if defined(USE_XPU)
    case DeviceType::XPU:
      return xpu_dispatch_ptr != nullptr ? DispatchResult(xpu_dispatch_ptr) : ErrorType::MissingDeviceKernel;
#endif

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
#ifdef HAVE_VSX_CPU_DEFINITION
      ,
      VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      ,
      ZVECTOR
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
      ,
      SVE128
      ,
      SVE256
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
      ,
      RVVM1
      ,
      RVVM2
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
  ){

  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (TP_UNLIKELY(!AVX512)) {
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
#ifdef HAVE_VSX_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::VSX)) {
    return VSX != nullptr ? DispatchResult(VSX) : ErrorType::MissingDeviceKernel;
  }
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::ZVECTOR)) {
    return ZVECTOR != nullptr ? DispatchResult(ZVECTOR) : ErrorType::MissingDeviceKernel;
  }
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
  if (capability == static_cast<int>(CPUCapability::SVE128)) {
    if (TP_UNLIKELY(!SVE128)) {
      return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
    }
    return DispatchResult(SVE128);
  }
  if (capability == static_cast<int>(CPUCapability::SVE256)) {
    if (TP_UNLIKELY(!SVE256)) {
      return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
    }
    return DispatchResult(SVE256);
  }
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
  if (capability == static_cast<int>(CPUCapability::RVVM1)) {
    if (TP_UNLIKELY(!RVVM1)) {
      return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
    }
    return DispatchResult(RVVM1);
  }
  if (capability == static_cast<int>(CPUCapability::RVVM2)) {
    if (TP_UNLIKELY(!RVVM2)) {
      return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
    }
    return DispatchResult(RVVM2);
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
) {
  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (TP_UNLIKELY(!AVX512)) {
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
#ifdef HAVE_VSX_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::VSX)) {
    if (VSX == nullptr) {
      throw std::runtime_error("DispatchStub: missing VSX kernel");
    }
    return VSX;
  }
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::ZVECTOR)) {
    if (ZVECTOR == nullptr) {
      throw std::runtime_error("DispatchStub: missing ZVECTOR kernel");
    }
    return ZVECTOR;
  }
#endif
#ifdef HAVE_SVE_CPU_DEFINITION
  if (capability == static_cast<int>(CPUCapability::SVE128)) {
    if (SVE128 == nullptr) {
      if (DEFAULT == nullptr) {
        throw std::runtime_error("DispatchStub: missing default kernel");
      }
      return DEFAULT;
    }
    return SVE128;
  }
  if (capability == static_cast<int>(CPUCapability::SVE256)) {
    if (SVE256 == nullptr) {
      if (DEFAULT == nullptr) {
        throw std::runtime_error("DispatchStub: missing default kernel");
      }
      return DEFAULT;
    }
    return SVE256;
  }
#endif
#ifdef HAVE_RVV_CPU_DEFINITION
  if (capability == static_cast<int>(CPUCapability::RVVM1)) {
    if (RVVM1 == nullptr) {
      if (DEFAULT == nullptr) {
        throw std::runtime_error("DispatchStub: missing default kernel");
      }
      return DEFAULT;
    }
    return RVVM1;
  }
  if (capability == static_cast<int>(CPUCapability::RVVM2)) {
    if (RVVM2 == nullptr) {
      if (DEFAULT == nullptr) {
        throw std::runtime_error("DispatchStub: missing default kernel");
      }
      return DEFAULT;
    }
    return RVVM2;
  }
#endif
  if (DEFAULT == nullptr) {
    throw std::runtime_error("DispatchStub: missing default kernel");
  }
  return DEFAULT;
}

} // namespace cpu
}  // namespace tensorplay
