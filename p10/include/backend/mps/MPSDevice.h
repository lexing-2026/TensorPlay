#pragma once

#include "Allocator.h"
#include "Device.h"
#include "Macros.h"

#include <string>

#ifdef __OBJC__
// Metal.h pulls in Foundation.h, which transitively includes CarbonCore
// headers that emit a flood of -Wdeprecated-declarations on recent macOS
// SDKs; the diagnostic is scoped to the include and popped immediately.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <Metal/Metal.h>
#pragma clang diagnostic pop
typedef id<MTLDevice> MTLDevice_t;
#else
typedef void* MTLDevice_t;
#endif

namespace tensorplay {
namespace mps {

// Helper enum to check if a MPSGraph op is supported in a given macOS version
enum class MacOSVersion : uint32_t {
  MACOS_14_4 = 0,
  MACOS_15_0,
  MACOS_15_1,
  MACOS_15_2,
  MACOS_26_0,
  MACOS_26_2,
  MACOS_26_4,
  MACOS_27_0,
};

// Helper enum for GPU-family-gated workarounds
enum class AppleGPUFamily : uint32_t {
  APPLE_7_PLUS = 1007,  // M1
  APPLE_8_PLUS = 1008,  // M2
  APPLE_9_PLUS = 1009,  // M3 / M4
  APPLE_10_PLUS = 1010, // M5
};

//-----------------------------------------------------------------
//  MPSDevice
//
// MPSDevice is a singleton class that returns the default device.
// macOS exposes a single Metal device per process, so the "device index"
// concept reduces to index 0; the class exists to keep the MTLDevice
// handle and capability queries in one place.
//-----------------------------------------------------------------
class P10_API MPSDevice {
 public:
  MPSDevice(MPSDevice& other) = delete;
  void operator=(const MPSDevice&) = delete;

  // Gets single instance of the Device.
  static MPSDevice* getInstance();

  // Returns the single device.
  MTLDevice_t device() {
    return _mtl_device;
  }

  // Returns whether running on the given macOS version or newer.
  bool isMacOS13Plus(MacOSVersion version) const;

  // Returns device name.
  std::string getName() const;

  // Returns number of GPU cores.
  unsigned getCoreCount() const;

  ~MPSDevice();

 private:
  static MPSDevice* _device;
  MTLDevice_t _mtl_device;
  MPSDevice();
};

// Whether a Metal device is usable on this machine.
P10_API bool is_available();
P10_API bool is_macos_at_least(MacOSVersion version);
P10_API bool is_apple_family_or_newer(AppleGPUFamily family);
// Whether MetalPerformancePrimitives (cooperative tensors) is usable.
P10_API bool has_mpp();
P10_API Allocator* GetMPSAllocator();

} // namespace mps
} // namespace tensorplay