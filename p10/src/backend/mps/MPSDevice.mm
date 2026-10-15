#include "backend/mps/MPSDevice.h"

#include "backend/mps/MPSAllocator.h"
#include "Exception.h"

#include <IOKit/IOKitLib.h>
#include <string>

namespace tensorplay {
namespace mps {

MPSDevice* MPSDevice::getInstance() {
  static MPSDevice mps_device;
  return &mps_device;
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  _mtl_device = nil;
}

MPSDevice::MPSDevice() : _mtl_device(nil) {
  // The MPS backend relies on the Metal Performance Shaders Graph API.
  // Verify that the MPSGraph class and the HermiteanToRealFFTWithTensor
  // selector (introduced in macOS 14.0) are present before adopting the
  // device, matching the supported OS baseline of the backend.
  id mpsCD = NSClassFromString(@"MPSGraph");
  if ([mpsCD instancesRespondToSelector:@selector(HermiteanToRealFFTWithTensor:axes:descriptor:name:)] == NO) {
    return;
  }

  @autoreleasepool {
    NSArray* devices = [MTLCopyAllDevices() autorelease];
    for (unsigned long i = 0; i < [devices count]; i++) {
      id<MTLDevice> device = devices[i];
      if ([device isLowPower]) { // exclude Intel GPUs
        continue;
      }
      if (![device supportsFamily:MTLGPUFamilyMac2]) {
        // Exclude devices that do not support Metal 2.0; virtualised MPS
        // devices on macOS 12.6 fail this check.
        continue;
      }
      _mtl_device = [device retain];
      break;
    }
  }
  if (!_mtl_device) {
    // No usable Metal device; the backend stays unavailable. Callers gate
    // on is_available() so this is a soft failure, not a hard abort.
    return;
  }
}

bool MPSDevice::isMacOS13Plus(MacOSVersion version) const {
  auto is_os_version_at_least = [](int major, int minor) {
    @autoreleasepool {
      NSProcessInfo* processInfo = [[NSProcessInfo new] autorelease];
      return [processInfo
          isOperatingSystemAtLeastVersion:{.majorVersion = major, .minorVersion = minor, .patchVersion = 0}];
    }
  };
  static bool _macos_14_4_plus = is_os_version_at_least(14, 4);
  static bool _macos_15_0_plus = is_os_version_at_least(15, 0);
  static bool _macos_15_1_plus = is_os_version_at_least(15, 1);
  static bool _macos_15_2_plus = is_os_version_at_least(15, 2);
  static bool _macos_26_0_plus = is_os_version_at_least(26, 0);
  static bool _macos_26_2_plus = is_os_version_at_least(26, 2);
  static bool _macos_26_4_plus = is_os_version_at_least(26, 4);
  static bool _macos_27_0_plus = is_os_version_at_least(27, 0);

  switch (version) {
    case MacOSVersion::MACOS_14_4:
      return _macos_14_4_plus;
    case MacOSVersion::MACOS_15_0:
      return _macos_15_0_plus;
    case MacOSVersion::MACOS_15_1:
      return _macos_15_1_plus;
    case MacOSVersion::MACOS_15_2:
      return _macos_15_2_plus;
    case MacOSVersion::MACOS_26_0:
      return _macos_26_0_plus;
    case MacOSVersion::MACOS_26_2:
      return _macos_26_2_plus;
    case MacOSVersion::MACOS_26_4:
      return _macos_26_4_plus;
    case MacOSVersion::MACOS_27_0:
      return _macos_27_0_plus;
    default:
      return false;
  }
}

std::string MPSDevice::getName() const {
  @autoreleasepool {
    return [[_mtl_device name] UTF8String];
  }
}

unsigned MPSDevice::getCoreCount() const {
  io_iterator_t iterator = 0;
  io_registry_entry_t entry = 0;
  int core_count = 0;
  auto matchingDict = IOServiceMatching("AGXAccelerator");
  if (!matchingDict) {
    return 0;
  }
  const auto status = IOServiceGetMatchingServices(kIOMainPortDefault, matchingDict, &iterator);
  if (status != KERN_SUCCESS) {
    return 0;
  }
  while ((entry = IOIteratorNext(iterator)) != 0) {
    auto property = IORegistryEntryCreateCFProperty(entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    if (property) {
      auto found = CFNumberGetValue(static_cast<CFNumberRef>(property), kCFNumberIntType, &core_count);
      CFRelease(property);
      if (found) {
        break;
      }
    }
    IOObjectRelease(entry);
  }
  IOObjectRelease(iterator);
  return core_count;
}

Allocator* GetMPSAllocator() {
  return getMPSAllocator();
}

bool is_available() {
  return MPSDevice::getInstance()->device() != nil;
}

bool is_macos_at_least(MacOSVersion version) {
  return MPSDevice::getInstance()->isMacOS13Plus(version);
}

bool is_apple_family_or_newer(AppleGPUFamily family) {
  auto mtl_family = static_cast<MTLGPUFamily>(family);
  return [MPSDevice::getInstance()->device() supportsFamily:mtl_family];
}

bool has_mpp() {
  // MetalPerformancePrimitives matmul2d (cooperative tensors) needs macOS 26.2+.
  return is_macos_at_least(MacOSVersion::MACOS_26_2);
}

} // namespace mps
} // namespace tensorplay