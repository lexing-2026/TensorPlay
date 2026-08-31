#pragma once

#include "Allocator.h"
#include "Device.h"
#include "DType.h"
#include "Macros.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace tensorplay {
namespace vulkan {

// Byte copy between any pair of {cpu, vulkan} devices.  Synchronous; used by
// copyAllocationBytes (storage resize) and copy_ kernels.
P10_API void copyHostVisibleBytes(void* destination, const Device& destination_device,
                                  const void* source, const Device& source_device,
                                  size_t nbytes);

// True when the Vulkan runtime is up (Vulkan context available).
P10_API bool is_available();

// Blocking wait until all GPU work submitted so far has completed.
P10_API void synchronize(int device);

// Device enumeration for the Python frontend.
P10_API int device_count();
P10_API std::string device_name(int device);
P10_API uint32_t device_api_version(int device);
P10_API uint64_t device_total_memory(int device);

} // namespace vulkan
} // namespace tensorplay
