#pragma once

#include <cstdint>

#include "DataPtr.h"
#include "Device.h"
#include "Macros.h"

namespace tensorplay {

class P10_API Allocator {
public:
    virtual ~Allocator() = default;
    virtual DataPtr allocate(size_t nbytes) const = 0;
    virtual DataPtr allocate(size_t nbytes, const Device& device) const {
        (void)device;
        return allocate(nbytes);
    }
};

// Get the allocator for a specific device
P10_API Allocator* getAllocator(DeviceType t);

// Installs the allocator of a device type that has none of its own in this
// build: the out-of-tree PrivateUse1 slot, or an in-tree type whose backend
// was not compiled in.  ``allocator`` must outlive the process.  A later
// registration replaces an earlier one only with an equal or higher
// ``priority``, so a wrapping allocator (a memory tracker, say) can claim
// the slot over the backend's own.
P10_API void SetAllocator(DeviceType t, Allocator* allocator, uint8_t priority = 0);

namespace detail {
struct AllocatorRegisterer {
    AllocatorRegisterer(DeviceType t, Allocator* allocator, uint8_t priority = 0) {
        SetAllocator(t, allocator, priority);
    }
};
} // namespace detail

// Registers an allocator at load time:
//   static MyAllocator allocator;
//   REGISTER_ALLOCATOR(DeviceType::PrivateUse1, &allocator);
#define REGISTER_ALLOCATOR(device_type, allocator_ptr, ...)                   \
    static ::tensorplay::detail::AllocatorRegisterer                         \
        TP_CONCAT(tensorplay_allocator_registerer_, __LINE__)(               \
            device_type, allocator_ptr, ##__VA_ARGS__)
P10_API Allocator* getCPUAllocator();
P10_API void copyAllocationBytes(void* destination, const Device& destination_device,
                                 const void* source, const Device& source_device,
                                 size_t nbytes);

#ifdef USE_CUDA
// Pageable CPU memory cannot participate in genuinely asynchronous host/device
// copies.  The pinned allocator is backed by cudaHostAlloc and keeps released
// blocks alive until every recorded CUDA stream has finished using them.
P10_API Allocator* getPinnedMemoryAllocator();
#endif

} // namespace tensorplay
