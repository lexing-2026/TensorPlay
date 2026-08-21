#pragma once

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
