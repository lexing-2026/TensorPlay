#pragma once

#include "DataPtr.h"
#include "Device.h"
#include "Macros.h"

namespace tensorplay {

class P10_API Allocator {
public:
    virtual ~Allocator() = default;
    virtual DataPtr allocate(size_t nbytes) const = 0;
};

// Get the allocator for a specific device
P10_API Allocator* getAllocator(DeviceType t);
P10_API Allocator* getCPUAllocator();

} // namespace tensorplay
