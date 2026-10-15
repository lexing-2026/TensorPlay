#pragma once

#include "Allocator.h"
#include "Macros.h"
#include "backend/mps/MPSDevice.h"

#include <cstdint>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <sys/types.h>

#ifndef MB
#define MB(x) (x * 1048576UL)
#endif

namespace tensorplay {
namespace mps {

// Public interface to the MPS caching allocator. The methods mirror the
// buffer-pool bookkeeping used by the Metal backend: sizes, buffer ids and
// pool state are queried by copy kernels and memory-accounting bindings.
// No Metal or Objective-C types appear here so the interface stays usable
// from plain C++ translation units.
class P10_API IMPSAllocator : public Allocator {
 public:
  // Releases cached buffers that are not referenced by live tensors.
  virtual void freeInactiveBuffers() const = 0;
  // Requested (pre-alignment) byte size of the buffer backing `ptr`.
  virtual ssize_t getUnalignedBufferSize(const void* ptr) const = 0;
  // Unique id of the buffer backing `ptr`.
  virtual uint64_t getBufferId(const void* ptr) const = 0;
  // Whether the buffer backing `ptr` uses shared (host-visible) storage.
  virtual bool isSharedBuffer(const void* ptr) const = 0;
  // Total bytes handed out to Metal across all live and cached buffers.
  virtual size_t getTotalAllocatedMemory() const = 0;
  // Bytes currently owned by live tensors (cached buffers excluded).
  virtual size_t getCurrentAllocatedMemory() const = 0;
  // Bytes retained by Metal on this process' behalf.
  virtual size_t getDriverAllocatedMemory() const = 0;
  // Recommended maximum total allocation for the current device.
  virtual size_t getRecommendedMaxMemory() const = 0;
  // Returns a previously allocated block to the pool for reuse.
  virtual void freeBuffer(void* ptr) = 0;
  // Returns the MTLBuffer backing `ptr` as an opaque pointer (bit-castable
  // to id<MTLBuffer>), or nullptr when `ptr` is not MPS-managed.
  virtual void* getMTLBuffer(const void* ptr) const = 0;
  // Releases every cached buffer back to Metal.
  virtual void emptyCache() = 0;
};

// Returns the process-wide MPS caching allocator.
P10_API IMPSAllocator* getIMPSAllocator();

// Returns the MTLBuffer (as an opaque pointer, bit-castable to id<MTLBuffer>)
// backing a host-visible pointer previously handed out by the MPS allocator.
// Returns nullptr when the pointer is not MPS-managed. Only meaningful in
// Objective-C++ translation units; the opaque void* keeps the header portable.
P10_API void* getMTLBufferFromPointer(const void* ptr);

} // namespace mps

// Global-namespace entry point used by getAllocator(DeviceType::MPS).
P10_API Allocator* getMPSAllocator();

} // namespace tensorplay