#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../api/Context.h"

#include "Allocator.h"
#include "DataPtr.h"
#include "Exception.h"

namespace tensorplay {

namespace {

using StoragePtr = vulkan::ops::detail::StoragePtr;

void delete_vulkan_allocation(void* ctx) {
  delete static_cast<StoragePtr*>(ctx);
}

//
// The dense Vulkan allocator.  Each allocation owns one vTensorStorage (a
// VkImage texture or VkBuffer backed by device memory); the DataPtr carries
// the shared_ptr as the deleter context so the resource is released through
// the context clearlist only after in-flight GPU work has drained.
//
class VulkanCachingAllocator : public Allocator {
 public:
  DataPtr allocate(size_t nbytes) const override {
    TP_THROW(
        RuntimeError,
        "Vulkan allocator requires an explicit device; use allocate(nbytes, "
        "device)");
  }

  DataPtr allocate(size_t nbytes, const Device& device) const override {
    const int64_t index = device.index() < 0 ? 0 : device.index();

    vulkan::api::vTensorStorage* storage = allocate_storage(nbytes, index);

    // The shared_ptr takes ownership; no reset() afterwards (that would
    // release and re-adopt the same pointer).
    auto* holder = new StoragePtr(storage);

    // The DataPtr value carries the owning VulkanBuffer object; the object
    // stays alive because the DataPtr context holds the vTensorStorage
    // shared_ptr.  copyHostVisibleBytes recovers the object from this value.
    void* opaque_handle = storage->buffer().has_memory()
        ? static_cast<void*>(&storage->buffer())
        : nullptr;

    return DataPtr(
        opaque_handle,
        holder,
        &delete_vulkan_allocation,
        Device(DeviceType::Vulkan, static_cast<int8_t>(index)));
  }

 private:
  // Allocates on the context for the requested device.  Multi-adapter
  // support tracks the runtime's adapter list; the first adapter backs
  // device 0, matching the runtime's default adapter selection.
  vulkan::api::vTensorStorage* allocate_storage(size_t nbytes, int64_t index) const {
    // Multi-device: the context singleton serves the default adapter.
    // Additional adapters can be addressed through the runtime once the
    // per-device context registry lands.
    (void)index;
    vulkan::api::Context* context = vulkan::api::context();
    TP_CHECK(context, "Vulkan context is not available");

    // Storage-level allocation is dtype-opaque: one byte per payload byte.
    // Ops that construct tensors always create the storage through vTensor
    // with the real dtype; this raw path only backs the Storage ctor.
    return new vulkan::api::vTensorStorage(
        context,
        vulkan::api::StorageType::BUFFER,
        vulkan::api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        std::vector<int64_t>{static_cast<int64_t>(nbytes)},
        DType::UInt8);
  }
};

} // namespace

Allocator* getVulkanAllocator() {
  static VulkanCachingAllocator allocator;
  return &allocator;
}

} // namespace tensorplay

#endif /* USE_VULKAN */
