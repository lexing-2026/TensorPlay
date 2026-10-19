#include "backend/mps/MPSAllocator.h"

#include "Exception.h"
#include "backend/mps/DiagTimer.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace mps {
namespace {

// One cached Metal buffer. A block is created per allocation and, once the
// owning tensor is freed, is parked in the allocator's free list so a later
// same-sized request can reuse the Metal resource instead of round-tripping
// through [MTLDevice newBufferWithLength:].
struct BufferBlock {
  id<MTLBuffer> buffer = nil;
  // Host-visible mapping of the buffer (unified memory). This is the pointer
  // handed to tensor storage as data_ptr().
  void* cpu_ptr = nullptr;
  // Backing allocation size (post-alignment).
  size_t size = 0;
  // Requested size (pre-alignment).
  size_t requested_size = 0;
  bool in_use = false;
  uint64_t buf_id = 0;

  static uint64_t buffer_counter;

  BufferBlock(size_t Size, size_t RequestedSize)
      : size(Size), requested_size(RequestedSize) {
    buf_id = ++buffer_counter;
  }

  static bool Comparator(const BufferBlock* a, const BufferBlock* b) {
    if (a->size != b->size) {
      return a->size < b->size;
    }
    return a->buf_id < b->buf_id;
  }
};

uint64_t BufferBlock::buffer_counter = 0;

// Release function installed on tensor DataPtrs. The context is the owning
// BufferBlock; freeing returns the block to the pool for reuse.
void releaseMPSBlock(void* ctx) noexcept {
  if (!ctx) {
    return;
  }
  auto* block = static_cast<BufferBlock*>(ctx);
  getIMPSAllocator()->freeBuffer(block->cpu_ptr);
}

class MPSAllocatorImpl final : public IMPSAllocator {
 public:
  static MPSAllocatorImpl* instance() {
    static auto* inst = new MPSAllocatorImpl();
    return inst;
  }

  MPSAllocatorImpl()
      : m_device(MPSDevice::getInstance()->device()),
        m_alignment(
            [m_device heapBufferSizeAndAlignWithLength:1
                                              options:MTLResourceStorageModeShared]
                .align),
        m_low_watermark_limit(static_cast<size_t>(
            kLowWatermarkRatio *
            static_cast<double>([m_device recommendedMaxWorkingSetSize]))) {}

  ~MPSAllocatorImpl() override {
    emptyCache();
  }

  DataPtr allocate(size_t nbytes) const override {
    return allocate(nbytes, Device(DeviceType::MPS, 0));
  }

  DataPtr allocate(size_t nbytes, const Device& requested_device) const override {
    if (!requested_device.is_mps()) {
      TP_THROW(ValueError, "MPSAllocator received a non-MPS device");
    }
    if (nbytes == 0) {
      return DataPtr(nullptr, nullptr, Device(DeviceType::MPS, 0));
    }
    BufferBlock* block = instance()->malloc(nbytes);
    return DataPtr(block->cpu_ptr, block, &releaseMPSBlock,
                   Device(DeviceType::MPS, 0));
  }

  // Allocates (or reuses) a shared-storage MTLBuffer of at least `size`
  // bytes and returns its BufferBlock. The block stays owned by the pool
  // until freeBuffer() is called.
  BufferBlock* malloc(size_t size) {
    const unsigned long long diag_t0 = diagNowNs();
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t aligned = alignUp(size);
    BufferBlock* block = findReusableBlock(aligned);
    const bool reused = block != nullptr;
    if (!block) {
      block = createBlock(aligned, size);
    }
    block->requested_size = size;
    block->in_use = true;
    m_current_allocated_memory += block->size;
    m_ptr_to_block[block->cpu_ptr] = block;
    {
      static thread_local DiagTimer timer{"alloc malloc"};
      static thread_local unsigned long long fresh_calls = 0;
      static thread_local unsigned long long fresh_count = 0;
      timer.record(diagNowNs() - diag_t0);
      ++fresh_calls;
      if (!reused) {
        ++fresh_count;
      }
      if (fresh_calls % 512 == 0) {
        std::fprintf(stderr, "[mps-t] alloc fresh n=%llu fresh=%llu\n",
                     fresh_calls, fresh_count);
      }
    }
    return block;
  }

  // Returns a previously allocated block to the pool. The Metal buffer is
  // kept alive for reuse by a later allocation.
  void freeBuffer(void* ptr) override {
    if (!ptr) {
      return;
    }
    const unsigned long long diag_t0 = diagNowNs();
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_ptr_to_block.find(ptr);
    if (it == m_ptr_to_block.end()) {
      // Foreign pointer (e.g. a host pointer passed into the MPS device by
      // external code); nothing to recycle.
      return;
    }
    BufferBlock* block = it->second;
    if (!block->in_use) {
      return;
    }
    block->in_use = false;
    m_current_allocated_memory -= block->size;
    m_available_buffers.insert(block);
    static thread_local DiagTimer timer{"alloc free"};
    timer.record(diagNowNs() - diag_t0);
  }

  void emptyCache() override {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto* block : m_available_buffers) {
      m_ptr_to_block.erase(block->cpu_ptr);
      [block->buffer release];
      block->buffer = nil;
      block->cpu_ptr = nullptr;
      m_total_allocated_memory -= block->size;
      delete block;
    }
    m_available_buffers.clear();
  }

  void freeInactiveBuffers() const override {
    // With the free-list design every freed block is already inactive; there
    // is nothing pending on in-flight command buffers to sweep.
  }

  ssize_t getUnalignedBufferSize(const void* ptr) const override {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_ptr_to_block.find(ptr);
    return it == m_ptr_to_block.end()
        ? -1
        : static_cast<ssize_t>(it->second->requested_size);
  }

  uint64_t getBufferId(const void* ptr) const override {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_ptr_to_block.find(ptr);
    return it == m_ptr_to_block.end() ? 0 : it->second->buf_id;
  }

  bool isSharedBuffer(const void* ptr) const override {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_ptr_to_block.find(ptr);
    if (it == m_ptr_to_block.end()) {
      return false;
    }
    return [it->second->buffer storageMode] == MTLStorageModeShared;
  }

  size_t getTotalAllocatedMemory() const override {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_total_allocated_memory;
  }

  size_t getCurrentAllocatedMemory() const override {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_current_allocated_memory;
  }

  size_t getDriverAllocatedMemory() const override {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_total_allocated_memory;
  }

  size_t getRecommendedMaxMemory() const override {
    return [m_device recommendedMaxWorkingSetSize];
  }

  // How far, in megabytes, the live allocations still sit below the low
  // watermark. Zero is the limit itself: the device is under memory pressure
  // and work queued in the current command buffer should be committed.
  ssize_t getLowWatermarkValue() const override {
    const ssize_t headroom = static_cast<ssize_t>(m_low_watermark_limit) -
                             static_cast<ssize_t>(getCurrentAllocatedMemory());
    return std::max<ssize_t>(0, headroom / static_cast<ssize_t>(MB(1)));
  }

  // Returns the owning MTLBuffer for a host-visible pointer, or nil.
  id<MTLBuffer> mtlBufferForPointer(const void* ptr) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_ptr_to_block.find(ptr);
    return it == m_ptr_to_block.end() ? nil : it->second->buffer;
  }

  void* getMTLBuffer(const void* ptr) const override {
    return (void*)(mtlBufferForPointer(ptr));
  }

 private:
  static constexpr size_t kSmallAllocation = MB(1);
  static constexpr size_t kSmallRound = 512;
  // The low watermark sits this multiple above the device's recommended maximum
  // working set, so that a run only reads as pressurised once it has outgrown
  // the device by a wide margin.
  static constexpr double kLowWatermarkRatio = 1.4;

  size_t alignUp(size_t nbytes) const {
    if (nbytes == 0) {
      return 0;
    }
    const size_t quantum = std::max(
        m_alignment,
        nbytes <= kSmallAllocation ? kSmallRound : kSmallRound * 4096);
    return ((nbytes + quantum - 1) / quantum) * quantum;
  }

  BufferBlock* findReusableBlock(size_t aligned) {
    // Best-fit: the smallest cached block large enough to serve the request.
    // The key's id is zero so that a cached block of exactly this size compares
    // equal or greater; the set's id tiebreaker would otherwise order every
    // same-sized block before the key and hide it from the search.
    BufferBlock key(aligned, aligned);
    key.buf_id = 0;
    auto it = m_available_buffers.lower_bound(&key);
    if (it != m_available_buffers.end()) {
      BufferBlock* block = *it;
      m_available_buffers.erase(it);
      return block;
    }
    return nullptr;
  }

  BufferBlock* createBlock(size_t aligned, size_t requested) {
    id<MTLBuffer> buffer =
        [m_device newBufferWithLength:aligned
                              options:MTLResourceStorageModeShared];
    if (!buffer) {
      TP_THROW(RuntimeError,
               "MPSAllocator failed to allocate ", aligned, " bytes");
    }
    auto* block = new BufferBlock(aligned, requested);
    block->buffer = buffer;
    block->cpu_ptr = [buffer contents];
    m_total_allocated_memory += aligned;
    return block;
  }

  MTLDevice_t m_device;
  const size_t m_alignment;
  const size_t m_low_watermark_limit;
  mutable std::mutex m_mutex;
  std::set<BufferBlock*, decltype(&BufferBlock::Comparator)> m_available_buffers{
      &BufferBlock::Comparator};
  std::unordered_map<const void*, BufferBlock*> m_ptr_to_block;
  size_t m_total_allocated_memory = 0;
  size_t m_current_allocated_memory = 0;
};

} // namespace

IMPSAllocator* getIMPSAllocator() {
  return MPSAllocatorImpl::instance();
}

void* getMTLBufferFromPointer(const void* ptr) {
  return getIMPSAllocator()->getMTLBuffer(ptr);
}

} // namespace mps

Allocator* getMPSAllocator() {
  return mps::getIMPSAllocator();
}

} // namespace tensorplay