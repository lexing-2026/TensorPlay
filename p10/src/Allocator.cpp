#include "Allocator.h"
#include "Exception.h"
#include <memory>
#include <mutex>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace tensorplay {

namespace {

// Portable aligned allocation
void* alloc_aligned(size_t nbytes, size_t alignment = 64) {
    if (nbytes == 0) return nullptr;
#ifdef _WIN32
    return _aligned_malloc(nbytes, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, nbytes) != 0) return nullptr;
    return ptr;
#endif
}

void free_aligned(void* data) {
    if (!data) return;
#ifdef _WIN32
    _aligned_free(data);
#else
    free(data);
#endif
}

// Caching Allocator
class CachingAllocator : public Allocator {
    // Header structure
    struct Header {
        size_t size; // Total allocated size (including header)
    };
    static constexpr size_t HEADER_SIZE = 64; // Keep 64-byte alignment

    mutable std::mutex mutex_;
    mutable std::unordered_map<size_t, std::vector<void*>> free_blocks_;

public:
    static CachingAllocator* instance() {
        static CachingAllocator* inst = new CachingAllocator();
        return inst;
    }

    static void deleter(void* ptr) {
        if (!ptr) return;
        // Pointer points to data, header is before it
        char* raw_ptr = static_cast<char*>(ptr) - HEADER_SIZE;
        Header* header = reinterpret_cast<Header*>(raw_ptr);
        size_t size = header->size;
        
        instance()->free(raw_ptr, size);
    }

    void free(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_blocks_[size].push_back(ptr);
    }

    DataPtr allocate(size_t nbytes) const override {
        // Calculate total size: nbytes + header, aligned to 64 bytes
        // We want the *data* to be aligned to 64.
        // If we allocate X, and return X+64, X+64 is aligned if X is aligned to 64.
        
        size_t total_size = nbytes + HEADER_SIZE;
        // Normalize size to reduce fragmentation (buckets of 64 bytes)
        total_size = (total_size + 63) & ~63;

        void* ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = free_blocks_.find(total_size);
            if (it != free_blocks_.end() && !it->second.empty()) {
                ptr = it->second.back();
                it->second.pop_back();
            }
        }

        if (!ptr) {
            ptr = alloc_aligned(total_size);
        }
        
        if (!ptr) TP_THROW(RuntimeError, "Out of memory");

        // Setup header
        Header* header = reinterpret_cast<Header*>(ptr);
        header->size = total_size;

        // Return data pointer
        void* data_ptr = static_cast<char*>(ptr) + HEADER_SIZE;
        
        return DataPtr(data_ptr, deleter, Device(DeviceType::CPU));
    }
};

} // namespace

Allocator* getCPUAllocator() {
    return CachingAllocator::instance();
}

#ifdef USE_CUDA
Allocator* getCUDAAllocator();
#endif

Allocator* getAllocator(DeviceType t) {
    if (t == DeviceType::CPU) {
        return getCPUAllocator();
    }
#ifdef USE_CUDA
    if (t == DeviceType::CUDA) {
        return getCUDAAllocator();
    }
#endif
    
    TP_THROW(NotImplementedError, "Allocator not implemented for this device type");
}

} // namespace tensorplay
