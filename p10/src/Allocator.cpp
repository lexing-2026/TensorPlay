#include "Allocator.h"
#include "Exception.h"
#include "Profiler.h"
#include <memory>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#include "CUDARuntime.h"
#include <cuda_runtime.h>
#endif

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

        // Allocator-level memory capture (profile_memory sessions).  The
        // requested size is not recoverable after 64-byte bucket rounding,
        // so frees report the block's rounded size -- consistent with the
        // alloc side, which also records the rounded size.
        prof::mem_record_free(ptr, static_cast<int64_t>(size - HEADER_SIZE),
                              /*cuda=*/false, /*device=*/-1, /*stream=*/-1);
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

        // Allocator-level memory capture (profile_memory sessions); reports
        // the bucket-rounded block so alloc/free bytes stay consistent.
        prof::mem_record_alloc(data_ptr,
                               static_cast<int64_t>(total_size - HEADER_SIZE),
                               /*cuda=*/false, /*device=*/-1, /*stream=*/-1);

        return DataPtr(data_ptr, deleter, Device(DeviceType::CPU));
    }
};

} // namespace

Allocator* getCPUAllocator() {
    return CachingAllocator::instance();
}

void copyAllocationBytes(void* destination, const Device& destination_device,
                         const void* source, const Device& source_device,
                         size_t nbytes) {
    if (!destination || !source || nbytes == 0) return;
    if (destination_device.is_cpu() && source_device.is_cpu()) {
        std::memcpy(destination, source, nbytes);
        return;
    }
#ifdef USE_CUDA
    const Device cuda_device = destination_device.is_cuda()
        ? destination_device
        : source_device;
    cuda::CUDAGuard guard(cuda_device);
    auto stream = cuda::getCurrentCUDAStream(static_cast<int>(cuda_device.index()));
    cudaMemcpyKind kind = cudaMemcpyDefault;
    if (destination_device.is_cuda() && source_device.is_cuda()) {
        kind = cudaMemcpyDeviceToDevice;
    } else if (destination_device.is_cuda()) {
        kind = cudaMemcpyHostToDevice;
    } else {
        kind = cudaMemcpyDeviceToHost;
    }
    cuda::checkCuda(cudaMemcpyAsync(destination, source, nbytes, kind, stream.stream()),
                    "cudaMemcpyAsync (storage resize)");
    if (destination_device.is_cuda()) cuda::recordStream(destination, stream);
    if (source_device.is_cuda()) cuda::recordStream(const_cast<void*>(source), stream);
    if (destination_device.is_cpu() || source_device.is_cpu()) stream.synchronize();
    return;
#else
    TP_THROW(RuntimeError, "cannot copy CUDA storage in a CPU-only TensorPlay build");
#endif
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
