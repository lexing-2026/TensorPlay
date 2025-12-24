#include "Allocator.h"
#include "Exception.h"
#include "Device.h" // For memory tracking declarations
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <atomic>

namespace tensorplay {
namespace cuda {

// Global memory statistics
static std::atomic<size_t> g_memory_allocated(0);
static std::atomic<size_t> g_max_memory_allocated(0);
static std::mutex g_memory_map_mutex;
static std::unordered_map<void*, size_t> g_memory_map;

size_t memory_allocated(int device) {
    // Currently tracking global memory, not per device
    return g_memory_allocated.load();
}

size_t max_memory_allocated(int device) {
    return g_max_memory_allocated.load();
}

void reset_max_memory_allocated(int device) {
    g_max_memory_allocated.store(g_memory_allocated.load());
}

void empty_cache() {
    // Simple allocator doesn't cache, so this is a no-op or just cudaDeviceSynchronize
    cudaDeviceSynchronize();
}

} // namespace cuda

namespace {

void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::string msg = "CUDA Error: " + std::string(cudaGetErrorString(result));
        TP_THROW(RuntimeError, msg);
    }
}

class CUDAAllocator : public Allocator {
public:
    static void deleter(void* ptr) {
        if (!ptr) return;
        
        // Update stats
        {
            std::lock_guard<std::mutex> lock(cuda::g_memory_map_mutex);
            auto it = cuda::g_memory_map.find(ptr);
            if (it != cuda::g_memory_map.end()) {
                cuda::g_memory_allocated -= it->second;
                cuda::g_memory_map.erase(it);
            }
        }

        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            // Don't throw in destructor
            std::cerr << "CUDA Error in deleter: " << cudaGetErrorString(err) << std::endl;
        }
    }

    DataPtr allocate(size_t nbytes) const override {
        void* ptr = nullptr;
        checkCudaErrors(cudaMalloc(&ptr, nbytes));
        
        // Update stats
        {
            std::lock_guard<std::mutex> lock(cuda::g_memory_map_mutex);
            cuda::g_memory_map[ptr] = nbytes;
            cuda::g_memory_allocated += nbytes;
            
            size_t current = cuda::g_memory_allocated.load();
            size_t max = cuda::g_max_memory_allocated.load();
            if (current > max) {
                cuda::g_max_memory_allocated.store(current);
            }
        }

        // Device(DeviceType::CUDA, 0) - hardcoded device 0 for now
        return DataPtr(ptr, deleter, Device(DeviceType::CUDA, 0)); 
    }
};

} // namespace


Allocator* getCUDAAllocator() {
    static CUDAAllocator allocator;
    return &allocator;
}

} // namespace tensorplay
