#pragma once

#include <cstdint>
#include <memory>

#include "Device.h"
#include "Macros.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace tensorplay {
namespace cuda {

#ifdef USE_CUDA

// Throws a TensorPlay RuntimeError with the original CUDA error text.
P10_API void checkCuda(cudaError_t error, const char* operation);
P10_API int currentDevice();
P10_API int deviceCount();

class P10_API CUDAGuard {
public:
    explicit CUDAGuard(int device_index);
    explicit CUDAGuard(const Device& device);
    ~CUDAGuard() noexcept;

    CUDAGuard(const CUDAGuard&) = delete;
    CUDAGuard& operator=(const CUDAGuard&) = delete;

private:
    int original_device_ = -1;
    bool changed_ = false;
};

// A value object around a CUDA stream. Non-default streams come from a
// per-device pool and intentionally outlive user wrappers, matching PyTorch's
// cheap, reusable Stream semantics.
class P10_API CUDAStream {
public:
    int device_index() const noexcept { return device_index_; }
    Device device() const { return Device(DeviceType::CUDA, device_index_); }
    cudaStream_t stream() const noexcept { return stream_; }
    uintptr_t id() const noexcept { return reinterpret_cast<uintptr_t>(stream_); }
    int priority() const;
    bool query() const;
    void synchronize() const;

    bool operator==(const CUDAStream& other) const noexcept {
        return device_index_ == other.device_index_ && stream_ == other.stream_;
    }
    bool operator!=(const CUDAStream& other) const noexcept { return !(*this == other); }

    // Unbound placeholder ("no stream yet"), mirroring c10's default-constructed
    // Stream. Needed by components like CUDAGraph that stash a stream slot
    // before capture begins; not a usable launch stream.
    static CUDAStream undefined() noexcept { return CUDAStream(-1, nullptr); }

    // Adopts a raw driver stream (e.g. a conditional-node child stream) as a
    // value object without taking ownership of its lifetime.
    static CUDAStream fromExternal(cudaStream_t stream,
                                   int device_index) noexcept {
        return CUDAStream(device_index, stream);
    }

private:
    CUDAStream(int device_index, cudaStream_t stream) noexcept
        : device_index_(device_index), stream_(stream) {}

    int device_index_;
    cudaStream_t stream_;

    friend CUDAStream getStreamFromPool(int, int);
    friend CUDAStream getDefaultCUDAStream(int);
    friend CUDAStream getCurrentCUDAStream(int);
    friend void setCurrentCUDAStream(const CUDAStream&);
};

P10_API CUDAStream getStreamFromPool(int priority = 0, int device_index = -1);
P10_API CUDAStream getDefaultCUDAStream(int device_index = -1);
P10_API CUDAStream getCurrentCUDAStream(int device_index = -1);
P10_API void setCurrentCUDAStream(const CUDAStream& stream);
P10_API void sleep(uint64_t cycles);

class P10_API CUDAStreamGuard {
public:
    explicit CUDAStreamGuard(const CUDAStream& stream);
    ~CUDAStreamGuard() noexcept;

    CUDAStreamGuard(const CUDAStreamGuard&) = delete;
    CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;

private:
    int original_device_ = -1;
    int stream_device_ = -1;
    cudaStream_t original_stream_ = nullptr;
    bool active_ = false;
};

// Device guard that is a no-op for CPU devices. Generated dispatch wrappers
// use this so tensors on cuda:N execute correctly even when another device is
// current in the calling thread.
class P10_API OptionalCUDAGuard {
public:
    explicit OptionalCUDAGuard(const Device& device);
    ~OptionalCUDAGuard();

    OptionalCUDAGuard(const OptionalCUDAGuard&) = delete;
    OptionalCUDAGuard& operator=(const OptionalCUDAGuard&) = delete;

private:
    std::unique_ptr<CUDAGuard> guard_;
};

class P10_API CUDAEvent {
public:
    CUDAEvent(bool enable_timing = false, bool blocking = false, bool interprocess = false);
    ~CUDAEvent();
    CUDAEvent(const CUDAEvent&);
    CUDAEvent(CUDAEvent&&) noexcept;
    CUDAEvent& operator=(const CUDAEvent&);
    CUDAEvent& operator=(CUDAEvent&&) noexcept;

    void record();
    void record(const CUDAStream& stream);
    void block(const CUDAStream& stream) const;
    bool query() const;
    void synchronize() const;
    float elapsed_time(const CUDAEvent& end_event) const;
    int device_index() const noexcept;
    bool is_created() const noexcept;
    uintptr_t id() const noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
};

// Called by TensorImpl whenever CUDA storage participates in an operation.
// The caching allocator delays reuse until work on every recorded stream has
// completed. Unknown pointers (for externally-owned storage) are ignored.
P10_API void recordStream(void* base_ptr, const CUDAStream& stream);
P10_API void recordStream(void* base_ptr, const Device& device);

// Pins the lifetime of a cudaHostAlloc block to work enqueued on `stream`.
// Unknown/pageable pointers are ignored.
P10_API void recordPinnedStream(void* base_ptr, const CUDAStream& stream);

#else

class OptionalCUDAGuard {
public:
    explicit OptionalCUDAGuard(const Device&) {}
};

#endif // USE_CUDA

} // namespace cuda
} // namespace tensorplay
