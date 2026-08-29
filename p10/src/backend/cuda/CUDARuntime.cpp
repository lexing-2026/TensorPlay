#include "CUDARuntime.h"

#ifdef USE_CUDA

#include "CUDAContext.h"
#include "Exception.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

constexpr size_t kStreamsPerPool = 32;
constexpr unsigned int kStreamFlags = cudaStreamNonBlocking;
// Upper bound for the flat per-thread current-stream table; devices beyond it
// (exotic multi-GPU hosts) fall back to the map below.
constexpr size_t kMaxFlatDevices = 64;

// Per-thread cached current device (see currentDevice) - cold until the
// first real query on this thread.
thread_local int t_cached_device = -1;

// Device count never changes within a process; caching it turns every
// getCurrentCUDAStream()/guard construction into pure arithmetic instead of a
// cudaGetDeviceCount round trip.
std::atomic<int>& cachedDeviceCount() {
    // Deliberately leaked: queried during interpreter teardown.
    static auto* count = new std::atomic<int>(-1);
    return *count;
}

int deviceCountCached() {
    int count = cachedDeviceCount().load(std::memory_order_relaxed);
    if (count < 0) {
        checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        cachedDeviceCount().store(count, std::memory_order_relaxed);
    }
    return count;
}

int normalizeDevice(int device_index) {
    const int count = deviceCountCached();
    if (device_index < 0) {
        device_index = currentDevice();
    }
    if (device_index < 0 || device_index >= count) {
        TP_THROW(ValueError,
                 "CUDA device index " + std::to_string(device_index) +
                 " is out of range for " + std::to_string(count) + " visible device(s)");
    }
    return device_index;
}

struct StreamPool {
    std::mutex mutex;
    bool low_initialized = false;
    bool high_initialized = false;
    std::array<cudaStream_t, kStreamsPerPool> low{};
    std::array<cudaStream_t, kStreamsPerPool> high{};
    std::atomic<uint32_t> low_index{0};
    std::atomic<uint32_t> high_index{0};
};

std::mutex& poolTableMutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

std::vector<StreamPool*>& poolTable() {
    // CUDA streams are deliberately leaked. Destroying process-global streams
    // stream pools follow the same lifetime rule.
    static auto* pools = new std::vector<StreamPool*>();
    return *pools;
}

StreamPool& poolFor(int device_index) {
    std::lock_guard<std::mutex> lock(poolTableMutex());
    auto& pools = poolTable();
    if (pools.size() <= static_cast<size_t>(device_index)) {
        pools.resize(static_cast<size_t>(device_index) + 1, nullptr);
    }
    if (!pools[device_index]) pools[device_index] = new StreamPool();
    return *pools[device_index];
}

int clampPriority(int requested) {
    int least = 0;
    int greatest = 0;
    checkCuda(cudaDeviceGetStreamPriorityRange(&least, &greatest),
              "cudaDeviceGetStreamPriorityRange");
    return std::max(greatest, std::min(least, requested));
}

void initializePool(StreamPool& pool, int device_index, bool high_priority) {
    std::lock_guard<std::mutex> lock(pool.mutex);
    bool& initialized = high_priority ? pool.high_initialized : pool.low_initialized;
    if (initialized) return;

    CUDAGuard guard(device_index);
    auto& streams = high_priority ? pool.high : pool.low;
    const int priority = clampPriority(high_priority ? -1 : 0);
    for (auto& stream : streams) {
        checkCuda(cudaStreamCreateWithPriority(&stream, kStreamFlags, priority),
                  "cudaStreamCreateWithPriority");
    }
    initialized = true;
}

// Per-thread current stream: flat array indexed by device for O(1) access on
// the kernel-launch hot path (the previous unordered_map paid a hash lookup
// per launch); oversized device indices spill into the map.
thread_local std::array<cudaStream_t, kMaxFlatDevices> current_streams_flat{};
thread_local std::unordered_map<int, cudaStream_t> current_streams_overflow;

inline void setCurrentStreamRaw(int device_index, cudaStream_t stream) {
    if (static_cast<size_t>(device_index) < kMaxFlatDevices) {
        current_streams_flat[static_cast<size_t>(device_index)] = stream;
    } else {
        current_streams_overflow[device_index] = stream;
    }
}

inline cudaStream_t getCurrentStreamRaw(int device_index) {
    if (static_cast<size_t>(device_index) < kMaxFlatDevices) {
        return current_streams_flat[static_cast<size_t>(device_index)];
    }
    auto it = current_streams_overflow.find(device_index);
    return it == current_streams_overflow.end() ? nullptr : it->second;
}

} // namespace

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string(operation) + " failed: " + cudaGetErrorString(error) +
                 " (CUDA error " + std::to_string(static_cast<int>(error)) + ")");
    }
    noteCudaRuntimeCall();
}

int currentDevice() {
    // Thread-local cache: cudaGetDevice is per-thread state, so every set
    // point in this translation unit refreshes the cache and the hot read
    // path (kernel launches, stream queries, guards) costs one TLS load.
    // All cudaSetDevice calls in the codebase are funnelled through
    // setDeviceCached/CUDAGuard here; external .cu kernels only ever *read*
    // the device, which cannot invalidate the cache.
    int device = t_cached_device;
    if (device < 0) {
        checkCuda(cudaGetDevice(&device), "cudaGetDevice");
        t_cached_device = device;
    }
    return device;
}

void setDeviceCached(int device_index) {
    if (currentDevice() == device_index) return;
    checkCuda(cudaSetDevice(device_index), "cudaSetDevice");
    t_cached_device = device_index;
}

int deviceCount() {
    return deviceCountCached();
}

CUDAGuard::CUDAGuard(int device_index) {
    original_device_ = currentDevice();
    device_index = normalizeDevice(device_index);
    if (device_index != original_device_) {
        setDeviceCached(device_index);
        changed_ = true;
    }
}

CUDAGuard::CUDAGuard(const Device& device)
    : CUDAGuard(device.is_cuda() ? static_cast<int>(device.index()) : -1) {
    if (!device.is_cuda()) {
        TP_THROW(ValueError, "CUDAGuard requires a CUDA device");
    }
}

CUDAGuard::~CUDAGuard() noexcept {
    if (changed_) {
        // Restore through the raw call: destructors must not throw, and the
        // cache tracks exactly this value either way.
        (void)cudaSetDevice(original_device_);
        t_cached_device = original_device_;
    }
}

int CUDAStream::priority() const {
    // Stream handles carry their own context; querying the priority needs no
    // device switch (matches c10::cuda::CUDAStream).
    int value = 0;
    checkCuda(cudaStreamGetPriority(stream_, &value), "cudaStreamGetPriority");
    return value;
}

bool CUDAStream::query() const {
    cudaError_t error = cudaStreamQuery(stream_);
    if (error == cudaSuccess) return true;
    if (error == cudaErrorNotReady) {
        (void)cudaGetLastError();
        return false;
    }
    checkCuda(error, "cudaStreamQuery");
    return false;
}

void CUDAStream::synchronize() const {
    checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

CUDAStream getStreamFromPool(int priority, int device_index) {
    device_index = normalizeDevice(device_index);
    StreamPool& pool = poolFor(device_index);
    const bool high_priority = priority < 0;
    initializePool(pool, device_index, high_priority);
    auto& counter = high_priority ? pool.high_index : pool.low_index;
    auto& streams = high_priority ? pool.high : pool.low;
    const size_t index = counter.fetch_add(1, std::memory_order_relaxed) % kStreamsPerPool;
    return CUDAStream(device_index, streams[index]);
}

CUDAStream getDefaultCUDAStream(int device_index) {
    return CUDAStream(normalizeDevice(device_index), nullptr);
}

CUDAStream getCurrentCUDAStream(int device_index) {
    device_index = normalizeDevice(device_index);
    return CUDAStream(device_index, getCurrentStreamRaw(device_index));
}

void setCurrentCUDAStream(const CUDAStream& stream) {
    setCurrentStreamRaw(stream.device_index_, stream.stream_);
}

CUDAStreamGuard::CUDAStreamGuard(const CUDAStream& stream) {
    original_device_ = currentDevice();
    stream_device_ = stream.device_index();
    original_stream_ = getCurrentCUDAStream(stream_device_).stream();
    if (original_device_ != stream_device_) setDeviceCached(stream_device_);
    setCurrentCUDAStream(stream);
    active_ = true;
}

CUDAStreamGuard::~CUDAStreamGuard() noexcept {
    if (!active_) return;
    setCurrentStreamRaw(stream_device_, original_stream_);
    if (original_device_ != stream_device_) {
        (void)cudaSetDevice(original_device_);
        t_cached_device = original_device_;
    }
}

OptionalCUDAGuard::OptionalCUDAGuard(const Device& device) {
    if (device.is_cuda()) guard_ = std::make_unique<CUDAGuard>(device);
}

OptionalCUDAGuard::~OptionalCUDAGuard() = default;

struct CUDAEvent::State {
    explicit State(bool timing, bool blocking, bool ipc)
        : enable_timing(timing), blocking_sync(blocking), interprocess(ipc) {}

    ~State() {
        if (!created) return;
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device_index && cudaSetDevice(device_index) != cudaSuccess) return;
        (void)cudaEventDestroy(event);
        if (previous != device_index) (void)cudaSetDevice(previous);
    }

    mutable std::mutex mutex;
    cudaEvent_t event = nullptr;
    int device_index = -1;
    bool enable_timing = false;
    bool blocking_sync = false;
    bool interprocess = false;
    bool created = false;
    bool recorded = false;
};

CUDAEvent::CUDAEvent(bool enable_timing, bool blocking, bool interprocess)
    : state_(std::make_shared<State>(enable_timing, blocking, interprocess)) {
    if (interprocess && enable_timing) {
        TP_THROW(ValueError, "CUDA interprocess events do not support timing");
    }
}

CUDAEvent::~CUDAEvent() = default;
CUDAEvent::CUDAEvent(const CUDAEvent&) = default;
CUDAEvent::CUDAEvent(CUDAEvent&&) noexcept = default;
CUDAEvent& CUDAEvent::operator=(const CUDAEvent&) = default;
CUDAEvent& CUDAEvent::operator=(CUDAEvent&&) noexcept = default;

void CUDAEvent::record() {
    record(getCurrentCUDAStream());
}

void CUDAEvent::record(const CUDAStream& stream) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->created) {
        unsigned int flags = state_->enable_timing ? cudaEventDefault : cudaEventDisableTiming;
        if (state_->blocking_sync) flags |= cudaEventBlockingSync;
        if (state_->interprocess) flags |= cudaEventInterprocess | cudaEventDisableTiming;
        CUDAGuard guard(stream.device_index());
        checkCuda(cudaEventCreateWithFlags(&state_->event, flags), "cudaEventCreateWithFlags");
        state_->device_index = stream.device_index();
        state_->created = true;
    } else if (state_->device_index != stream.device_index()) {
        TP_THROW(DeviceMismatchError,
                 "CUDA event was created on cuda:" + std::to_string(state_->device_index) +
                 " and cannot be recorded on cuda:" + std::to_string(stream.device_index()));
    }
    CUDAGuard guard(stream.device_index());
    checkCuda(cudaEventRecord(state_->event, stream.stream()), "cudaEventRecord");
    state_->recorded = true;
}

void CUDAEvent::block(const CUDAStream& stream) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->created || !state_->recorded) return;
    CUDAGuard guard(stream.device_index());
    checkCuda(cudaStreamWaitEvent(stream.stream(), state_->event, 0), "cudaStreamWaitEvent");
}

bool CUDAEvent::query() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->created || !state_->recorded) return true;
    CUDAGuard guard(state_->device_index);
    cudaError_t error = cudaEventQuery(state_->event);
    if (error == cudaSuccess) return true;
    if (error == cudaErrorNotReady) {
        (void)cudaGetLastError();
        return false;
    }
    checkCuda(error, "cudaEventQuery");
    return false;
}

void CUDAEvent::synchronize() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->created || !state_->recorded) return;
    CUDAGuard guard(state_->device_index);
    checkCuda(cudaEventSynchronize(state_->event), "cudaEventSynchronize");
}

float CUDAEvent::elapsed_time(const CUDAEvent& end_event) const {
    if (state_.get() == end_event.state_.get()) return 0.0f;
    std::scoped_lock lock(state_->mutex, end_event.state_->mutex);
    if (!state_->created || !state_->recorded ||
        !end_event.state_->created || !end_event.state_->recorded) {
        TP_THROW(RuntimeError, "Both CUDA events must be recorded before elapsed_time");
    }
    if (!state_->enable_timing || !end_event.state_->enable_timing) {
        TP_THROW(RuntimeError, "elapsed_time requires events created with enable_timing=True");
    }
    if (state_->device_index != end_event.state_->device_index) {
        TP_THROW(DeviceMismatchError, "Cannot measure elapsed time between events on different devices");
    }
    CUDAGuard guard(state_->device_index);
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, state_->event, end_event.state_->event),
              "cudaEventElapsedTime");
    return milliseconds;
}

int CUDAEvent::device_index() const noexcept {
    return state_->device_index;
}

bool CUDAEvent::is_created() const noexcept {
    return state_->created;
}

uintptr_t CUDAEvent::id() const noexcept {
    return reinterpret_cast<uintptr_t>(state_->event);
}

void recordStream(void* base_ptr, const Device& device) {
    if (!base_ptr || !device.is_cuda()) return;
    recordStream(base_ptr, getCurrentCUDAStream(static_cast<int>(device.index())));
}

} // namespace cuda
} // namespace tensorplay

#endif // USE_CUDA
