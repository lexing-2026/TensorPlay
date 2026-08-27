// GPU-timeline support for the native profiler (USE_CUDA builds only; this
// TU compiles to nothing otherwise).  A pool-backed cudaEvent pair is armed
// around dispatched CUDA work from the generated redispatch funnels
// (tools/codegen/gen_api.py emits the arm/close calls under USE_CUDA when
// profiler gpu-timing is requested).
//
// Elapsed times are NOT read per-op (that would force a stream sync in the
// hot path).  At stop, the binding layer waits on only the last end event of
// each recorded stream, resolves all pairs, and recycles them.  This avoids a
// device-wide synchronize and keeps the event-pool mutex away from CUDA API
// calls, which is important when several launch threads are unwinding.
//
// Validation status: written against CUDA 12.4 semantics, pending remote
// run on the sm_89 box (.remote_build.md) -- same flow as the RNN/sparse
// batches.

#include "Profiler.h"

#ifdef USE_CUDA

#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <deque>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace prof {

namespace {

std::mutex g_gpu_mutex;
// Recycled cudaEvent_t handles.  Created with default flags (timing on).
std::unordered_map<int, std::deque<cudaEvent_t>>* g_pool = nullptr;
// Live pairs awaiting resolution: (slot_index, start, end).  Flattened at
// stop by the binding layer via gpu_resolve_all().
struct LivePair {
    size_t slot;
    int device;
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t end;
};
std::vector<LivePair>* g_live = nullptr;

cudaEvent_t acquire_event(int device) {
    {
        std::lock_guard<std::mutex> lock(g_gpu_mutex);
        if (g_pool) {
            auto it = g_pool->find(device);
            if (it != g_pool->end() && !it->second.empty()) {
                cudaEvent_t ev = it->second.back();
                it->second.pop_back();
                return ev;
            }
        }
    }

    // The generated redispatcher has already installed the target device
    // guard.  Do not hold g_gpu_mutex while CUDA allocates an event.
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDefault) != cudaSuccess) {
        return nullptr;  // profiling must never make a valid op fail
    }
    return ev;
}

uint64_t resolve_timeout_ms() {
    // A bounded wait turns a broken/hung CUDA stream into a profile with
    // missing GPU fields instead of an unkillable profiler stop.  Five
    // seconds is ample for the measured 0.6B decode cases; set to zero for a
    // query-only stop or raise it for unusually long kernels.
    const char* raw = std::getenv("TP_PROFILER_GPU_TIMEOUT_MS");
    if (!raw || !*raw) return 5000;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0') return 5000;
    return static_cast<uint64_t>(value);
}

bool wait_event(cudaEvent_t event,
                const std::chrono::steady_clock::time_point& deadline) {
    for (;;) {
        const cudaError_t status = cudaEventQuery(event);
        if (status == cudaSuccess) return true;
        if (status != cudaErrorNotReady) return false;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

struct StreamKey {
    int device;
    uintptr_t stream;
    bool operator==(const StreamKey& other) const {
        return device == other.device && stream == other.stream;
    }
};

struct StreamKeyHash {
    size_t operator()(const StreamKey& key) const {
        return std::hash<uintptr_t>{}(
            (static_cast<uintptr_t>(static_cast<unsigned>(key.device)) << 32) ^
            key.stream);
    }
};

bool select_device(int device, int* current) {
    if (*current == device) return true;
    if (cudaSetDevice(device) != cudaSuccess) return false;
    *current = device;
    return true;
}

void destroy_event_pair(const LivePair& pair, int* current) {
    if (!select_device(pair.device, current)) return;
    // CUDA permits destroying an event before its recorded work completes;
    // the runtime releases its backing storage after the queued work retires.
    if (pair.start) (void)cudaEventDestroy(pair.start);
    if (pair.end) (void)cudaEventDestroy(pair.end);
}

} // namespace

// Global (not TLS): backward node applies run on device worker threads whose
// TLS never saw the session start.
TENSORPLAY_API std::atomic<bool> g_gpu_timing{false};

GpuTimerPair::~GpuTimerPair() {
    if (!armed_) return;
    close();
}

void GpuTimerPair::arm(const Device& device) {
    if (!g_gpu_timing.load(std::memory_order_acquire) || !rec_.live_ ||
        !device.is_cuda()) return;
    const auto stream = cuda::getCurrentCUDAStream();
    cudaEvent_t start = acquire_event(stream.device_index());
    if (!start || cudaEventRecord(start, stream.stream()) != cudaSuccess) {
        if (start) (void)cudaEventDestroy(start);
        return;
    }
    gpu_start_ = start;
    gpu_stream_ = stream.stream();
    gpu_device_ = stream.device_index();
    armed_ = true;
}

void GpuTimerPair::close() {
    if (!armed_) return;
    armed_ = false;
    auto* start = static_cast<cudaEvent_t>(gpu_start_);
    auto stream = static_cast<cudaStream_t>(gpu_stream_);
    const int device = gpu_device_;
    gpu_start_ = nullptr;
    gpu_stream_ = nullptr;
    gpu_device_ = -1;
    if (!start) return;
    // A dispatcher can be unwound after the OpRecord has already stopped
    // being live (for example, an exception path).  Do not strand the start
    // event merely because there is no Event slot to attach the pair to.
    if (!rec_.live_) {
        (void)cudaEventDestroy(start);
        return;
    }
    cudaEvent_t end = acquire_event(device);
    if (!end || cudaEventRecord(end, stream) != cudaSuccess) {
        (void)cudaEventDestroy(start);
        if (end) (void)cudaEventDestroy(end);
        return;
    }
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (!g_live) g_live = new std::vector<LivePair>();
    g_live->push_back({rec_.slot_, device, stream, start, end});
}

// Binding-layer resolution API.
TENSORPLAY_API void gpu_resolve_all(
        std::vector<Event>& events,
        const std::function<void(Event&, float ms)>& emit) {
    int original_device = -1;
    (void)cudaGetDevice(&original_device);
    std::vector<LivePair> pairs;
    {
        std::lock_guard<std::mutex> lock(g_gpu_mutex);
        if (!g_live || g_live->empty()) return;
        pairs.swap(*g_live);
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(resolve_timeout_ms());

    // One completion wait per stream is enough: CUDA orders event records on
    // a stream, so the last end event covers all earlier pairs on that stream.
    std::unordered_map<StreamKey, size_t, StreamKeyHash> last_on_stream;
    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto& p = pairs[i];
        last_on_stream[{p.device,
                        reinterpret_cast<uintptr_t>(p.stream)}] = i;
    }

    int current = original_device;
    std::vector<std::pair<int, cudaEvent_t>> reusable;
    reusable.reserve(pairs.size() * 2);
    for (const auto& item : last_on_stream) {
        const LivePair& p = pairs[item.second];
        if (!select_device(p.device, &current)) continue;
        (void)wait_event(p.end, deadline);
    }

    // Resolve outside the pool lock.  This is both faster and eliminates the
    // lock-order deadlock that used to occur when recycling from this loop.
    for (const auto& p : pairs) {
        if (select_device(p.device, &current)) {
            float ms = -1.f;
            const cudaError_t status = cudaEventElapsedTime(&ms, p.start, p.end);
            if (status == cudaSuccess && ms >= 0.f) {
                if (p.slot < events.size()) {
                    events[p.slot].gpu_ms = ms;
                    emit(events[p.slot], ms);
                    events[p.slot].gpu_start = nullptr;
                    events[p.slot].gpu_end = nullptr;
                }
                reusable.emplace_back(p.device, p.start);
                reusable.emplace_back(p.device, p.end);
                continue;
            }
        }
        if (p.slot < events.size()) {
            events[p.slot].gpu_start = nullptr;
            events[p.slot].gpu_end = nullptr;
        }
        // If the deadline expired, cudaEventDestroy is intentionally used
        // instead of returning the still-pending pair to the reusable pool.
        // This bounds memory and makes the next profile independent.
        destroy_event_pair(p, &current);
    }

    // One pool lock for the whole batch, rather than one lock per event.
    if (!reusable.empty()) {
        std::lock_guard<std::mutex> lock(g_gpu_mutex);
        if (!g_pool) g_pool = new std::unordered_map<int, std::deque<cudaEvent_t>>();
        for (const auto& item : reusable)
            (*g_pool)[item.first].push_back(item.second);
    }

    // Restore the caller's device.  Failing to restore would make a
    // subsequent op on cuda:N launch on the profiler's last device.
    if (original_device >= 0 && current != original_device)
        (void)cudaSetDevice(original_device);
}

TENSORPLAY_API void gpu_drain_pool() {
    int original_device = -1;
    (void)cudaGetDevice(&original_device);
    std::unordered_map<int, std::deque<cudaEvent_t>> pool;
    std::vector<LivePair> live;
    {
        std::lock_guard<std::mutex> lock(g_gpu_mutex);
        if (g_pool) pool.swap(*g_pool);
        if (g_live) live.swap(*g_live);
    }
    int current = original_device;
    for (auto& bucket : pool) {
        if (!select_device(bucket.first, &current)) continue;
        for (auto ev : bucket.second) {
            if (ev) (void)cudaEventDestroy(ev);
        }
    }
    for (const auto& p : live) {
        destroy_event_pair(p, &current);
    }
    if (original_device >= 0 && current != original_device)
        (void)cudaSetDevice(original_device);
}

} // namespace prof
} // namespace tensorplay

#else  // !USE_CUDA

#include <functional>
#include <vector>

namespace tensorplay {
namespace prof {
// CPU builds: keep the symbols so generated code links without ifdef noise.
GpuTimerPair::~GpuTimerPair() {}
void GpuTimerPair::arm(const Device&) {}
void GpuTimerPair::close() {}
TENSORPLAY_API std::atomic<bool> g_gpu_timing{false};
TENSORPLAY_API void gpu_resolve_all(
        std::vector<Event>&,
        const std::function<void(Event&, float)>&) {}
TENSORPLAY_API void gpu_drain_pool() {}
} // namespace prof
} // namespace tensorplay

#endif // USE_CUDA
