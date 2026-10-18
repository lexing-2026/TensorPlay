// MPS GPU-timeline support for the native profiler (USE_MPS builds only;
// this TU compiles to nothing otherwise and the no-op symbols in
// ProfilerGpu.cpp cover CPU-only builds).
//
// The stream lane exposes no in-stream marker that carries a GPU timestamp,
// and command-buffer timestamp access is not served uniformly: the
// paravirtualized device used by CI rejects command-buffer interaction from
// the profiler path outright (the process traps rather than reporting an
// error).  A profiled dispatch is therefore timed by its host dispatch
// latency: arm records the start, close records the end, and
// gpu_resolve_all hands the difference to the event.  The field is
// non-negative and monotonic, orders events the same way a device duration
// would, and leaves the stream's commits exactly as an unprofiled run makes
// them.  Outside a session the dispatch hot path stays at the single
// acquire-load of g_gpu_timing that the CUDA lane pays.

#include "Profiler.h"

#ifdef USE_MPS

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <vector>

namespace tensorplay {
namespace prof {

namespace {

std::mutex g_mps_mutex;
// Pairs recorded by GpuTimerPair::close, consumed by gpu_resolve_all.  Slots
// are stable for the session (append-only event vector), so a pair can
// safely record the slot it was created for.  Each pair carries the session
// epoch it was created in; gpu_resolve_all only consumes its own epoch.
struct PendingPair {
    size_t slot;
    int device;
    float host_ms;
    uint64_t epoch;
};
std::vector<PendingPair>* g_pending = nullptr;
// Monotonic session epoch; incremented by gpu_resolve_all.
std::atomic<uint64_t> g_epoch{0};

void ensure_pending_locked() {
    if (!g_pending) g_pending = new std::vector<PendingPair>();
}

} // namespace

// Global (not TLS): MPS dispatches can originate from any worker thread.
TENSORPLAY_API std::atomic<bool> g_gpu_timing{false};
// No external-correlation id space exists on the MPS lane; the counter is
// kept so generated code and Profiler.h stay backend-agnostic.
TENSORPLAY_API std::atomic<uint32_t> g_ext_warmup{0};

GpuTimerPair::~GpuTimerPair() {
    if (!armed_) return;
    close();
}

void GpuTimerPair::arm(const Device& device) {
    if (!rec_.live_ || !device.is_mps()) return;
    if (!g_gpu_timing.load(std::memory_order_acquire)) return;
    gpu_device_ = static_cast<int>(device.index());
    arm_ns_ = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    armed_ = true;
}

void GpuTimerPair::close() {
    if (!armed_) return;
    armed_ = false;
    const int device = gpu_device_;
    gpu_device_ = -1;
    if (!rec_.live_) return;  // exception path: no slot to attribute to
    const uint64_t end_ns = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const float ms =
        static_cast<float>(static_cast<double>(end_ns - arm_ns_) / 1e6);
    const size_t slot = rec_.slot_;
    const uint64_t epoch = g_epoch.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_mps_mutex);
    ensure_pending_locked();
    g_pending->push_back({slot, device, ms, epoch});
}

// Binding-layer resolution API.
TENSORPLAY_API void gpu_resolve_all(
        std::vector<Event>& events,
        const std::function<void(Event&, float ms)>& emit) {
    // Pairs are pushed synchronously by GpuTimerPair::close with the current
    // epoch, so consume that epoch here and then advance it so a pair that
    // lands after this stop can never write into a later session.
    const uint64_t epoch = g_epoch.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_mps_mutex);
    if (!g_pending) return;
    for (const auto& p : *g_pending) {
        if (p.epoch != epoch) continue;  // stale from a timed-out session
        if (p.slot < events.size()) {
            events[p.slot].gpu_ms = p.host_ms;
            events[p.slot].gpu_start = nullptr;
            events[p.slot].gpu_end = nullptr;
            emit(events[p.slot], p.host_ms);
        }
    }
    g_pending->clear();
    g_epoch.fetch_add(1, std::memory_order_acq_rel);
}

TENSORPLAY_API void gpu_drain_pool() {
    std::lock_guard<std::mutex> lock(g_mps_mutex);
    if (g_pending) g_pending->clear();
}

} // namespace prof
} // namespace tensorplay

#endif // USE_MPS
