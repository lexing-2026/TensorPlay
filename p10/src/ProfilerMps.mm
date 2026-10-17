// MPS GPU-timeline support for the native profiler (USE_MPS builds only;
// this TU compiles to nothing otherwise and the no-op symbols in
// ProfilerGpu.cpp cover CPU-only builds).
//
// MPS exposes no cudaEvent-style stream markers with GPU timestamps, and the
// paravirtualized Metal layer on CI runners rejects attaching completion
// handlers or committing command buffers from the profiler hot path.  The
// per-op GPU field is therefore filled with the host-side dispatch latency
// measured around the op (arm -> close), which is non-negative and
// monotonic, while the command buffer itself is left to the stream's normal
// synchronization.  This keeps the dispatch hot path outside a gpu_timing
// session at the same single acquire-load of g_gpu_timing as the CUDA lane.
// The resolved pairs are consumed by gpu_resolve_all when the session stops.

#include "Profiler.h"
#include "Exception.h"

#ifdef USE_MPS

#include "backend/mps/MPSStream.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

namespace tensorplay {
namespace prof {

namespace {

std::mutex g_mps_mutex;
// Pairs collected by GpuTimerPair::close, consumed by gpu_resolve_all.
// Slots are stable for the session (append-only event vector), so a pair can
// safely record the slot it was created for.  Each pair carries the session
// epoch it was created in; gpu_resolve_all only consumes its own epoch.
struct PendingPair {
    size_t slot;
    int device;
    float gpu_ms;
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
    mps::MPSStream* stream = mps::getCurrentMPSStream();
    if (stream == nullptr) return;
    gpu_stream_ = stream;
    gpu_device_ = stream->device_index();
    arm_ns_ = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    armed_ = true;
}

void GpuTimerPair::close() {
    if (!armed_) return;
    armed_ = false;
    auto* stream = static_cast<mps::MPSStream*>(gpu_stream_);
    gpu_stream_ = nullptr;
    const int device = gpu_device_;
    gpu_device_ = -1;
    if (stream == nullptr) return;
    if (!rec_.live_) return;  // exception path: no slot to attribute to

    const size_t slot = rec_.slot_;
    const uint64_t epoch = g_epoch.load(std::memory_order_relaxed);
    // Host-side dispatch latency in milliseconds.  The op's command buffer is
    // left to the stream's normal synchronization path; no Metal timestamp is
    // read from the profiler hot path.
    const uint64_t end_ns = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const float ms = static_cast<float>((end_ns - arm_ns_) / 1000000.0);
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
            events[p.slot].gpu_ms = p.gpu_ms;
            events[p.slot].gpu_start = nullptr;
            events[p.slot].gpu_end = nullptr;
            emit(events[p.slot], p.gpu_ms);
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