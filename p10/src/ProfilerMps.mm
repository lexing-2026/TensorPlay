// MPS GPU-timeline support for the native profiler (USE_MPS builds only;
// this TU compiles to nothing otherwise and the no-op symbols in
// ProfilerGpu.cpp cover CPU-only builds).
//
// MPS exposes no cudaEvent-style stream markers with GPU timestamps, so the
// per-op GPU window is read from the Metal command buffer boundary instead:
// while g_gpu_timing is active, graph ops have a completion handler attached
// by MPSStream::executeMPSGraph before their command buffer is committed,
// and shader-based ops are committed and waited on inside
// GpuTimerPair::close, reading the committed buffer's GPUStartTime/GPUEndTime
// synchronously.  This yields a per-op GPU duration that is paid only inside
// an active gpu_timing session -- the dispatch hot path outside a session
// costs the same single acquire-load of g_gpu_timing as the CUDA lane.  The
// resolved pairs are consumed by gpu_resolve_all when the session stops.

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
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tensorplay {
namespace prof {

namespace {

std::mutex g_mps_mutex;
// Pairs collected by GpuTimerPair::close and by graph-op completion
// handlers, consumed by gpu_resolve_all.  Slots are stable for the session
// (append-only event vector), so a pair can safely record the slot it was
// created for.  Each pair carries the session epoch it was created in;
// gpu_resolve_all only consumes its own epoch.
struct PendingPair {
    size_t slot;
    int device;
    float gpu_ms;
    uint64_t epoch;
};
std::vector<PendingPair>* g_pending = nullptr;
// Number of graph-op completion handlers that have not yet run.  The stop
// path waits on this reaching zero so the events vector is only touched
// after every in-flight command buffer has completed.
std::atomic<int> g_pending_count{0};
std::condition_variable g_pending_cv;
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

// The GpuTimerPair whose op dispatch is currently running.  MPS dispatches
// are submitted serially from the calling thread, and MPSStream::executeMPSGraph
// runs its encode block on the stream's serial queue (a different GCD
// thread), so this must be a process-global rather than thread-local to be
// visible to the graph encode path.
GpuTimerPair* g_current_gpu_pair = nullptr;

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
    gpu_epoch_ = g_epoch.load(std::memory_order_relaxed);
    armed_ = true;
    g_current_gpu_pair = this;
}

void GpuTimerPair::close() {
    if (!armed_) return;
    armed_ = false;
    if (g_current_gpu_pair == this) {
        g_current_gpu_pair = nullptr;
    }
    auto* stream = static_cast<mps::MPSStream*>(gpu_stream_);
    gpu_stream_ = nullptr;
    const int device = gpu_device_;
    gpu_device_ = -1;
    if (stream == nullptr) return;
    if (!rec_.live_) return;  // exception path: no slot to attribute to

    const size_t slot = rec_.slot_;
    const uint64_t epoch = gpu_epoch_;
    // Graph ops committed inside MPSStream::executeMPSGraph already installed
    // a completion handler that records the GPU duration; waiting on the
    // committed buffer here would race MPSGraph's deferred encoding.  For
    // shader-based ops the buffer is still uncommitted, so commit it, wait,
    // and read the timestamps directly.
    if (gpu_handler_installed_) {
        return;
    }
    float ms = 0.0f;
    @try {
        fprintf(stderr, "TP_PROF_DEBUG close slot=%zu before commitAndWait\n", slot);
        ms = static_cast<float>(stream->commitAndWaitGetGpuTimeMs());
        fprintf(stderr, "TP_PROF_DEBUG close slot=%zu after commitAndWait ms=%f\n", slot, ms);
    } @catch (NSException* e) {
        fprintf(stderr, "TP_PROF_DEBUG close slot=%zu caught NSException: %s\n",
                slot, [[e reason] UTF8String]);
        ms = 0.0f;
    }
    std::lock_guard<std::mutex> lock(g_mps_mutex);
    ensure_pending_locked();
    g_pending->push_back({slot, device, ms, epoch});
}

// Returns the active MPS GpuTimerPair, or nullptr.
TENSORPLAY_API GpuTimerPair* mps_current_gpu_pair() {
    return g_current_gpu_pair;
}

// Called by MPSStream::executeMPSGraph when it installs a graph-op completion
// handler; the stop path waits for every installed handler to fire.
TENSORPLAY_API void mps_on_handler_installed() {
    g_pending_count.fetch_add(1, std::memory_order_relaxed);
}

// Records a profiled MPS op's GPU duration from a completed command buffer.
TENSORPLAY_API void mps_record_gpu_time(uint64_t slot, int device,
                                         uint64_t epoch,
                                         id<MTLCommandBuffer> cb) {
    float ms = 0.f;
    if (cb.GPUEndTime > cb.GPUStartTime) {
        ms = static_cast<float>((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
    }
    std::lock_guard<std::mutex> lock(g_mps_mutex);
    ensure_pending_locked();
    g_pending->push_back({slot, device, ms, epoch});
    if (g_pending_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        g_pending_cv.notify_all();
    }
}

// Binding-layer resolution API.
TENSORPLAY_API void gpu_resolve_all(
        std::vector<Event>& events,
        const std::function<void(Event&, float ms)>& emit) {
    // Graph-op pairs are pushed by completion handlers; wait (bounded) for
    // every installed handler to land, then consume this session's epoch and
    // advance it so a pair that lands after this stop can never write into a
    // later session.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(5000);
    std::unique_lock<std::mutex> lock(g_mps_mutex);
    g_pending_cv.wait(lock, [&] {
        return g_pending_count.load(std::memory_order_acquire) == 0 ||
               std::chrono::steady_clock::now() >= deadline;
    });
    if (!g_pending) return;
    const uint64_t epoch = g_epoch.load(std::memory_order_relaxed);
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
    g_pending_count.store(0, std::memory_order_release);
    g_pending_cv.notify_all();
}

} // namespace prof
} // namespace tensorplay

#endif // USE_MPS