// MPS GPU-timeline support for the native profiler (USE_MPS builds only;
// this TU compiles to nothing otherwise and the no-op symbols in
// ProfilerGpu.cpp cover CPU-only builds).
//
// MPS exposes no cudaEvent-style stream markers with GPU timestamps, so the
// per-op GPU window is read from the Metal command buffer boundary instead:
// while g_gpu_timing is active, each profiled MPS dispatch commits its
// MPSCommandBuffer (commit-and-continue) at close() and reads the committed
// buffer's GPUStartTime/GPUEndTime in its completion handler.  This yields a
// per-op GPU duration at the cost of one commit per op, which is paid only
// inside an active gpu_timing session -- the dispatch hot path outside a
// session costs the same single acquire-load of g_gpu_timing as the CUDA
// lane.  The completion handler runs asynchronously; gpu_resolve_all waits
// for every in-flight handler (bounded) before writing gpu_ms into events.

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
// Pairs collected by completion handlers, awaiting gpu_resolve_all.  Slots
// are stable for the session (append-only event vector), so a handler can
// safely record the slot it was created for.  Each pair carries the session
// epoch it was created in; gpu_resolve_all only consumes its own epoch, so
// a handler that lands after a timed-out stop can never write into a later
// session's events.
struct PendingPair {
    size_t slot;
    int device;
    float gpu_ms;
    uint64_t epoch;
};
std::vector<PendingPair>* g_pending = nullptr;
// Number of completion handlers that have not yet run.  gpu_resolve_all
// waits on this reaching zero (with a timeout) so the events vector is only
// touched after every in-flight command buffer has completed.
std::atomic<int> g_pending_count{0};
std::condition_variable g_pending_cv;
// Monotonic session epoch; incremented by gpu_resolve_all.
std::atomic<uint64_t> g_epoch{0};

void ensure_pending_locked() {
    if (!g_pending) g_pending = new std::vector<PendingPair>();
}

uint64_t resolve_timeout_ms() {
    // A bounded wait turns a hung Metal command buffer into a profile with
    // missing GPU fields instead of an unkillable profiler stop.
    const char* raw = std::getenv("TP_PROFILER_GPU_TIMEOUT_MS");
    if (!raw || !*raw) return 5000;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0') return 5000;
    return static_cast<uint64_t>(value);
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
    {
        std::lock_guard<std::mutex> lock(g_mps_mutex);
        ensure_pending_locked();
    }
    // Increment before installing the handler: Metal may invoke a completed
    // command buffer's handler synchronously inside addCompletedHandler, and
    // the decrement must always have a matching increment.
    g_pending_count.fetch_add(1, std::memory_order_relaxed);
    @try {
        stream->addCompletedHandler(^(id<MTLCommandBuffer> cb) {
            float ms = 0.f;
            if (cb.GPUEndTime > cb.GPUStartTime) {
                ms = static_cast<float>((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
            }
            std::lock_guard<std::mutex> lock(g_mps_mutex);
            if (g_pending) g_pending->push_back({slot, device, ms, epoch});
            if (g_pending_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                g_pending_cv.notify_all();
            }
        });
        // Commit the command buffer so the completion handler above fires once
        // this op's encoded work has executed.  The stream runs with
        // commit-and-continue disabled, so a plain COMMIT (flush) is used: the
        // handler was attached before the commit, and the stream is left
        // with a nil command buffer ready for the next dispatch.
        stream->synchronize(mps::SyncType::COMMIT);
    } @catch (NSException* e) {
        // A Metal failure here must surface as a normal runtime error
        // rather than aborting the process from the profiler hot path.
        g_pending_count.fetch_sub(1, std::memory_order_acq_rel);
        std::string reason =
            [e reason] ? std::string([[e reason] UTF8String])
                       : std::string("unknown Metal exception");
        throw tensorplay::RuntimeError(
            {__FILE__, __func__, __LINE__},
            tensorplay::detail::format_msg("MPS profiler close failed: ", reason));
    }
}

// Binding-layer resolution API.
TENSORPLAY_API void gpu_resolve_all(
        std::vector<Event>& events,
        const std::function<void(Event&, float ms)>& emit) {
    const uint64_t epoch = g_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(resolve_timeout_ms());
    std::unique_lock<std::mutex> lock(g_mps_mutex);
    g_pending_cv.wait(lock, [&] {
        return g_pending_count.load(std::memory_order_acquire) == 0 ||
               std::chrono::steady_clock::now() >= deadline;
    });
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
}

TENSORPLAY_API void gpu_drain_pool() {
    std::lock_guard<std::mutex> lock(g_mps_mutex);
    if (g_pending) g_pending->clear();
    g_pending_cv.notify_all();
}

} // namespace prof
} // namespace tensorplay

#endif // USE_MPS