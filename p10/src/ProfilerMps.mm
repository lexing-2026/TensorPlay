// MPS GPU-timeline support for the native profiler (USE_MPS builds only;
// this TU compiles to nothing otherwise and the no-op symbols in
// ProfilerGpu.cpp cover CPU-only builds).
//
// The stream lane exposes no in-stream markers readable with GPU
// timestamps, so per-op GPU durations are taken from the op's own command
// buffer: close() pins the buffer the dispatch's work landed in --
// committing one the op left open (shader ops end their encoders without
// committing) on the stream serial queue -- and the timestamp read is
// deferred to session stop, where gpu_resolve_all waits for completion
// (bounded) and reads GPUStartTime/GPUEndTime from the buffer's root
// command buffer.  Waiting, committing extra buffers, and reading
// timestamps therefore never happen on the dispatch hot path; a profiled
// op pays one serial-queue snapshot plus one pin, the same class of
// per-op session cost as the CUDA lane's event pair.
//
// A driver that reports no GPU timestamps degrades to the host dispatch
// latency recorded at arm/close, keeping the gpu_ms field non-negative;
// mps_timing_take_counters reports which source each resolved pair used.
// The pin holds its own retain, so the stream's normal synchronization can
// retire the buffer in the meantime.  Outside a session the dispatch hot
// path stays at the single acquire-load of g_gpu_timing the CUDA lane pays.

#include "Profiler.h"
#include "Exception.h"

#ifdef USE_MPS

#include "backend/mps/MPSStream.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace tensorplay {
namespace prof {

namespace {

std::mutex g_mps_mutex;
// Pairs pinned by GpuTimerPair::close, consumed by gpu_resolve_all.  Slots
// are stable for the session (append-only event vector), so a pair can
// safely record the slot it was created for.  Each pair carries the session
// epoch it was created in; gpu_resolve_all only consumes its own epoch.
struct PendingPair {
    size_t slot;
    int device;
    uint64_t epoch;
    // +1 retained command buffer carrying the op's GPU work, or nil when no
    // buffer could be pinned (no GPU work dispatched, or Metal rejected the
    // pin) and host_ms below is final.
    MPSCommandBuffer_t buffer;
    // Host dispatch latency (arm -> close) in ms: the gpu_ms fallback when
    // the pinned buffer reports no GPU timestamps.
    float host_ms;
};
std::vector<PendingPair>* g_pending = nullptr;
// Monotonic session epoch; incremented by gpu_resolve_all.
std::atomic<uint64_t> g_epoch{0};
// Resolved-pair source counters (read-and-reset via mps_timing_take_counters).
std::atomic<uint64_t> g_native_count{0};
std::atomic<uint64_t> g_fallback_count{0};

// Bounded completion wait for pinned buffers at session stop; the same knob
// and default as the CUDA lane's stop-time wait.  A bounded wait turns a
// stalled driver into a profile with fallback timing instead of an
// unkillable stop.
uint64_t resolve_timeout_ms() {
    const char* raw = std::getenv("TP_PROFILER_GPU_TIMEOUT_MS");
    if (!raw || !*raw) return 5000;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0') return 5000;
    return static_cast<uint64_t>(value);
}

void ensure_pending_locked() {
    if (!g_pending) g_pending = new std::vector<PendingPair>();
}

// GPU window of a pinned buffer in milliseconds.  Completion is polled
// (never waited on unboundedly) so a stalled driver hits the deadline and
// falls back; a completed buffer whose root command buffer carries no usable
// timestamps falls back to the host dispatch latency as well.
float resolve_window_ms(MPSCommandBuffer_t buffer, float host_ms,
                        std::chrono::steady_clock::time_point deadline) {
    id<MTLCommandBuffer> root = [buffer rootCommandBuffer];
    if (root == nil) {
        g_fallback_count.fetch_add(1, std::memory_order_relaxed);
        return host_ms;
    }
    MTLCommandBufferStatus status = [root status];
    for (;;) {
        if (status == MTLCommandBufferStatusCompleted) break;
        if (status == MTLCommandBufferStatusError ||
            std::chrono::steady_clock::now() >= deadline) {
            g_fallback_count.fetch_add(1, std::memory_order_relaxed);
            return host_ms;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        status = [root status];
    }
    const double start = [root GPUStartTime];
    const double end = [root GPUEndTime];
    if (end > start) {
        g_native_count.fetch_add(1, std::memory_order_relaxed);
        return static_cast<float>((end - start) * 1000.0);
    }
    g_fallback_count.fetch_add(1, std::memory_order_relaxed);
    return host_ms;
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
    // Baseline of the stream's buffer slots: an op that dispatches no GPU
    // work (metadata-only ops on MPS tensors) leaves both slots exactly as
    // arm found them, and must not inherit the previous op's buffer.
    const auto slots = stream->bufferSlots();
    arm_buf_cur_ = const_cast<void*>(slots.current);
    arm_buf_last_ = const_cast<void*>(slots.last);
    armed_ = true;
}

void GpuTimerPair::close() {
    if (!armed_) return;
    armed_ = false;
    auto* stream = static_cast<mps::MPSStream*>(gpu_stream_);
    gpu_stream_ = nullptr;
    const int device = gpu_device_;
    gpu_device_ = -1;
    const uint64_t end_ns = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const float host_ms =
        static_cast<float>(static_cast<double>(end_ns - arm_ns_) / 1e6);
    arm_buf_cur_ = nullptr;
    arm_buf_last_ = nullptr;
    if (stream == nullptr) return;
    if (!rec_.live_) return;  // exception path: no slot to attribute to

    // Pin the op's command buffer when this dispatch added GPU work.  The
    // pin commits anything the op left open (shader ops end their encoders
    // without committing) so every profiled dispatch owns exactly one
    // buffer, takes a +1 reference that outlives the stream's normal
    // synchronization, and pushes the pair for the stop-time read.
    const auto slots = stream->bufferSlots();
    MPSCommandBuffer_t buffer = nil;
    if (slots.current != arm_buf_cur_ || slots.last != arm_buf_last_) {
        buffer = stream->commitPendingAndTakeLastBuffer();
    }
    if (buffer == nil) {
        g_fallback_count.fetch_add(1, std::memory_order_relaxed);
    }

    const size_t slot = rec_.slot_;
    const uint64_t epoch = g_epoch.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_mps_mutex);
    ensure_pending_locked();
    g_pending->push_back({slot, device, epoch, buffer, host_ms});
}

// Binding-layer resolution API.
TENSORPLAY_API void gpu_resolve_all(
        std::vector<Event>& events,
        const std::function<void(Event&, float ms)>& emit) {
    // Pairs are pushed synchronously by GpuTimerPair::close with the current
    // epoch, so consume that epoch here and then advance it so a pair that
    // lands after this stop can never write into a later session.
    const uint64_t epoch = g_epoch.load(std::memory_order_relaxed);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(resolve_timeout_ms());
    std::lock_guard<std::mutex> lock(g_mps_mutex);
    if (!g_pending) return;
    for (const auto& p : *g_pending) {
        if (p.epoch != epoch) continue;  // stale from a timed-out session
        const float ms = p.buffer == nil
            ? p.host_ms
            : resolve_window_ms(p.buffer, p.host_ms, deadline);
        if (p.slot < events.size()) {
            events[p.slot].gpu_ms = ms;
            events[p.slot].gpu_start = nullptr;
            events[p.slot].gpu_end = nullptr;
            emit(events[p.slot], ms);
        }
    }
    for (auto& p : *g_pending) {
        if (p.buffer) [p.buffer release];
    }
    g_pending->clear();
    g_epoch.fetch_add(1, std::memory_order_acq_rel);
}

// Timing-mode counters: (pairs resolved from command-buffer GPU timestamps,
// pairs resolved from the host dispatch latency).  Read-and-reset.
TENSORPLAY_API std::pair<uint64_t, uint64_t> mps_timing_take_counters() {
    const uint64_t native =
        g_native_count.exchange(0, std::memory_order_acq_rel);
    const uint64_t fallback =
        g_fallback_count.exchange(0, std::memory_order_acq_rel);
    return {native, fallback};
}

TENSORPLAY_API void gpu_drain_pool() {
    std::lock_guard<std::mutex> lock(g_mps_mutex);
    if (g_pending) {
        for (auto& p : *g_pending) {
            if (p.buffer) [p.buffer release];
        }
        g_pending->clear();
    }
}

} // namespace prof
} // namespace tensorplay

#endif // USE_MPS
