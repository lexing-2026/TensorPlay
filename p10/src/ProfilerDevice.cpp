// Device time of operations on the out-of-tree device (PrivateUse1).
//
// The backend registers DeviceTimerHooks.  While a session has device timing
// on, GpuTimerPair records one event before an op dispatched to the device
// and one after it; the pairs wait here until the session stops, when the
// binding layer resolves them into Event::gpu_ms.  Elapsed times are only
// read at stop, so the dispatch path never waits on the device.

#include "Profiler.h"

#include <atomic>
#include <mutex>
#include <vector>

namespace tensorplay {
namespace prof {

namespace {

std::atomic<const DeviceTimerHooks*> g_hooks{nullptr};
std::atomic<DeviceActivityHooks*> g_activity_hooks{nullptr};

struct DevicePair {
    size_t slot;
    void* start;
    void* end;
};

std::mutex& pairs_mutex() {
    static std::mutex m;
    return m;
}

// Leaked: pairs may still be closed by threads unwinding at exit.
std::vector<DevicePair>& live_pairs() {
    static auto* pairs = new std::vector<DevicePair>;
    return *pairs;
}

} // namespace

std::atomic<bool> g_device_trace{false};

void register_privateuse1_activity_hooks(DeviceActivityHooks* hooks) {
    g_activity_hooks.store(hooks, std::memory_order_release);
}

DeviceActivityHooks* privateuse1_activity_hooks() {
    return g_activity_hooks.load(std::memory_order_acquire);
}

void register_privateuse1_timer_hooks(const DeviceTimerHooks* hooks) {
    g_hooks.store(hooks, std::memory_order_release);
}

const DeviceTimerHooks* privateuse1_timer_hooks() {
    return g_hooks.load(std::memory_order_acquire);
}

bool GpuTimerPair::arm_device(const Device& device) {
    if (!device.is_privateuse1()) return false;
    if (g_device_trace.load(std::memory_order_acquire)) {
        if (DeviceActivityHooks* activity = privateuse1_activity_hooks()) {
            activity->push_external_id(static_cast<uint64_t>(rec_.slot_));
            device_pushed_ = true;
        }
    }
    if (!g_gpu_timing.load(std::memory_order_acquire)) return true;
    const DeviceTimerHooks* hooks = privateuse1_timer_hooks();
    if (hooks == nullptr) return true;
    const int index = device.index() < 0 ? hooks->current_device()
                                         : static_cast<int>(device.index());
    void* start = hooks->record(index);
    if (start == nullptr) return true;  // profiling never fails the op
    device_start_ = start;
    device_index_ = index;
    return true;
}

void GpuTimerPair::close_device() {
    if (device_pushed_) {
        device_pushed_ = false;
        if (DeviceActivityHooks* activity = privateuse1_activity_hooks()) {
            activity->pop_external_id();
        }
    }
    void* start = device_start_;
    const int index = device_index_;
    device_start_ = nullptr;
    device_index_ = -1;
    const DeviceTimerHooks* hooks = privateuse1_timer_hooks();
    if (start == nullptr || hooks == nullptr) return;
    // An op unwound after its record closed has no slot to attribute to.
    void* end = rec_.live_ ? hooks->record(index) : nullptr;
    if (end == nullptr) {
        hooks->release(start);
        return;
    }
    std::lock_guard<std::mutex> lock(pairs_mutex());
    live_pairs().push_back({rec_.slot_, start, end});
}

void resolve_device_pairs(std::vector<Event>& events,
                          const std::function<void(Event&, float ms)>& emit) {
    std::vector<DevicePair> pairs;
    {
        std::lock_guard<std::mutex> lock(pairs_mutex());
        pairs.swap(live_pairs());
    }
    const DeviceTimerHooks* hooks = privateuse1_timer_hooks();
    if (hooks == nullptr) return;
    // A queue completes in order, so the elapsed reads after the first wait
    // mostly find their end events done already.
    for (const DevicePair& p : pairs) {
        float ms = -1.f;
        if (hooks->elapsed_ms(p.start, p.end, &ms) && ms >= 0.f && p.slot < events.size()) {
            events[p.slot].gpu_ms = ms;
            emit(events[p.slot], ms);
        }
        hooks->release(p.start);
        hooks->release(p.end);
    }
}

void drain_device_pairs() {
    std::vector<DevicePair> pairs;
    {
        std::lock_guard<std::mutex> lock(pairs_mutex());
        pairs.swap(live_pairs());
    }
    const DeviceTimerHooks* hooks = privateuse1_timer_hooks();
    if (hooks == nullptr) return;
    for (const DevicePair& p : pairs) {
        hooks->release(p.start);
        hooks->release(p.end);
    }
}

} // namespace prof
} // namespace tensorplay
