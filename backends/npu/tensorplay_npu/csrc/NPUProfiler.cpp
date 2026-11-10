// Device time for the profiler: an event recorded on the device's stream
// before and after each operation (see DeviceTimerHooks in Profiler.h).
// A session with device timing on then reports per-operation NPU time; the
// toolkit exposes no in-process activity stream, so per-operation spans are
// the finest granularity the device can report here.
//
// Events are pooled per device: a session records two per operation, and
// creating one is a driver call the dispatch path should not pay for each
// time.  Elapsed times are read only when the session stops, so the dispatch
// path never waits for the device.

#include <array>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "AscendRuntime.h"
#include "Profiler.h"

namespace tensorplay {
namespace npu {

namespace {

constexpr int kMaxDevices = 64;

class EventTimer final : public prof::DeviceTimerHooks {
public:
    int current_device() const override { return npu::current_device(); }

    void* record(int device) const override {
        if (device < 0 || device >= device_count()) return nullptr;
        aclrtEvent event = nullptr;
        try {
            DeviceGuard guard(device);
            event = take(device);
            if (event == nullptr) return nullptr;
            if (aclrtRecordEvent(event, stream(device)) == ACL_SUCCESS) return event;
        } catch (...) {
            // The hooks must never fail an operation.
        }
        give_back(event, device);
        return nullptr;
    }

    bool elapsed_ms(void* start, void* end, float* ms) const override {
        auto* begin = static_cast<aclrtEvent>(start);
        auto* finish = static_cast<aclrtEvent>(end);
        int device = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = device_of_.find(finish);
            if (it != device_of_.end()) device = it->second;
        }
        if (device < 0 || device >= device_count()) return false;
        try {
            // The wait and the elapsed read run on the event's device.
            DeviceGuard guard(device);
            if (aclrtSynchronizeEvent(finish) != ACL_SUCCESS) return false;
            return aclrtEventElapsedTime(ms, begin, finish) == ACL_SUCCESS;
        } catch (...) {
            return false;
        }
    }

    void release(void* event) const override {
        if (event == nullptr) return;
        const aclrtEvent handle = static_cast<aclrtEvent>(event);
        int device = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = device_of_.find(handle);
            if (it != device_of_.end()) device = it->second;
        }
        give_back(handle, device);
    }

private:
    aclrtEvent take(int device) const {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<aclrtEvent>& pool = pool_[device];
            if (!pool.empty()) {
                aclrtEvent event = pool.back();
                pool.pop_back();
                return event;
            }
        }
        aclrtEvent event = nullptr;
        if (aclrtCreateEvent(&event) != ACL_SUCCESS) return nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        device_of_[event] = device;
        return event;
    }

    void give_back(aclrtEvent event, int device) const {
        if (event == nullptr || device < 0 || device >= kMaxDevices) return;
        std::lock_guard<std::mutex> lock(mutex_);
        pool_[device].push_back(event);
    }

    mutable std::mutex mutex_;
    mutable std::array<std::vector<aclrtEvent>, kMaxDevices> pool_;
    // Where each pooled event was created: an event can only be recorded on
    // its own device.
    mutable std::unordered_map<aclrtEvent, int> device_of_;
};

// Leaked with its pool: events must not be destroyed while the driver is
// tearing down at exit.
[[maybe_unused]] const EventTimer* const timer = [] {
    auto* hooks = new EventTimer;
    prof::register_privateuse1_timer_hooks(hooks);
    return hooks;
}();

} // namespace

} // namespace npu
} // namespace tensorplay
