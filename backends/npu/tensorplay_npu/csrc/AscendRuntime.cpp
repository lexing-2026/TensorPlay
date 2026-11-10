#include "AscendRuntime.h"

#include <array>
#include <atomic>
#include <mutex>

namespace tensorplay {
namespace npu {

namespace {

constexpr int kMaxDevices = 64;

std::array<std::atomic<aclrtStream>, kMaxDevices> g_streams{};
std::mutex g_stream_lock;

// Bringing the runtime up is a process-wide, one-time step.  A process in
// which another library already did it answers with an error here; the
// runtime is usable either way.
void runtime_init() {
    static std::once_flag flag;
    std::call_once(flag, [] { (void)aclInit(nullptr); });
}

// The device this thread is set to, as far as this backend knows.  The
// runtime keeps the same value per thread, so the cache saves a call per
// kernel.
int& thread_device() {
    thread_local int cached = -1;
    return cached;
}

void check_index(int index) {
    const int count = device_count();
    TP_CHECK(index >= 0 && index < count, "invalid NPU device index ", index, " (",
             count, " device(s) visible)");
}

} // namespace

int device_count() noexcept {
    static const int count = [] {
        runtime_init();
        uint32_t n = 0;
        if (aclrtGetDeviceCount(&n) != ACL_SUCCESS) return 0;
        return static_cast<int>(n < kMaxDevices ? n : kMaxDevices);
    }();
    return count;
}

int current_device() {
    int& cached = thread_device();
    if (cached >= 0) return cached;
    runtime_init();
    int32_t device = 0;
    if (aclrtGetDevice(&device) != ACL_SUCCESS) {
        // A thread that has not used the device yet starts on device 0.
        (void)aclrtSetDevice(0);
        device = 0;
    }
    cached = device;
    return cached;
}

void set_device(int index) {
    check_index(index);
    int& cached = thread_device();
    if (cached == index) return;
    TP_ACL_CHECK(aclrtSetDevice(index));
    cached = index;
}

int index_of(const Device& device) {
    return device.index() < 0 ? current_device() : static_cast<int>(device.index());
}

DeviceGuard::DeviceGuard(int index) : previous_(current_device()), current_(index) {
    if (current_ != previous_) set_device(current_);
}

DeviceGuard::~DeviceGuard() {
    if (current_ != previous_) {
        aclrtSetDevice(previous_);
        thread_device() = previous_;
    }
}

aclrtStream stream(int index) {
    check_index(index);
    aclrtStream s = g_streams[index].load(std::memory_order_acquire);
    if (s != nullptr) return s;
    std::lock_guard<std::mutex> lock(g_stream_lock);
    s = g_streams[index].load(std::memory_order_relaxed);
    if (s == nullptr) {
        DeviceGuard guard(index);
        TP_ACL_CHECK(aclrtCreateStream(&s));
        g_streams[index].store(s, std::memory_order_release);
    }
    return s;
}

void synchronize(int index) {
    DeviceGuard guard(index);
    TP_ACL_CHECK(aclrtSynchronizeStream(stream(index)));
}

void copy_to_device(void* device_dst, const void* host_src, size_t nbytes, int index) {
    if (nbytes == 0) return;
    DeviceGuard guard(index);
    aclrtStream s = stream(index);
    TP_ACL_CHECK(aclrtMemcpyAsync(device_dst, nbytes, const_cast<void*>(host_src), nbytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE, s));
    // The source is ordinary host memory the caller may free on return.
    TP_ACL_CHECK(aclrtSynchronizeStream(s));
}

void copy_to_host(void* host_dst, const void* device_src, size_t nbytes, int index) {
    if (nbytes == 0) return;
    DeviceGuard guard(index);
    aclrtStream s = stream(index);
    TP_ACL_CHECK(aclrtMemcpyAsync(host_dst, nbytes, const_cast<void*>(device_src), nbytes,
                                  ACL_MEMCPY_DEVICE_TO_HOST, s));
    TP_ACL_CHECK(aclrtSynchronizeStream(s));
}

void copy_on_device(void* dst, const void* src, size_t nbytes, int index) {
    if (nbytes == 0) return;
    DeviceGuard guard(index);
    TP_ACL_CHECK(aclrtMemcpyAsync(dst, nbytes, const_cast<void*>(src), nbytes,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE, stream(index)));
}

void memset_on_device(void* dst, int32_t value, size_t nbytes, int index) {
    if (nbytes == 0) return;
    DeviceGuard guard(index);
    TP_ACL_CHECK(aclrtMemsetAsync(dst, nbytes, value, nbytes, stream(index)));
}

std::pair<size_t, size_t> memory_info(int index) {
    DeviceGuard guard(index);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes) != ACL_SUCCESS) {
        TP_ACL_CHECK(aclrtGetMemInfo(ACL_DDR_MEM, &free_bytes, &total_bytes));
    }
    return {free_bytes, total_bytes};
}

} // namespace npu
} // namespace tensorplay
