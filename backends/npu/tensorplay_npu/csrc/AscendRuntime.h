#pragma once

// Ascend runtime glue: error checking, the per-device stream every kernel of
// this backend is ordered on, the thread's current device, and a guard that
// makes a device current for the length of a call.

#include <cstddef>
#include <cstdint>
#include <utility>

#include <acl/acl.h>

#include "Device.h"
#include "Exception.h"

namespace tensorplay {
namespace npu {

#define TP_ACL_CHECK(expr)                                                     \
    do {                                                                       \
        const aclError tp_acl_status = (expr);                                 \
        TP_CHECK(tp_acl_status == ACL_SUCCESS, "Ascend runtime error: ",       \
                 aclGetRecentErrMsg(), " (", #expr, ")");                      \
    } while (0)

// Number of visible Ascend devices (0 when the driver is absent).
int device_count() noexcept;

int current_device();
void set_device(int index);

// Device index a tensor on this backend lives on (a negative index means
// the current device).
int index_of(const Device& device);

// Make ``index`` the current device for the guard's lifetime.
class DeviceGuard {
public:
    explicit DeviceGuard(int index);
    explicit DeviceGuard(const Device& device) : DeviceGuard(index_of(device)) {}
    ~DeviceGuard();
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int previous_;
    int current_;
};

// The stream all work on ``index`` is issued to.  One stream per device
// keeps kernels, copies and allocator reuse in program order.
aclrtStream stream(int index);

// Block until all work issued to the device's stream has finished.
void synchronize(int index);

// Byte copies between the host and a device, ordered on the device's stream;
// they return once the bytes have landed.
void copy_to_device(void* device_dst, const void* host_src, size_t nbytes, int index);
void copy_to_host(void* host_dst, const void* device_src, size_t nbytes, int index);
void copy_on_device(void* dst, const void* src, size_t nbytes, int index);

// Fill ``nbytes`` of device memory with the repeating byte ``value``.
void memset_on_device(void* dst, int32_t value, size_t nbytes, int index);

// (free, total) bytes of device ``index``.
std::pair<size_t, size_t> memory_info(int index);

} // namespace npu
} // namespace tensorplay
