#pragma once

#include <cstdint>

#include "Device.h"
#include "Macros.h"

namespace tensorplay {

// A backend-agnostic value class representing a stream on a device.  A stream
// is identified by its device plus an opaque id, where id 0 is always the
// device's default stream.  Autograd uses this to record which stream
// produced a gradient and which stream a node expects to consume on, so the
// engine can insert the proper event-based ordering between producers and
// consumers.
class P10_API Stream final {
public:
    enum Unsafe { UNSAFE };
    enum Default { DEFAULT };

    explicit Stream(Unsafe /*unused*/, Device device, uint64_t id)
        : device_(device), id_(id) {}

    explicit Stream(Default /*unused*/, Device device)
        : device_(device), id_(0) {}

    bool operator==(const Stream& other) const noexcept {
        return device_ == other.device_ && id_ == other.id_;
    }
    bool operator!=(const Stream& other) const noexcept {
        return !(*this == other);
    }

    Device device() const noexcept { return device_; }
    DeviceType device_type() const noexcept { return device_.type(); }
    int64_t device_index() const noexcept { return device_.index(); }
    uint64_t id() const noexcept { return id_; }

    // Opaque handle to the underlying backend stream (a cudaStream_t for
    // CUDA), valid only on the stream's own device.
    void* native_handle() const noexcept {
        return reinterpret_cast<void*>(id_);
    }

private:
    Device device_;
    uint64_t id_;
};

} // namespace tensorplay