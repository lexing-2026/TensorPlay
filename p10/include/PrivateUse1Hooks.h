#pragma once

// Device hooks of the out-of-tree backend.
//
// An extension that implements a device in the PrivateUse1 slot provides, at
// load time:
//
//   1. a name for the device (register_privateuse1_backend, Device.h);
//   2. an allocator (REGISTER_ALLOCATOR(DeviceType::PrivateUse1, ...),
//      Allocator.h), which every factory kernel of the device draws from;
//   3. kernels under the PrivateUse1 dispatch key
//      (TENSORPLAY_LIBRARY_IMPL(PrivateUse1, ...), Dispatcher.h); the
//      autograd, autocast and vmap layers of the key exist already, so
//      gradients flow through the backend's kernels with no further work;
//   4. these hooks, which answer the runtime's device-level questions:
//      how many devices exist, which one is current, how to wait for one,
//      and how to move raw bytes in and out of device memory.
//
// Every hook has a default that suits a single, synchronous device whose
// memory the host cannot read; a backend overrides what differs.

#include <cstddef>
#include <cstdint>
#include <utility>

#include "Device.h"
#include "Macros.h"

namespace tensorplay {

struct P10_API PrivateUse1HooksInterface {
    virtual ~PrivateUse1HooksInterface();

    // Whether the device can run work in this process (a driver is present,
    // a device answers).
    virtual bool is_available() const { return true; }

    // Number of devices of this type, and the index the thread works on.
    virtual int64_t device_count() const { return 1; }
    virtual int64_t current_device() const { return 0; }
    // Make ``index`` the thread's current device.  The default accepts the
    // only device a single-device backend has.
    virtual void set_device(int64_t index) const;

    // Block until all work queued on device ``index`` (current device when
    // negative) has finished.  The default suits a backend whose kernels
    // complete before they return.
    virtual void synchronize(int64_t index) const { (void)index; }

    // Copy ``nbytes`` between two allocations, at least one of which lives
    // on this device; the other is host memory or another allocation of
    // this device.  Storage resizing relies on it.  The default raises.
    virtual void copy_bytes(void* destination, const Device& destination_device,
                            const void* source, const Device& source_device,
                            size_t nbytes) const;

    // (free, total) bytes of device ``index``.  The default raises.
    virtual std::pair<size_t, size_t> memory_info(int64_t index) const;

    // Whether ``data`` is host memory pinned for fast transfers to this
    // device.
    virtual bool is_pinned_ptr(const void* data) const {
        (void)data;
        return false;
    }
};

// Installs the backend's hooks; ``hooks`` must outlive the process (a
// function-local static or a leaked allocation).  Registering a second,
// different object raises.
P10_API void register_privateuse1_hooks(PrivateUse1HooksInterface* hooks);
P10_API bool is_privateuse1_hooks_registered();
// The installed hooks; raises naming the missing registration when none is
// installed.
P10_API PrivateUse1HooksInterface& get_privateuse1_hooks();

namespace detail {
struct PrivateUse1HooksRegisterer {
    explicit PrivateUse1HooksRegisterer(PrivateUse1HooksInterface* hooks) {
        register_privateuse1_hooks(hooks);
    }
};
} // namespace detail

// Registers the hooks object at load time:
//   static MyHooks hooks;
//   REGISTER_PRIVATEUSE1_HOOKS(&hooks);
#define REGISTER_PRIVATEUSE1_HOOKS(hooks_ptr)                              \
    static ::tensorplay::detail::PrivateUse1HooksRegisterer                \
        TP_CONCAT(tensorplay_privateuse1_hooks_, __LINE__)(hooks_ptr)

} // namespace tensorplay
