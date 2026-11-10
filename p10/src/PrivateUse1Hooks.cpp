#include "PrivateUse1Hooks.h"

#include <atomic>
#include <mutex>

#include "Exception.h"

namespace tensorplay {

namespace {

std::atomic<PrivateUse1HooksInterface*> installed_hooks{nullptr};
std::mutex hooks_lock;

} // namespace

PrivateUse1HooksInterface::~PrivateUse1HooksInterface() = default;

void PrivateUse1HooksInterface::set_device(int64_t index) const {
    TP_CHECK(index >= 0 && index < device_count(),
             "device index ", index, " is out of range for '",
             get_privateuse1_backend(), "', which has ", device_count(),
             " device(s)");
    TP_CHECK(index == current_device(),
             "the '", get_privateuse1_backend(),
             "' backend does not implement set_device in its hooks");
}

void PrivateUse1HooksInterface::copy_bytes(void* destination,
                                           const Device& destination_device,
                                           const void* source,
                                           const Device& source_device,
                                           size_t nbytes) const {
    (void)destination;
    (void)source;
    (void)nbytes;
    TP_THROW(NotImplementedError,
             "the '" + get_privateuse1_backend() +
                 "' backend does not implement copy_bytes in its hooks, so "
                 "bytes cannot move from " + source_device.toString() +
                 " to " + destination_device.toString());
}

std::pair<size_t, size_t> PrivateUse1HooksInterface::memory_info(int64_t index) const {
    (void)index;
    TP_THROW(NotImplementedError,
             "the '" + get_privateuse1_backend() +
                 "' backend does not implement memory_info in its hooks");
}

void register_privateuse1_hooks(PrivateUse1HooksInterface* hooks) {
    TP_CHECK(hooks != nullptr, "register_privateuse1_hooks: hooks must not be null");
    std::lock_guard<std::mutex> guard(hooks_lock);
    PrivateUse1HooksInterface* current = installed_hooks.load(std::memory_order_acquire);
    if (current == hooks) return;
    TP_CHECK(current == nullptr,
             "hooks for the out-of-tree backend '", get_privateuse1_backend(),
             "' are already registered");
    installed_hooks.store(hooks, std::memory_order_release);
}

bool is_privateuse1_hooks_registered() {
    return installed_hooks.load(std::memory_order_acquire) != nullptr;
}

PrivateUse1HooksInterface& get_privateuse1_hooks() {
    PrivateUse1HooksInterface* hooks = installed_hooks.load(std::memory_order_acquire);
    if (hooks == nullptr) {
        TP_THROW(NotImplementedError,
                 "no device hooks are registered for the out-of-tree backend '" +
                     get_privateuse1_backend() +
                     "'; its extension registers them with "
                     "REGISTER_PRIVATEUSE1_HOOKS");
    }
    return *hooks;
}

} // namespace tensorplay
