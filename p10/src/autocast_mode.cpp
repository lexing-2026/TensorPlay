#include "autocast_mode.h"

#include "Exception.h"

namespace tensorplay {
namespace autocast {

namespace {

// nesting tracks the nesting depth of the Python-side context manager.
// When the autocast context manager exits to a nesting level that's outside
// any instance of autocast (which should occur at the end of each forward
// pass) it calls clear_cache() to ensure cached Tensors don't leak outside
// the autocasting region.
thread_local int nesting = 0;

// The order of this array MUST exactly match the definition order of
// DeviceType in Device.h.
constexpr size_t kNumDeviceTypes = 3; // CPU, CUDA, Unknown
thread_local std::array<DType, kNumDeviceTypes> autocast_dtype = {
    DType::BFloat16, // CPU
    DType::Float16,  // CUDA
    DType::Undefined, // Unknown
};

// Should we enable the cache inside autocast.
thread_local bool cache_enabled = true;

// Per-device enabled flags.  torch models these through the dispatcher's TLS
// excluded set (all Autocast keys start excluded); a plain flag per device is
// the equivalent contract for TensorPlay's explicit-key dispatch.
thread_local std::array<bool, kNumDeviceTypes> autocast_enabled = {
    false, // CPU
    false, // CUDA
    false, // Unknown
};

std::unordered_map<TensorImpl*, val_type>& get_cached_casts() {
    static std::unordered_map<TensorImpl*, val_type> cached_casts;
    return cached_casts;
}

std::mutex& cached_casts_mutex() {
    static std::mutex mutex;
    return mutex;
}

size_t device_index(DeviceType device_type) {
    switch (device_type) {
        case DeviceType::CPU:
            return 0;
        case DeviceType::CUDA:
            return 1;
        default:
            return 2;
    }
}

} // anonymous namespace

bool is_enabled(DispatchKey autocast_key) {
    return autocast_enabled[device_index(
        get_device_type_from_autocast_key(autocast_key))];
}

void set_enabled(DispatchKey autocast_key, bool enabled) {
    autocast_enabled[device_index(
        get_device_type_from_autocast_key(autocast_key))] = enabled;
}

DType get_autocast_dtype(DeviceType device_type) {
    return autocast_dtype[device_index(device_type)];
}

void set_autocast_dtype(DeviceType device_type, DType dtype) {
    autocast_dtype[device_index(device_type)] = dtype;
}

int increment_nesting() {
    return ++nesting;
}

int decrement_nesting() {
    return --nesting;
}

bool is_autocast_cache_enabled() {
    return cache_enabled;
}

void set_autocast_cache_enabled(bool enabled) {
    cache_enabled = enabled;
}

void clear_cache() {
    const std::lock_guard<std::mutex> lock(cached_casts_mutex());
    get_cached_casts().clear();
}

Tensor cache_lookup(TensorImpl* key) {
    const std::lock_guard<std::mutex> lock(cached_casts_mutex());
    auto it = get_cached_casts().find(key);
    if (it != get_cached_casts().end()) {
        return std::get<1>(it->second);
    }
    return Tensor();
}

void cache_store(TensorImpl* key, const Tensor& source, const Tensor& casted) {
    const std::lock_guard<std::mutex> lock(cached_casts_mutex());
    get_cached_casts().emplace(
        key, val_type{std::weak_ptr<TensorImpl>(source.impl()), casted});
}

} // namespace autocast
} // namespace tensorplay
