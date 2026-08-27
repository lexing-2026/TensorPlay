#include "autocast_mode.h"

#include <vector>

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

// ------------------------------------------------------------------
// Cast cache: thread-local, lock-free on the hot path.
//
// torch keeps one process-global map behind a mutex that every eligible
// op takes twice (lookup + store); under intra-op workers this both
// serializes and cache-line-pings.  A thread-local map needs no locking
// at all and is cleared by the same thread's context-manager exit.
// Entries additionally record the fp32 source's version counter so an
// in-place mutation invalidates the cached copy -- which is what makes
// caching inference tensors safe (torch only ever caches requires_grad
// leaves).
// ------------------------------------------------------------------

namespace {

struct CacheEntry {
    std::weak_ptr<TensorImpl> source;
    Tensor casted;
    uint32_t version;
};

thread_local std::unordered_map<TensorImpl*, CacheEntry> t_cached_casts;

constexpr size_t kCacheSoftLimit = 4096;

void prune_expired(std::unordered_map<TensorImpl*, CacheEntry>& cache) {
    for (auto it = cache.begin(); it != cache.end();) {
        if (it->second.source.expired()) {
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

} // anonymous namespace
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
    t_cached_casts.clear();
}

Tensor cache_lookup(const TensorImpl* key) {
    auto& cache = t_cached_casts;
    auto it = cache.find(const_cast<TensorImpl*>(key));
    if (it == cache.end()) {
        return Tensor();
    }
    auto& entry = it->second;
    // Stale if the source died (recycled-pointer guard) or was mutated
    // in place since the cast was taken.
    if (entry.source.expired() || entry.version != key->version()) {
        cache.erase(it);
        return Tensor();
    }
    return entry.casted;
}

void cache_store(TensorImpl* key, const Tensor& source, const Tensor& casted) {
    auto& cache = t_cached_casts;
    if (cache.size() >= kCacheSoftLimit) {
        prune_expired(cache);
    }
    cache.insert_or_assign(
        key,
        CacheEntry{std::weak_ptr<TensorImpl>(source.impl()), casted,
                   source.unsafeGetTensorImpl()->version()});
}

namespace {
struct PtrEntry {
    Tensor casted;
    uint32_t version;
    std::vector<int64_t> sizes;
};
thread_local std::unordered_map<const void*, PtrEntry> t_ptr_cached_casts;
} // anonymous namespace

Tensor cache_lookup_ptr(const void* key, const Tensor& probe) {
    auto& cache = t_ptr_cached_casts;
    auto it = cache.find(key);
    if (it == cache.end()) return Tensor();
    auto& e = it->second;
    // Views share the parent's version counter, so an in-place mutation of
    // the parameter bumps `probe.version()` and drops the stale cast.
    if (e.version != probe.unsafeGetTensorImpl()->version() ||
        e.sizes != static_cast<std::vector<int64_t>>(probe.shape())) {
        cache.erase(it);
        return Tensor();
    }
    return e.casted;
}

void cache_store_ptr(const void* key, const Tensor& probe, const Tensor& casted) {
    auto& cache = t_ptr_cached_casts;
    if (cache.size() >= kCacheSoftLimit) {
        // Entries carry tensors; drop the whole table when it grows past the
        // soft limit (same bound-and-clear policy as the primitive cache).
        cache.clear();
    }
    cache.insert_or_assign(
        key, PtrEntry{casted, probe.unsafeGetTensorImpl()->version(),
                      static_cast<std::vector<int64_t>>(probe.shape())});
}

} // namespace autocast
} // namespace tensorplay
