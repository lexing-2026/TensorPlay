#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "Device.h"
#include "DispatchKey.h"
#include "DType.h"
#include "Macros.h"
#include "Tensor.h"

namespace tensorplay {

// Autocast state and helpers, mirroring ATen/autocast_mode.{h,cpp}.
//
// PyTorch implements the enable/disable switch through the dispatcher's TLS
// excluded set: every Autocast* key lives in `default_excluded_set`, so
// `set_autocast_enabled(device_type, true)` merely removes the key from that
// set and lets the dispatcher consult Autocast kernels.  TensorPlay's
// dispatcher consults an explicit key per call site instead of a TLS keyset,
// so the same contract is provided by a thread-local enabled flag per device;
// call sites check `is_enabled(key)` before routing to the Autocast key.
//
// The cast-cache mirrors Apex/torch: the key is the fp32 source tensor's
// TensorImpl*, kept alive by a weak reference so a recycled pointer can never
// falsely hit.  `clear_cache()` is called by the Python context manager when
// the nesting level drops to zero.
namespace autocast {

// ------------------------------------------------------------------
// Device/key helpers (mirrors at::autocast inline helpers)
// ------------------------------------------------------------------

inline constexpr DispatchKey get_autocast_dispatch_key_from_device_type(
    DeviceType device_type) {
    switch (device_type) {
        case DeviceType::CPU:
            return DispatchKey::AutocastCPU;
        case DeviceType::CUDA:
            return DispatchKey::AutocastCUDA;
        default:
            TP_THROW(NotImplementedError,
                "unknown device type for autocast in get_autocast_dispatch_key_from_device_type");
    }
}

inline constexpr DeviceType get_device_type_from_autocast_key(DispatchKey key) {
    switch (key) {
        case DispatchKey::AutocastCPU:
            return DeviceType::CPU;
        case DispatchKey::AutocastCUDA:
            return DeviceType::CUDA;
        default:
            TP_THROW(NotImplementedError,
                "unknown autocast dispatch key in get_device_type_from_autocast_key");
    }
}

constexpr std::array<DeviceType, 2> _AUTOCAST_SUPPORTED_DEVICES{
    DeviceType::CPU,
    DeviceType::CUDA};

inline bool is_autocast_available(DeviceType device_type) {
    for (const auto& supported : _AUTOCAST_SUPPORTED_DEVICES) {
        if (supported == device_type) return true;
    }
    return false;
}

// ------------------------------------------------------------------
// Thread-local state (mirrors at::autocast state in autocast_mode.cpp)
// ------------------------------------------------------------------

P10_API bool is_enabled(DispatchKey autocast_key);
inline bool is_autocast_enabled(DeviceType device_type) {
    return is_enabled(get_autocast_dispatch_key_from_device_type(device_type));
}
P10_API void set_enabled(DispatchKey autocast_key, bool enabled);
inline void set_autocast_enabled(DeviceType device_type, bool enabled) {
    set_enabled(get_autocast_dispatch_key_from_device_type(device_type), enabled);
}

P10_API DType get_autocast_dtype(DeviceType device_type);
P10_API void set_autocast_dtype(DeviceType device_type, DType dtype);

P10_API int increment_nesting();
P10_API int decrement_nesting();

P10_API bool is_autocast_cache_enabled();
P10_API void set_autocast_cache_enabled(bool enabled);

P10_API void clear_cache();

// Mirrors c10::impl::ExcludeDispatchKeyGuard for the Autocast keys: while
// alive, autocast is disabled for the given device so nested dispatch from an
// autocast kernel cannot recurse.
class ExcludeAutocastGuard {
public:
    explicit ExcludeAutocastGuard(DeviceType device_type)
        : key_(get_autocast_dispatch_key_from_device_type(device_type)),
          prev_(is_enabled(key_)) {
        set_enabled(key_, false);
    }

    ~ExcludeAutocastGuard() { set_enabled(key_, prev_); }

    ExcludeAutocastGuard(const ExcludeAutocastGuard&) = delete;
    ExcludeAutocastGuard& operator=(const ExcludeAutocastGuard&) = delete;

private:
    DispatchKey key_;
    bool prev_;
};

// ------------------------------------------------------------------
// Cast cache storage (populated by tpx's cached_cast)
// ------------------------------------------------------------------

using val_type = std::pair<std::weak_ptr<TensorImpl>, Tensor>;

P10_API Tensor cache_lookup(TensorImpl* key);
P10_API void cache_store(TensorImpl* key, const Tensor& source, const Tensor& casted);

} // namespace autocast
} // namespace tensorplay
