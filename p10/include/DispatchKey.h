#pragma once

#include <cstdint>
#include <string>
#include <iostream>

namespace tensorplay {

// Dispatch keys used by the runtime.
//
// Backend keys occupy the low bits, autograd keys sit above them, and
// autocast keys sit above the autograd keys. Dispatch walks from the
// numerically largest key down, so the priority is Autocast > Autograd >
// backend.
enum class DispatchKey : uint8_t {
    // Backend component keys (dense backends).
    CPU = 0,
    CUDA = 1,

    // Autograd keys related to backends by a fixed offset.
    AutogradCPU = 2,
    AutogradCUDA = 3,

    // Autocast keys related to backends by a fixed offset.  They sit above
    // the autograd keys so casts happen before autograd history recording.
    AutocastCPU = 4,
    AutocastCUDA = 5,

    // Backend-neutral composite key. One registration serves every dense
    // backend until a backend registers its own kernel. Lookups never walk
    // this key from a tensor key set; the dispatcher consults it only when a
    // backend slot is empty.
    Composite = 6,

    EndOfKeys // Sentinel
};

constexpr size_t kBackendKeyCount = 2;           // CPU, CUDA
constexpr size_t kAutogradKeyOffset = 2;         // AutogradCPU - CPU
constexpr size_t kAutocastKeyOffset = 4;         // AutocastCPU - CPU

inline constexpr DispatchKey toAutocastKey(DispatchKey backend) {
    return static_cast<DispatchKey>(static_cast<uint8_t>(backend) + kAutocastKeyOffset);
}

inline constexpr DispatchKey toAutogradKey(DispatchKey backend) {
    return static_cast<DispatchKey>(static_cast<uint8_t>(backend) + kAutogradKeyOffset);
}

inline constexpr bool is_autocast_key(DispatchKey key) {
    // Autocast keys occupy [kAutocastKeyOffset, kAutocastKeyOffset + kBackendKeyCount).
    const uint8_t k = static_cast<uint8_t>(key);
    return k >= kAutocastKeyOffset && k < kAutocastKeyOffset + kBackendKeyCount;
}

inline constexpr bool is_autograd_key(DispatchKey key) {
    // Autograd keys occupy [kAutogradKeyOffset, kAutogradKeyOffset + kBackendKeyCount);
    // the Composite key sits above them and must not be classified as autograd.
    const uint8_t k = static_cast<uint8_t>(key);
    return k >= kAutogradKeyOffset && k < kAutogradKeyOffset + kBackendKeyCount;
}

// True for the dense backend component keys.
inline constexpr bool is_backend_key(DispatchKey key) {
    return key == DispatchKey::CPU || key == DispatchKey::CUDA;
}

// The backend component of an autocast or autograd key (identity for backend keys).
inline constexpr DispatchKey toBackendKey(DispatchKey key) {
    return is_autograd_key(key)
        ? static_cast<DispatchKey>(static_cast<uint8_t>(key) - kAutogradKeyOffset)
        : (is_autocast_key(key)
              ? static_cast<DispatchKey>(static_cast<uint8_t>(key) - kAutocastKeyOffset)
              : key);
}

inline std::string toString(DispatchKey key) {
    switch (key) {
        case DispatchKey::CPU: return "CPU";
        case DispatchKey::CUDA: return "CUDA";
        case DispatchKey::AutocastCPU: return "AutocastCPU";
        case DispatchKey::AutocastCUDA: return "AutocastCUDA";
        case DispatchKey::AutogradCPU: return "AutogradCPU";
        case DispatchKey::AutogradCUDA: return "AutogradCUDA";
        case DispatchKey::Composite: return "Composite";
        default: return "Unknown";
    }
}

// A small bitset over DispatchKey. Tensors
// carry one (TensorImpl::key_set_); dispatch walks it from highest-priority
// (autograd) to lowest (backend) bit.
class DispatchKeySet {
public:
    using raw_t = uint32_t;

    constexpr DispatchKeySet() noexcept : mask_(0) {}
    explicit constexpr DispatchKeySet(raw_t mask) noexcept : mask_(mask) {}

    static constexpr DispatchKeySet fromRaw(raw_t mask) { return DispatchKeySet(mask); }

    constexpr static DispatchKeySet make(DispatchKey key) {
        return DispatchKeySet(raw_t(1) << static_cast<uint8_t>(key));
    }

    void add(DispatchKey key) { mask_ |= (raw_t(1) << static_cast<uint8_t>(key)); }
    void remove(DispatchKey key) { mask_ &= ~(raw_t(1) << static_cast<uint8_t>(key)); }
    bool has(DispatchKey key) const {
        return mask_ & (raw_t(1) << static_cast<uint8_t>(key));
    }
    bool empty() const { return mask_ == 0; }
    raw_t raw() const { return mask_; }

    DispatchKeySet operator|(DispatchKeySet other) const { return DispatchKeySet(mask_ | other.mask_); }
    DispatchKeySet operator&(DispatchKeySet other) const { return DispatchKeySet(mask_ & other.mask_); }
    DispatchKeySet& operator|=(DispatchKeySet other) { mask_ |= other.mask_; return *this; }

    // Highest-priority (numerically largest) key in the set; EndOfKeys if empty.
    constexpr DispatchKey highest_priority_key() const {
        if (!mask_) return DispatchKey::EndOfKeys;
        uint8_t idx = 0;
        raw_t m = mask_;
        while (m >>= 1) ++idx;
        return static_cast<DispatchKey>(idx);
    }

    // Remove every autograd key (used for redispatch below the autograd layer,
    DispatchKeySet remove_autograd() const {
        raw_t autograd_mask = 0;
        for (size_t i = 0; i < kBackendKeyCount; ++i) {
            autograd_mask |= raw_t(1) << (i + kAutogradKeyOffset);
        }
        return DispatchKeySet(mask_ & ~autograd_mask);
    }

    // Remove every autocast key to re-enter dispatch below the autocast layer.
    DispatchKeySet remove_autocast() const {
        raw_t autocast_mask = 0;
        for (size_t i = 0; i < kBackendKeyCount; ++i) {
            autocast_mask |= raw_t(1) << (i + kAutocastKeyOffset);
        }
        return DispatchKeySet(mask_ & ~autocast_mask);
    }

private:
    raw_t mask_ = 0;
};

} // namespace tensorplay
