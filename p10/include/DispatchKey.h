#pragma once

#include <cstdint>
#include <string>
#include <iostream>

namespace tensorplay {

// Dispatch keys used by the runtime.
//
// Backend keys occupy the low bits; the Python key sits directly above them,
// autograd keys above that, and autocast keys above the autograd keys.
// Dispatch walks from the numerically largest key down, so the priority is
// Autocast > Autograd > Python > backend. Sparse layouts form their own
// backend component: a tensor whose storage is sparse metadata carries the
// Sparse key instead of the dense backend key of its device.
enum class DispatchKey : uint8_t {
    // Backend component keys (dense backends plus the sparse layout family).
    CPU = 0,
    CUDA = 1,
    Vulkan = 2,
    Sparse = 3,

    // Python dispatch modes.  Included in the thread-local key set while a
    // dispatch mode is active: every operator that reaches its backend --
    // including the ones autograd runs during backward -- is handed to the
    // innermost mode first.  Every key above it is excluded while a mode
    // handler runs, so re-entering an operator from the handler neither
    // records autograd history nor re-applies autocast or batching.
    Python = 4,

    // Autograd keys related to backends by a fixed offset.
    DynamicLayerBackMode = 5,
    AutogradCPU = 6,
    AutogradCUDA = 7,
    AutogradVulkan = 8,
    AutogradSparse = 9,

    // Autocast keys related to backends by a fixed offset.  They sit above
    // the autograd keys so casts happen before autograd history recording.
    AutocastCPU = 10,
    AutocastCUDA = 11,
    AutocastVulkan = 12,
    AutocastSparse = 13,

    // Backend-neutral composite key. One registration serves every backend
    // until a backend registers its own kernel. Lookups never walk this key
    // from a tensor key set; the dispatcher consults it only when a backend
    // slot is empty.
    Composite = 21,

    // Per-backend batching keys. These must outrank autograd and backend
    // keys so a transform can unwrap its operands before ordinary kernels
    // and autograd nodes observe them.
    VmapCPU = 14,
    VmapCUDA = 15,
    VmapVulkan = 16,
    VmapSparse = 17,
    VmapMode = 22,
    DynamicLayerFrontMode = 23,

    // One past every real key; the sentinel value must stay above all of
    // them, so it is spelled out rather than derived from the previous
    // entry.
    EndOfKeys = 24 // Sentinel
};

constexpr size_t kBackendKeyCount = 4;           // CPU, CUDA, Vulkan, Sparse
constexpr size_t kAutogradKeyOffset = 6;         // AutogradCPU - CPU
constexpr size_t kAutocastKeyOffset = 10;        // AutocastCPU - CPU
constexpr size_t kVmapKeyOffset = 14;            // VmapCPU - CPU

static_assert(static_cast<size_t>(DispatchKey::AutogradCPU) ==
                  static_cast<size_t>(DispatchKey::CPU) + kAutogradKeyOffset,
              "autograd key offset out of sync with the key layout");
static_assert(static_cast<size_t>(DispatchKey::AutocastCPU) ==
                  static_cast<size_t>(DispatchKey::CPU) + kAutocastKeyOffset,
              "autocast key offset out of sync with the key layout");
static_assert(static_cast<size_t>(DispatchKey::VmapCPU) ==
                  static_cast<size_t>(DispatchKey::CPU) + kVmapKeyOffset,
              "vmap key offset out of sync with the key layout");
static_assert(static_cast<size_t>(DispatchKey::Python) > static_cast<size_t>(DispatchKey::Sparse) &&
                  static_cast<size_t>(DispatchKey::Python) < static_cast<size_t>(DispatchKey::DynamicLayerBackMode),
              "the Python key must sit between the backends and every transform/autograd layer");

inline constexpr DispatchKey toAutocastKey(DispatchKey backend) {
    return static_cast<DispatchKey>(static_cast<uint8_t>(backend) + kAutocastKeyOffset);
}

inline constexpr DispatchKey toAutogradKey(DispatchKey backend) {
    return static_cast<DispatchKey>(static_cast<uint8_t>(backend) + kAutogradKeyOffset);
}

inline constexpr DispatchKey toVmapKey(DispatchKey backend) {
    return static_cast<DispatchKey>(static_cast<uint8_t>(backend) + kVmapKeyOffset);
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

inline constexpr bool is_vmap_key(DispatchKey key) {
    const uint8_t k = static_cast<uint8_t>(key);
    return k >= kVmapKeyOffset && k < kVmapKeyOffset + kBackendKeyCount;
}

// True for the backend component keys (dense backends and the sparse
// layout family).
inline constexpr bool is_backend_key(DispatchKey key) {
    return key == DispatchKey::CPU || key == DispatchKey::CUDA ||
           key == DispatchKey::Vulkan || key == DispatchKey::Sparse;
}

// The backend component of an autocast or autograd key (identity for backend keys).
inline constexpr DispatchKey toBackendKey(DispatchKey key) {
    return is_vmap_key(key)
        ? static_cast<DispatchKey>(static_cast<uint8_t>(key) - kVmapKeyOffset)
        : (is_autograd_key(key)
        ? static_cast<DispatchKey>(static_cast<uint8_t>(key) - kAutogradKeyOffset)
        : (is_autocast_key(key)
              ? static_cast<DispatchKey>(static_cast<uint8_t>(key) - kAutocastKeyOffset)
              : key));
}

inline std::string toString(DispatchKey key) {
    switch (key) {
        case DispatchKey::CPU: return "CPU";
        case DispatchKey::CUDA: return "CUDA";
        case DispatchKey::Vulkan: return "Vulkan";
        case DispatchKey::Sparse: return "Sparse";
        case DispatchKey::AutocastCPU: return "AutocastCPU";
        case DispatchKey::AutocastCUDA: return "AutocastCUDA";
        case DispatchKey::AutocastVulkan: return "AutocastVulkan";
        case DispatchKey::AutocastSparse: return "AutocastSparse";
        case DispatchKey::AutogradCPU: return "AutogradCPU";
        case DispatchKey::AutogradCUDA: return "AutogradCUDA";
        case DispatchKey::AutogradVulkan: return "AutogradVulkan";
        case DispatchKey::AutogradSparse: return "AutogradSparse";
        case DispatchKey::VmapCPU: return "VmapCPU";
        case DispatchKey::VmapCUDA: return "VmapCUDA";
        case DispatchKey::VmapVulkan: return "VmapVulkan";
        case DispatchKey::VmapSparse: return "VmapSparse";
        case DispatchKey::Composite: return "Composite";
        case DispatchKey::VmapMode: return "VmapMode";
        case DispatchKey::DynamicLayerFrontMode: return "DynamicLayerFrontMode";
        case DispatchKey::DynamicLayerBackMode: return "DynamicLayerBackMode";
        case DispatchKey::Python: return "Python";
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
    DispatchKeySet operator-(DispatchKeySet other) const { return DispatchKeySet(mask_ & ~other.mask_); }
    DispatchKeySet& operator|=(DispatchKeySet other) { mask_ |= other.mask_; return *this; }

    // Highest-priority (numerically largest) key in the set; EndOfKeys if empty.
    constexpr DispatchKey highest_priority_key() const {
        if (!mask_) return DispatchKey::EndOfKeys;
        uint8_t idx = 0;
        raw_t m = mask_;
        while (m >>= 1) ++idx;
        return static_cast<DispatchKey>(idx);
    }

    // Every key numerically above ``key`` -- the layers that have already
    // run by the time dispatch reaches ``key``.
    static constexpr DispatchKeySet above(DispatchKey key) {
        const raw_t end = raw_t(1) << static_cast<uint8_t>(DispatchKey::EndOfKeys);
        const raw_t upto = raw_t(1) << (static_cast<uint8_t>(key) + 1);
        return DispatchKeySet((end - 1) & ~(upto - 1));
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

    DispatchKeySet remove_vmap() const {
        raw_t vmap_mask = 0;
        for (size_t i = 0; i < kBackendKeyCount; ++i) {
            vmap_mask |= raw_t(1) << (i + kVmapKeyOffset);
        }
        return DispatchKeySet(mask_ & ~vmap_mask);
    }

private:
    raw_t mask_ = 0;
};

} // namespace tensorplay
