#pragma once

#include <cstdint>
#include <string>
#include <iostream>

namespace tensorplay {

// Backend components, in dispatch-key order.  Every layered key family
// (autograd, autocast, vmap) holds one key per component, laid out in this
// same order, so a backend added here receives all of its layers at once.
//
// PrivateUse1 is the slot reserved for a backend that lives outside this
// repository: an extension registers kernels, an allocator and device hooks
// for it at load time and may give the device a name of its own (see
// Device.h).
#define TENSORPLAY_FORALL_BACKEND_COMPONENTS(_) \
    _(CPU)                                      \
    _(CUDA)                                     \
    _(Vulkan)                                   \
    _(SparseCPU)                                \
    _(SparseCUDA)                               \
    _(Sparse)                                   \
    _(Meta)                                     \
    _(MPS)                                      \
    _(PrivateUse1)

// Dispatch keys used by the runtime.
//
// Backend keys occupy the low bits; the Python key sits directly above them,
// autograd keys above that, then autocast keys and vmap keys.  Dispatch
// walks from the numerically largest key down, so the priority is
// Vmap > Autocast > Autograd > Python > backend.  Sparse layouts form their
// own backend component: a tensor whose storage is sparse metadata carries
// the Sparse key instead of the dense backend key of its device.
enum class DispatchKey : uint8_t {
#define TP_DISPATCH_BACKEND_KEY(name) name,
    TENSORPLAY_FORALL_BACKEND_COMPONENTS(TP_DISPATCH_BACKEND_KEY)
#undef TP_DISPATCH_BACKEND_KEY
    Python,
    DynamicLayerBackMode,
#define TP_DISPATCH_AUTOGRAD_KEY(name) Autograd##name,
    TENSORPLAY_FORALL_BACKEND_COMPONENTS(TP_DISPATCH_AUTOGRAD_KEY)
#undef TP_DISPATCH_AUTOGRAD_KEY
#define TP_DISPATCH_AUTOCAST_KEY(name) Autocast##name,
    TENSORPLAY_FORALL_BACKEND_COMPONENTS(TP_DISPATCH_AUTOCAST_KEY)
#undef TP_DISPATCH_AUTOCAST_KEY
#define TP_DISPATCH_VMAP_KEY(name) Vmap##name,
    TENSORPLAY_FORALL_BACKEND_COMPONENTS(TP_DISPATCH_VMAP_KEY)
#undef TP_DISPATCH_VMAP_KEY
    Composite,
    VmapMode,
    DynamicLayerFrontMode,
    EndOfKeys
};

constexpr size_t kBackendKeyCount =
    static_cast<size_t>(DispatchKey::Python);
constexpr size_t kAutogradKeyOffset =
    static_cast<size_t>(DispatchKey::AutogradCPU);
constexpr size_t kAutocastKeyOffset =
    static_cast<size_t>(DispatchKey::AutocastCPU);
constexpr size_t kVmapKeyOffset =
    static_cast<size_t>(DispatchKey::VmapCPU);

static_assert(static_cast<size_t>(DispatchKey::EndOfKeys) <= 64,
              "DispatchKeySet holds one bit per key in a 64-bit mask");
static_assert(kAutocastKeyOffset == kAutogradKeyOffset + kBackendKeyCount &&
                  kVmapKeyOffset == kAutocastKeyOffset + kBackendKeyCount &&
                  static_cast<size_t>(DispatchKey::Composite) ==
                      kVmapKeyOffset + kBackendKeyCount,
              "each layered key family must hold exactly one key per backend");
static_assert(static_cast<size_t>(DispatchKey::Python) == kBackendKeyCount &&
                  static_cast<size_t>(DispatchKey::Python) <
                      static_cast<size_t>(DispatchKey::DynamicLayerBackMode),
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
    const uint8_t k = static_cast<uint8_t>(key);
    return k >= kAutocastKeyOffset && k < kAutocastKeyOffset + kBackendKeyCount;
}

inline constexpr bool is_autograd_key(DispatchKey key) {
    // The Composite key sits above every layered family and is not
    // classified as autograd.
    const uint8_t k = static_cast<uint8_t>(key);
    return k >= kAutogradKeyOffset && k < kAutogradKeyOffset + kBackendKeyCount;
}

inline constexpr bool is_vmap_key(DispatchKey key) {
    const uint8_t k = static_cast<uint8_t>(key);
    return k >= kVmapKeyOffset && k < kVmapKeyOffset + kBackendKeyCount;
}

// True for the backend component keys (dense backends, the sparse layout
// family and the out-of-tree slot).
inline constexpr bool is_backend_key(DispatchKey key) {
    return static_cast<size_t>(key) < kBackendKeyCount;
}

// The backend component of an autocast, autograd or vmap key (identity for
// backend keys).
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
#define TP_DISPATCH_KEY_NAMES(name)                                   \
        case DispatchKey::name: return #name;                         \
        case DispatchKey::Autograd##name: return "Autograd" #name;    \
        case DispatchKey::Autocast##name: return "Autocast" #name;    \
        case DispatchKey::Vmap##name: return "Vmap" #name;
        TENSORPLAY_FORALL_BACKEND_COMPONENTS(TP_DISPATCH_KEY_NAMES)
#undef TP_DISPATCH_KEY_NAMES
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
    using raw_t = uint64_t;

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

    // One family of layered keys, every backend included.
    static constexpr DispatchKeySet autograd_keys() {
        return family(kAutogradKeyOffset);
    }
    static constexpr DispatchKeySet autocast_keys() {
        return family(kAutocastKeyOffset);
    }
    static constexpr DispatchKeySet vmap_keys() {
        return family(kVmapKeyOffset);
    }
    static constexpr DispatchKeySet backend_keys() {
        return family(0);
    }

private:
    static constexpr DispatchKeySet family(size_t offset) {
        raw_t mask = 0;
        for (size_t i = 0; i < kBackendKeyCount; ++i) {
            mask |= raw_t(1) << (i + offset);
        }
        return DispatchKeySet(mask);
    }

    raw_t mask_ = 0;
};

} // namespace tensorplay
