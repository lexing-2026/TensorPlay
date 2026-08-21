#pragma once

#include <cstdint>
#include <memory>
#include <atomic>
#include "Macros.h"

namespace tensorplay {

// The actual counter, shared by every tensor that aliases the same memory
// (base + views), mirroring c10::VersionCounter. Atomic so that bumps from
// one alias are visible to readers holding another.
struct P10_API VersionCounter {
    std::atomic<uint32_t> version_{0};

    void bump() noexcept { version_.fetch_add(1, std::memory_order_relaxed); }
    uint32_t current_version() const noexcept { return version_.load(std::memory_order_relaxed); }
};

// Handle to a VersionCounter. Mirrors c10::VariableVersion: the counter is
// allocated eagerly so that views created before ANY mutation share the same
// counter object as their base (lazy allocation would let each alias grow its
// own counter, silently breaking mutation tracking).
class P10_API VariableVersion {
private:
    std::shared_ptr<VersionCounter> counter_;
    bool enabled_ = true;

public:
    VariableVersion() : counter_(std::make_shared<VersionCounter>()) {}

    explicit VariableVersion(bool enabled)
        : counter_(enabled ? std::make_shared<VersionCounter>() : nullptr), enabled_(enabled) {}

    // Get current version
    uint32_t current_version() const {
        return counter_ ? counter_->current_version() : 0;
    }

    // Check if version tracking is enabled
    bool is_enabled() const { return enabled_; }

    // Increment version.
    void bump() {
        if (!enabled_) return;
        counter_->bump();
    }

    // Reset version
    void reset() {
        if (counter_) counter_->version_.store(0, std::memory_order_relaxed);
    }

    // Whether this handle and `other` track the same counter (i.e. the two
    // tensors alias each other's mutation state).
    bool shares_counter(const VariableVersion& other) const {
        return counter_ && counter_ == other.counter_;
    }

    // Equality operator
    bool operator==(const VariableVersion& other) const {
        return enabled_ == other.enabled_ && current_version() == other.current_version();
    }

    bool operator!=(const VariableVersion& other) const {
        return !(*this == other);
    }
};

} // namespace tensorplay
