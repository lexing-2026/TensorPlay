#pragma once

#include <cstdint>
#include "Macros.h"

namespace tensorplay {

// Version counter for tracking tensor modifications
class P10_API VariableVersion {
private:
    uint32_t version_;
    bool enabled_;
    
public:
    VariableVersion() : version_(0), enabled_(true) {}
    
    explicit VariableVersion(bool enabled) : version_(0), enabled_(enabled) {}
    
    // Get current version
    uint32_t current_version() const { return version_; }
    
    // Check if version tracking is enabled
    bool is_enabled() const { return enabled_; }
    
    // Increment version
    void bump() {
        if (enabled_) {
            ++version_;
        }
    }

    // Reset version
    void reset() {
        if (enabled_) {
            version_ = 0;
        }
    }
    
    // Equality operator
    bool operator==(const VariableVersion& other) const {
        return version_ == other.version_ && enabled_ == other.enabled_;
    }
    
    bool operator!=(const VariableVersion& other) const {
        return !(*this == other);
    }
};

} // namespace tensorplay
