#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include <cctype>
#include "Macros.h"

namespace tensorplay {

// Define device types using macro for extensibility
#define TENSORPLAY_FORALL_DEVICE_TYPES(_) \
    _(CPU)                                \
    _(CUDA)                               \
    _(Unknown)

// Device types supported by TensorPlay
enum class DeviceType : int8_t {
    #define DEFINE_DEVICE_TYPE(name) name,
    TENSORPLAY_FORALL_DEVICE_TYPES(DEFINE_DEVICE_TYPE)
    #undef DEFINE_DEVICE_TYPE
};

// Device class to represent computation device
class P10_API Device {
private:
    DeviceType type_;
    int64_t index_;  // For multi-GPU systems, -1 for CPU

public:
    Device() : type_(DeviceType::CPU), index_(-1) {}
    Device(DeviceType type, int64_t index = -1) : type_(type), index_(index) {}
    Device(const std::string& device_str);
    Device(const std::string& type_str, int64_t index);
    
    // Expose type_ and index_ for external use
    DeviceType type() const { return type_; }
    int64_t index() const { return index_; }
       
    bool is_cpu() const { return type_ == DeviceType::CPU; }
    bool is_cuda() const { return type_ == DeviceType::CUDA; }
    
    // Convert device to string representation using macro-based mapping
    std::string toString() const {
        switch (type_) {
            #define CASE_DEVICE_TYPE(name) \
                case DeviceType::name: { \
                    std::string result = #name; \
                    /* Convert to lowercase */ \
                    for (auto& c : result) c = std::tolower(c); \
                    /* Handle device index for CUDA devices */ \
                    if ((DeviceType::name == DeviceType::CUDA || DeviceType::name == DeviceType::CPU) && index_ >= 0) { \
                        result += ":" + std::to_string(index_); \
                    } \
                    return result; \
                }
            TENSORPLAY_FORALL_DEVICE_TYPES(CASE_DEVICE_TYPE)
            #undef CASE_DEVICE_TYPE
            default:
                return "unknown";
        }
    }
    
    bool operator==(const Device& other) const {
        if (type_ != other.type_) return false;
        // Treat all CPU devices as the same, regardless of index
        if (type_ == DeviceType::CPU) return true;
        return index_ == other.index_;
    }
    
    bool operator!=(const Device& other) const {
        return !(*this == other);
    }
};

namespace cuda {
    P10_API size_t memory_allocated(int device = 0);
    P10_API size_t max_memory_allocated(int device = 0);
    P10_API void reset_max_memory_allocated(int device = 0);
    P10_API void empty_cache();
    P10_API void manual_seed(uint64_t seed);
}

} // namespace tensorplay
