#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include <unordered_map>
#include <cctype>
#include "Macros.h"

namespace tensorplay {

// Define device types using macro for extensibility
#define TENSORPLAY_FORALL_DEVICE_TYPES(_) \
    _(CPU)                                \
    _(CUDA)                               \
    _(HIP)                                \
    _(MPS)                                \
    _(MTIA)                               \
    _(XPU)                                \
    _(HPU)                                \
    _(PrivateUse1)                        \
    _(Vulkan)                             \
    _(Meta)                               \
    _(Unknown)

// Device types supported by TensorPlay
enum class DeviceType : int8_t {
    #define DEFINE_DEVICE_TYPE(name) name,
    TENSORPLAY_FORALL_DEVICE_TYPES(DEFINE_DEVICE_TYPE)
    #undef DEFINE_DEVICE_TYPE
};

constexpr size_t kDeviceTypeCount = static_cast<size_t>(DeviceType::Unknown) + 1;

// The out-of-tree device slot.  Until a backend names it, the slot is
// spelled "privateuseone"; an extension may register one name of its own
// (e.g. "npu"), once per process, after which device strings, printing and
// the Python ``device.type`` all use that name.  The default spelling keeps
// parsing either way.
P10_API void register_privateuse1_backend(const std::string& backend_name);
P10_API bool is_privateuse1_backend_registered();
P10_API std::string get_privateuse1_backend(bool lower_case = true);

// Lower-case spelling of a device type ("cuda", "mps", or the registered
// out-of-tree name).
P10_API std::string device_type_name(DeviceType type);
// Device type for a lower-case spelling; throws ValueError when unknown.
P10_API DeviceType parse_device_type(const std::string& name);

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
    bool is_vulkan() const { return type_ == DeviceType::Vulkan; }
    // The meta device models shape and dtype only; it never owns or aliases
    // real memory, so its storage hands out a null data pointer.
    bool is_meta() const { return type_ == DeviceType::Meta; }

    // The accelerator slots use the same ":index" suffix as CUDA, so a
    // parsed "mps:0" round-trips to the same spelling.
    bool is_mps() const { return type_ == DeviceType::MPS; }

    bool is_privateuse1() const { return type_ == DeviceType::PrivateUse1; }

    // "cpu", "cuda:1", "mps:0", or the registered out-of-tree name with its
    // index.  Indexed devices print their index when one is set.
    std::string toString() const;

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
    P10_API size_t memory_reserved(int device = 0);
    P10_API size_t max_memory_allocated(int device = 0);
    P10_API size_t max_memory_reserved(int device = 0);
    P10_API void reset_peak_memory_stats(int device = 0);
    P10_API void empty_cache();
    // Fragmentation-aware accounting: allocated/reserved/peaks plus segment
    // counts, free-block bytes (largest block), event-pending blocks and the
    // number of live graph pools.
    P10_API std::unordered_map<std::string, uint64_t> memory_stats(
        int device = -1);
    P10_API void manual_seed(uint64_t seed);
    P10_API void manual_seed_all(uint64_t seed);
    P10_API void recordStream(void* base_ptr, const Device& device);
}

} // namespace tensorplay
