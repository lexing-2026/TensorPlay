#include "Device.h"
#include "Exception.h"
#include <algorithm>
#include <string>

namespace tensorplay {

Device::Device(const std::string& device_str) : type_(DeviceType::CPU), index_(-1) {
    std::string s = device_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });

    if (s.rfind("cpu", 0) == 0) {
        type_ = DeviceType::CPU;
        size_t colon = s.find(':');
        if (colon != std::string::npos) {
            try {
                // Let's store it as parsed for now.
                index_ = std::stoi(s.substr(colon + 1));
            } catch (...) {
                TP_THROW(ValueError, "Invalid device string: " + device_str);
            }
        }
    } else if (s.rfind("cuda", 0) == 0) {
        type_ = DeviceType::CUDA;
        size_t colon = s.find(':');
        if (colon != std::string::npos) {
             try {
                index_ = std::stoi(s.substr(colon + 1));
            } catch (...) {
                TP_THROW(ValueError, "Invalid device string: " + device_str);
            }
        } else {
            // our Device class defaults index to -1.
            // If we want to support default device, -1 is correct.
            // However, to avoid mismatch with explicit cuda:0, we default to 0 here.
            index_ = 0; 
        }
    } else {
        TP_THROW(ValueError, "Invalid device string: " + device_str);
    }
}

Device::Device(const std::string& type_str, int64_t index) : index_(index) {
    std::string s = type_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    
    if (s == "cpu") {
        type_ = DeviceType::CPU;
    } else if (s == "cuda") {
        type_ = DeviceType::CUDA;
    } else {
        TP_THROW(ValueError, "Invalid device type: " + type_str);
    }
}

} // namespace tensorplay
