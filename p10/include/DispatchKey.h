#pragma once

#include <string>
#include <iostream>

namespace tensorplay {

enum class DispatchKey {
    CPU,
    CUDA,
    Autograd,
    // Add more keys as needed (e.g., AutogradCPU, AutogradCUDA)
    EndOfKeys // Sentinel
};

inline std::string toString(DispatchKey key) {
    switch (key) {
        case DispatchKey::CPU: return "CPU";
        case DispatchKey::CUDA: return "CUDA";
        case DispatchKey::Autograd: return "Autograd";
        default: return "Unknown";
    }
}

} // namespace tensorplay
