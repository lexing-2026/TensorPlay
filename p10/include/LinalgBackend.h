#pragma once

// Selection knob for the library that executes dense linear-algebra
// factorizations on the GPU. This build ships a single native backend
// (cuSOLVER); the enum keeps the user-facing selection surface so scripts
// can pin the backend explicitly, and the "magma" slot exists only to
// reject the request with a clear message.

#include "Macros.h"

#include <string>

namespace tensorplay {

enum class P10_API LinalgBackend : int8_t { Default, Cusolver, Magma };

inline std::string LinalgBackendToString(LinalgBackend backend) {
    switch (backend) {
        case LinalgBackend::Default: return "default";
        case LinalgBackend::Cusolver: return "cusolver";
        case LinalgBackend::Magma: return "magma";
    }
    return "unknown";
}

}  // namespace tensorplay
