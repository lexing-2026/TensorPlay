#pragma once

// Selection knob for the BLAS library that executes matrix products on
// the GPU: the classic cuBLAS API or the cuBLASLt API. "Default" defers
// to the per-call heuristics in the GEMM dispatcher.

#include "Macros.h"

#include <string>

namespace tensorplay {

enum class P10_API BlasBackend : int8_t { Default, Cublas, Cublaslt };

inline std::string BlasBackendToString(BlasBackend backend) {
    switch (backend) {
        case BlasBackend::Default: return "default";
        case BlasBackend::Cublas: return "cublas";
        case BlasBackend::Cublaslt: return "cublaslt";
    }
    return "unknown";
}

}  // namespace tensorplay
