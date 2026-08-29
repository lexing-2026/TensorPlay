#pragma once

// Human-readable formatting helpers for error messages.
//
// Goal: error text that is self-sufficient for diagnosis, e.g.
//   "shape=[2, 3], dtype=Float, device=cpu"
// instead of raw enum integers ("Found dtype 6 but expected 1").
//
// These are header-only so any kernel or iterator site can build richer

#include <sstream>
#include <string>
#include <vector>

#include "DType.h"
#include "LinearAlgebraNames.h"
#include "Tensor.h"

namespace tensorplay {

// "[2, 3]" style size list
inline std::string format_sizes(const std::vector<int64_t>& sizes) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0) os << ", ";
        os << sizes[i];
    }
    os << "]";
    return os.str();
}

// One-line summary of a tensor's metadata, safe on undefined tensors:
//   "shape=[2, 3], dtype=Float, device=cpu"
//   "<undefined tensor>"
inline std::string describe_tensor(const Tensor& t) {
    if (!t.defined()) {
        return "<undefined tensor>";
    }
    std::ostringstream os;
    os << "shape=" << format_sizes(static_cast<std::vector<int64_t>>(t.shape()))
       << ", dtype=" << pretty_dtype_name(t.dtype())
       << ", device=" << t.device().toString();
    return os.str();
}

//   "Dimension out of range (expected to be in range of [-2, 1], but got 5)"
inline std::string format_dim_range(int64_t ndim, int64_t dim) {
    std::ostringstream os;
    os << "Dimension out of range (expected to be in range of [" << -ndim
       << ", " << ndim - 1 << "], but got " << dim << ")";
    return os.str();
}

} // namespace tensorplay
