#pragma once

#include <string>
#include <vector>

#include "DType.h"
#include "Exception.h"
#include "Tensor.h"

namespace tensorplay {

// Writes a computed result into the destination of an out= overload.
//
// The destination keeps its identity: it is resized only when its shape
// differs from the result, and the values are copied into its storage, so
// views of the destination observe the result.  The copy may cast, but only
// along safe casts, and never across devices.  An undefined destination
// adopts the result.
inline Tensor& write_out(Tensor& out, const Tensor& value) {
    if (!out.defined()) {
        out = value;
        return out;
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 std::string("Attempting to copy from device ") +
                     value.device().toString() + " to device " +
                     out.device().toString() +
                     ", but cross-device copies are not allowed!");
    }
    if (!canCast(value.dtype(), out.dtype())) {
        TP_THROW(TypeError,
                 std::string("Attempting to cast from ") +
                     toString(value.dtype()) + " to out tensor with dtype " +
                     toString(out.dtype()) +
                     ", but this can't be cast because it is not safe!");
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

}  // namespace tensorplay
