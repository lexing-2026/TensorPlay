// Rank-generic padding dispatcher.
//
// `pad` and `_pad_enum` are the backend-neutral entry points that fan a
// padding request out to the per-mode kernels (constant / reflect /
// replicate / circular).  They own no padding math themselves: the mode is
// resolved once here and each branch forwards to the operator that owns the
// semantics for that mode.  The per-mode kernels handle the actual edge
// addressing, so every backend those ops support is available here for free.
//
// Mode numbering follows the padding-mode enum: reflect, replicate,
// circular, constant.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

constexpr int kModeReflect = 0;
constexpr int kModeReplicate = 1;
constexpr int kModeCircular = 2;
constexpr int kModeConstant = 3;

std::string_view padding_mode_string(int mode) {
    switch (mode) {
        case kModeReflect: return "reflect";
        case kModeReplicate: return "replicate";
        case kModeCircular: return "circular";
        case kModeConstant: return "constant";
        default: return "unknown";
    }
}

std::string unsupported_padding_error(int64_t pad_size, int64_t input_dim) {
    std::ostringstream msg;
    msg << "Padding size " << pad_size << " is not supported for " << input_dim
        << "D input tensor.\n";
    msg << "Supported combinations for non-constant padding:\n";
    msg << "  - 2D or 3D input: padding size = 2 (pads last dimension)\n";
    msg << "  - 3D or 4D input: padding size = 4 (pads last 2 dimensions)\n";
    msg << "  - 4D or 5D input: padding size = 6 (pads last 3 dimensions)";
    return msg.str();
}

// Dispatch a non-constant padding request onto the rank-specific kernels.
// Each supported (padding length, input rank) pair maps to exactly one
// rank-generic kernel; anything else is rejected with the layout matrix.
Tensor pad_by_mode(const Tensor& self, const std::vector<int64_t>& pad, int mode) {
    const int64_t input_dim = self.dim();
    if (pad.size() == 2 && (input_dim == 2 || input_dim == 3)) {
        switch (mode) {
            case kModeReflect: return ops::reflection_pad1d(self, pad);
            case kModeReplicate: return ops::replication_pad1d(self, pad);
            case kModeCircular: return ops::_pad_circular(self, pad);
            default: break;
        }
    } else if (pad.size() == 4 && (input_dim == 3 || input_dim == 4)) {
        switch (mode) {
            case kModeReflect: return ops::reflection_pad2d(self, pad);
            case kModeReplicate: return ops::replication_pad2d(self, pad);
            case kModeCircular: return ops::_pad_circular(self, pad);
            default: break;
        }
    } else if (pad.size() == 6 && (input_dim == 4 || input_dim == 5)) {
        switch (mode) {
            case kModeReflect: return ops::reflection_pad3d(self, pad);
            case kModeReplicate: return ops::replication_pad3d(self, pad);
            case kModeCircular: return ops::_pad_circular(self, pad);
            default: break;
        }
    }
    TP_THROW(NotImplementedError, unsupported_padding_error(
        static_cast<int64_t>(pad.size()), input_dim));
}

}  // namespace

Tensor _pad_circular_native(const Tensor& self, const std::vector<int64_t>& pad) {
    return ops::circular_pad_nd(self, pad);
}


Tensor _pad_enum_native(const Tensor& self, const std::vector<int64_t>& pad,
                        int64_t mode, std::optional<double> value) {
    const int64_t input_dim = self.dim();
    if (pad.size() % 2 != 0) {
        TP_THROW(RuntimeError, "Padding length must be divisible by 2");
    }
    if (static_cast<int64_t>(pad.size()) > input_dim * 2) {
        TP_THROW(RuntimeError,
                 "Padding length should be less than or equal to two times the "
                 "input dimension but got padding length ", pad.size(),
                 " and input of dimension ", input_dim);
    }
    if (mode == kModeConstant) {
        return ops::constant_pad_nd(self, pad, value.value_or(0.0));
    }
    if (value.has_value() && *value != 0) {
        TP_THROW(RuntimeError, "Padding mode \"", padding_mode_string(static_cast<int>(mode)),
                 "\" doesn't take in value argument");
    }
    return pad_by_mode(self, pad, static_cast<int>(mode));
}

Tensor pad_native(const Tensor& self, const std::vector<int64_t>& pad,
                  const std::string& mode, std::optional<double> value) {
    int resolved;
    if (mode == "reflect") {
        resolved = kModeReflect;
    } else if (mode == "constant") {
        resolved = kModeConstant;
    } else if (mode == "replicate") {
        resolved = kModeReplicate;
    } else if (mode == "circular") {
        resolved = kModeCircular;
    } else {
        TP_THROW(NotImplementedError,
                 std::string("Unrecognised padding mode ") + mode);
    }
    return _pad_enum_native(self, pad, resolved, value);
}

}  // namespace composite
}  // namespace tensorplay

namespace tensorplay {

TENSORPLAY_LIBRARY_IMPL(Composite, PadNdComposite) {
    m.impl("_pad_circular", composite::_pad_circular_native);
    m.impl("_pad_enum", composite::_pad_enum_native);
    m.impl("pad", composite::pad_native);
}

}  // namespace tensorplay
