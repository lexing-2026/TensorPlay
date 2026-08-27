#pragma once
#include <memory>
#include <cstdint>
#include <optional>
#include <vector>
#include "Device.h"
#include "DType.h"
#include "Macros.h"

namespace tensorplay {
namespace tpx {
class Node;

struct TENSORPLAY_API Edge {
    std::shared_ptr<Node> function;
    uint32_t input_nr;
    // Shape of the forward input this edge was created from. The engine uses
    // it as torch's InputMetadata: gradients arriving with a different
    // (broadcast-inflated) shape are sum-reduced back to it before reaching
    // the consumer node.  `has_shape_hint` distinguishes a recorded scalar
    // shape () from "no hint" -- both are empty vectors.
    std::vector<int64_t> shape_hint;
    bool has_shape_hint = false;
    // Dtype of the forward input this edge was created from. The engine casts
    // incoming floating gradients to it (torch's InputMetadata::grad_dtype /
    // validate_outputs contract); this is what lets an fp32 gradient produced
    // by unwrapped promote ops re-enter autocast backward nodes whose saved
    // tensors are low precision.
    std::optional<DType> grad_dtype;
    // Device (type + index) of the forward input this edge was created from.
    // Together with shape/dtype hints this is the complete torch-style
    // InputMetadata triple, letting the ENGINE materialize missing gradients
    // (zeros) without crossing into Python.  The type is recorded separately
    // from the index because CPU tensors may carry index 0 (DLPack imports
    // report "cpu:0"), so an index alone cannot distinguish backends.
    std::optional<DeviceType> device_type_hint;
    std::optional<int64_t> device_index_hint;

    bool has_input_metadata() const {
        return has_shape_hint && grad_dtype.has_value()
            && device_type_hint.has_value()
            && device_index_hint.has_value();
    }

    Edge(std::shared_ptr<Node> function, uint32_t input_nr)
        : function(std::move(function)), input_nr(input_nr) {}
    Edge(std::shared_ptr<Node> function, uint32_t input_nr,
         std::vector<int64_t> shape_hint_)
        : function(std::move(function)), input_nr(input_nr),
          shape_hint(std::move(shape_hint_)), has_shape_hint(true) {}

    Edge() : function(nullptr), input_nr(0) {}

    bool is_valid() const { return function != nullptr; }
};
}
}
