#pragma once

#include "Macros.h"
#include "Tensor.h"

#include <cstdint>

namespace tensorplay {
namespace tpx {

// Saved forward tensor of a backward node, preserving
// save time; unpack() fails loudly if the tensor (or a view base sharing its
// counter) was mutated in-place between the forward and the backward, instead
// of silently producing wrong gradients.
class TENSORPLAY_API SavedVariable {
public:
    SavedVariable() = default;

    // Implicit on purpose: generated node constructors take plain Tensors and
    // store them directly (`self_(self)`).
    SavedVariable(const Tensor& tensor) { save(tensor); }

    void save(const Tensor& tensor);

    // Returns the saved tensor, or an undefined Tensor if nothing was saved.
    // Throws RuntimeError when the saved tensor was modified in-place after
    Tensor unpack() const;

    // Frees the stored tensor (Node::release_variables path); a later
    // unpack() returns an undefined Tensor rather than stale memory.
    void reset_data() {
        data_ = Tensor();
        saved_version_ = 0;
    }

    bool defined() const { return data_.defined(); }

private:
    Tensor data_;
    uint32_t saved_version_ = 0;
};

} // namespace tpx
} // namespace tensorplay
