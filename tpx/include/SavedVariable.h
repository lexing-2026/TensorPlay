#pragma once

#include "Macros.h"
#include "Tensor.h"

#include <cstdint>
#include <memory>

namespace tensorplay {
namespace tpx {

class TENSORPLAY_API SavedVariableHooks {
public:
    virtual ~SavedVariableHooks() = default;

    virtual std::shared_ptr<void> pack(const Tensor& tensor) = 0;
    virtual Tensor unpack(const std::shared_ptr<void>& packed) = 0;
};

TENSORPLAY_API std::shared_ptr<SavedVariableHooks> current_saved_variable_hooks();
TENSORPLAY_API void push_saved_variable_hooks(std::shared_ptr<SavedVariableHooks> hooks);
TENSORPLAY_API void pop_saved_variable_hooks();

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
        packed_.reset();
        hooks_.reset();
        saved_version_ = 0;
    }

    bool defined() const { return data_.defined() || hooks_ != nullptr; }

private:
    Tensor data_;
    std::shared_ptr<void> packed_;
    std::shared_ptr<SavedVariableHooks> hooks_;
    uint32_t saved_version_ = 0;
};

} // namespace tpx
} // namespace tensorplay
