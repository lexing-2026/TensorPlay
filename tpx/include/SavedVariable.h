#pragma once

#include "Macros.h"
#include "Tensor.h"
#include "ForwardGrad.h"

#include <cstdint>
#include <memory>

namespace tensorplay {
namespace tpx {

class Node;

class TENSORPLAY_API SavedVariableHooks {
public:
    virtual ~SavedVariableHooks() = default;

    virtual std::shared_ptr<void> pack(const Tensor& tensor) = 0;
    virtual Tensor unpack(const std::shared_ptr<void>& packed) = 0;
};

TENSORPLAY_API std::shared_ptr<SavedVariableHooks> current_saved_variable_hooks();
TENSORPLAY_API void push_saved_variable_hooks(std::shared_ptr<SavedVariableHooks> hooks);
TENSORPLAY_API void pop_saved_variable_hooks();

// Error text for reading state that a backward pass already freed.
TENSORPLAY_API const char* backward_twice_message();

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

    // A node's own output is kept as a detached alias: the same data and
    // version counter, without the autograd metadata whose grad_fn is that
    // very node.  Holding the output itself would keep it, the node and the
    // graph behind it alive in a cycle once the caller drops the output
    // without running backward.
    SavedVariable(const Tensor& tensor, bool is_output) { save(tensor, is_output); }

    void save(const Tensor& tensor, bool is_output = false);

    // A saved output for the node that produced it.  A recorded backward
    // (create_graph) hands out an alias attached to `owner` at `output_nr`,
    // so a second derivative also flows through the output.
    Tensor unpack_output(const std::shared_ptr<Node>& owner, uint32_t output_nr) const;

    // Returns the saved tensor, or an undefined Tensor if nothing was saved.
    // Throws RuntimeError when the saved tensor was modified in-place after
    Tensor unpack() const;

    // Frees the stored tensor (Node::release_variables path); a later
    // unpack() of something that was saved raises, since the graph was
    // already walked without retain_graph.
    void reset_data() {
        data_ = Tensor();
        packed_.reset();
        hooks_.reset();
        fw_grad_.reset();
        saved_version_ = 0;
    }

    bool defined() const { return data_.defined() || hooks_ != nullptr; }

private:
    // `value` with the tangent a saved output had, when it had one.
    Tensor with_fw_grad(const Tensor& value) const;

    Tensor data_;
    // The tangent of a saved output, held where the level that owns it can
    // clear it.
    std::shared_ptr<ForwardGrad> fw_grad_;
    std::shared_ptr<void> packed_;
    std::shared_ptr<SavedVariableHooks> hooks_;
    uint32_t saved_version_ = 0;
    // True until a defined tensor is saved; only then does an empty slot
    // mean "freed" rather than "nothing was saved".
    bool was_default_constructed_ = true;
};

} // namespace tpx
} // namespace tensorplay
