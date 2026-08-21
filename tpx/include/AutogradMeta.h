#pragma once

#include "AutogradMetaBase.h"
#include "Tensor.h"
#include "Macros.h"
#include <memory>
#include <cstdint>

namespace tensorplay {
namespace tpx {

class Node;

// Concrete autograd metadata, attached to a p10 TensorImpl through the
// AutogradMetaBase extension point. Mirrors torch::autograd::AutogradMeta.
class TENSORPLAY_API AutogradMeta : public AutogradMetaBase {
private:
    bool requires_grad_ = false;
    bool retains_grad_ = false;
    tensorplay::Tensor grad_;
    std::shared_ptr<Node> grad_fn_;
    // NB: weak reference, mirroring c10::AutogradMeta::grad_accumulator_
    // (weak_intrusive_ptr). The AccumulateGrad node strongly owns the leaf
    // tensor; if the tensor also strongly owned the node the two would form
    // an uncollectable shared_ptr cycle and leak every leaf that ever took
    // part in a graph.
    std::weak_ptr<Node> grad_accumulator_;
    uint32_t output_nr_ = 0;

public:
    explicit AutogradMeta(bool requires_grad = false) : requires_grad_(requires_grad) {}

    bool requires_grad() const override { return requires_grad_ || grad_fn_ != nullptr; }
    void set_requires_grad(bool requires_grad) override { requires_grad_ = requires_grad; }

    tensorplay::Tensor grad() const override { return grad_; }
    void set_grad(const tensorplay::Tensor& grad) override { grad_ = grad; }

    bool retains_grad() const override { return retains_grad_; }
    void set_retains_grad(bool retains_grad) override { retains_grad_ = retains_grad; }

    // tpx-specific extensions (not part of the p10 interface)
    void set_grad_fn(std::shared_ptr<Node> grad_fn) { grad_fn_ = std::move(grad_fn); }
    std::shared_ptr<Node> grad_fn() const { return grad_fn_; }

    void set_grad_accumulator(std::shared_ptr<Node> grad_accumulator) { grad_accumulator_ = std::move(grad_accumulator); }
    // Returns nullptr when the accumulator has expired (the tensor outlived
    // its graph, as with PyTorch's try_get_grad_accumulator).
    std::shared_ptr<Node> grad_accumulator() const { return grad_accumulator_.lock(); }

    uint32_t output_nr() const { return output_nr_; }
    void set_output_nr(uint32_t output_nr) { output_nr_ = output_nr; }

    // Accumulate a gradient into grad_, reusing storage when it is safe to do
    // so in-place (gradient tracking disabled and the buffer is uniquely held).
    void accum_grad(const tensorplay::Tensor& grad);
};

} // namespace tpx
} // namespace tensorplay