#pragma once

#include "AutogradMetaBase.h"
#include "Tensor.h"
#include "Macros.h"
#include <memory>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace tensorplay {
namespace tpx {

class Node;

enum class CreationMeta : uint8_t {
    DEFAULT,
    IN_CUSTOM_FUNCTION,
    MULTI_OUTPUT_NODE,
    NO_GRAD_MODE,
    INFERENCE_MODE,
};

// Concrete autograd metadata, attached to a p10 TensorImpl through the
class TENSORPLAY_API AutogradMeta : public AutogradMetaBase {
private:
    bool requires_grad_ = false;
    bool retains_grad_ = false;
    tensorplay::Tensor grad_;
    std::shared_ptr<Node> grad_fn_;
    // NB: weak reference. The AccumulateGrad node strongly owns the leaf
    // tensor; if the tensor also strongly owned the node the two would form
    // an uncollectable shared_ptr cycle and leak every leaf that ever took
    // part in a graph.
    std::weak_ptr<Node> grad_accumulator_;
    uint32_t output_nr_ = 0;

    bool has_view_info_ = false;
    tensorplay::Tensor view_base_;
    std::vector<int64_t> view_sizes_;
    std::vector<int64_t> view_strides_;
    int64_t view_storage_offset_ = 0;
    uint32_t attr_version_ = 0;
    CreationMeta creation_meta_ = CreationMeta::DEFAULT;
    std::function<tensorplay::Tensor(const tensorplay::Tensor&)> view_fn_;
    mutable std::mutex view_mutex_;

public:
    explicit AutogradMeta(bool requires_grad = false) : requires_grad_(requires_grad) {}

    bool requires_grad() const override {
        return requires_grad_ || grad_fn_ != nullptr ||
            (has_view_info_ && view_base_.defined() && view_base_.requires_grad());
    }
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
    std::shared_ptr<Node> grad_accumulator() const { return grad_accumulator_.lock(); }

    uint32_t output_nr() const { return output_nr_; }
    void set_output_nr(uint32_t output_nr) { output_nr_ = output_nr; }

    void set_view_info(
        const tensorplay::Tensor& view,
        const tensorplay::Tensor& base,
        CreationMeta creation_meta,
        std::function<tensorplay::Tensor(const tensorplay::Tensor&)> view_fn = {}) {
        has_view_info_ = true;
        view_base_ = base;
        view_sizes_ = static_cast<std::vector<int64_t>>(view.shape());
        view_strides_ = view.strides();
        view_storage_offset_ = static_cast<int64_t>(
            view.unsafeGetTensorImpl()->storage_offset());
        attr_version_ = view.unsafeGetTensorImpl()->version();
        creation_meta_ = creation_meta;
        view_fn_ = std::move(view_fn);
    }

    bool has_view_info() const { return has_view_info_; }
    const tensorplay::Tensor& view_base() const { return view_base_; }
    const std::vector<int64_t>& view_sizes() const { return view_sizes_; }
    const std::vector<int64_t>& view_strides() const { return view_strides_; }
    int64_t view_storage_offset() const { return view_storage_offset_; }
    uint32_t attr_version() const { return attr_version_; }
    void set_attr_version(uint32_t version) { attr_version_ = version; }
    CreationMeta creation_meta() const { return creation_meta_; }
    const std::function<tensorplay::Tensor(const tensorplay::Tensor&)>& view_fn() const {
        return view_fn_;
    }
    bool has_view_fn() const { return static_cast<bool>(view_fn_); }
    std::mutex& view_mutex() const { return view_mutex_; }

    // Accumulate a gradient into grad_, reusing storage when it is safe to do
    // so in-place (gradient tracking disabled and the buffer is uniquely held).
    void accum_grad(const tensorplay::Tensor& grad);
};

} // namespace tpx
} // namespace tensorplay
